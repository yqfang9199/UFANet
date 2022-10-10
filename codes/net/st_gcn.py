import torch
import torch.nn as nn
import numpy as np
import mmd
from net.utils.tgcn import ConvTemporalGraphical


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` wheretorch.nn
            :math:`N` is batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, edge_importance_weighting, **kwargs) -> object:
        super().__init__()
        A = np.load('../data/adj_matrix.npy')

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)

        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks (number of layers, final output features, kernel size)
        spatial_kernel_size = A.size(0)  # spatial_kernel_size = 1
        temporal_kernel_size = 11  # update temporal kernel size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # kernel_size = (11, 1)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}  # kwargs0: {}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 16, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.cls_fcn1 = nn.Conv2d(3200, 1024, kernel_size=1)
        self.cls_fcn2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.cls_fcn3 = nn.Conv2d(512, 64, kernel_size=1)
        self.cls_fcn4 = nn.Conv2d(64, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, source, target):

        N, C, T, V, M = source.size()  # (bs, 1, T, NodeNum, 1)
        source = source.permute(0, 4, 3, 1, 2).contiguous()
        source = source.view(N * M, V * C, T)
        source = self.data_bn(source.float())
        source = source.view(N, M, V, C, T)
        source = source.permute(0, 1, 3, 4, 2).contiguous()
        source = source.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            source, _ = gcn(source, self.A * importance)

        source = source.mean(axis=3)
        source = source.view(source.size(0), -1)

        #  prediction
        source_0 = source.view(N, M, -1, 1, 1).mean(dim=1)
        source_1 = self.cls_fcn1(source_0)
        source_2 = self.cls_fcn2(source_1)
        source_3 = self.cls_fcn3(source_2)
        source_4 = self.cls_fcn4(source_3)
        source_5 = self.sig(source_4)

        mmd_loss = 0
        if self.training:
            N, C, T, V, M = target.size()
            target = target.permute(0, 4, 3, 1, 2).contiguous()
            target = target.view(N * M, V * C, T)
            target = self.data_bn(target.float())
            target = target.view(N, M, V, C, T)
            target = target.permute(0, 1, 3, 4, 2).contiguous()
            target = target.view(N * M, C, T, V)

            for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
                target, _ = gcn(target, self.A * importance)

            target = target.mean(axis=3)
            target = target.view(target.size(0), -1)

            #  prediction
            target_0 = target.view(N, M, -1, 1, 1).mean(dim=1)
            target_1 = self.cls_fcn1(target_0)
            target_2 = self.cls_fcn2(target_1)
            target_3 = self.cls_fcn3(target_2)
            target_4 = self.cls_fcn4(target_3)
            target_5 = self.sig(target_4)

            # linear mmd
            mmd_loss += (mmd.mmd_linear(source_1.squeeze(), target_1.squeeze()) + mmd.mmd_linear(source_2.squeeze(), target_2.squeeze()) + mmd.mmd_linear(source_3.squeeze(), target_3.squeeze()))

        result = source_5.view(source_5.size(0), -1)

        return result, mmd_loss


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # (11, 1)
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()
        # print("Dropout={}".format(dropout))
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  # padding = (5, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])  # kernel_size[1] = 1

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),  # kernel_size[0] = 11
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class pre_Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` wheretorch.nn
            :math:`N` is batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, edge_importance_weighting, **kwargs) -> object:
        super().__init__()
        A = np.load('../data/adj_matrix.npy')

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)

        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks (number of layers, final output features, kernel size)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 11
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}  # kwargs0: {}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 16, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.cls_fcn1 = nn.Conv2d(3200, 1024, kernel_size=1)
        self.cls_fcn2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.cls_fcn3 = nn.Conv2d(512, 64, kernel_size=1)
        self.cls_fcn4 = nn.Conv2d(64, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, source):

        N, C, T, V, M = source.size()
        source = source.permute(0, 4, 3, 1, 2).contiguous()
        source = source.view(N * M, V * C, T)
        source = self.data_bn(source.float())
        source = source.view(N, M, V, C, T)
        source = source.permute(0, 1, 3, 4, 2).contiguous()
        source = source.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            source, _ = gcn(source, self.A * importance)

        source = source.mean(axis=3)
        source = source.view(source.size(0), -1)

        #  prediction
        source = source.view(N, M, -1, 1, 1).mean(dim=1)
        source = self.cls_fcn1(source)
        source = self.cls_fcn2(source)
        source = self.cls_fcn3(source)
        source = self.cls_fcn4(source)
        source = self.sig(source)
        result = source.view(source.size(0), -1)

        return result
