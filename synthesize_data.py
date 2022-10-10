import random
import numpy as np

np.random.seed(1)
random.seed(1)

data_dir = '../data/'

NodeNum = 116  # number of brain regions / ROIs
T = 200  # time points of a fMRI scan

# construct data and label of source and target domains
src_data1_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
src_data2_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
src_data3_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
src_data4_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
src_data5_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
src_data1_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
src_data2_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
src_data3_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
src_data4_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
src_data5_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
src_data = np.concatenate((src_data1_lbl1, src_data2_lbl1, src_data3_lbl1, src_data4_lbl1, src_data5_lbl1, src_data1_lbl0, src_data2_lbl0, src_data3_lbl0, src_data4_lbl0, src_data5_lbl0), axis=0)
src_lbl = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
np.save(data_dir + 'src_data.npy', src_data)  # (SrcNum, 1, NodeNum, T, 1)
np.save(data_dir + 'src_lbl.npy', src_lbl)

tgt_data1_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
tgt_data2_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
tgt_data3_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
tgt_data4_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
tgt_data5_lbl1 = np.random.rand(1, 1, T, NodeNum, 1)
tgt_data1_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
tgt_data2_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
tgt_data3_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
tgt_data4_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3
tgt_data5_lbl0 = np.random.rand(1, 1, T, NodeNum, 1) + 0.3

tgt_data = np.concatenate((tgt_data1_lbl1, tgt_data2_lbl1, tgt_data3_lbl1, tgt_data4_lbl1, tgt_data5_lbl1, tgt_data1_lbl0, tgt_data2_lbl0, tgt_data3_lbl0, tgt_data4_lbl0, tgt_data5_lbl0), axis=0)
tgt_lbl = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
np.save(data_dir + 'tgt_data.npy', tgt_data)  # (TgtNum, 1, NodeNum, T, 1)
np.save(data_dir + 'tgt_lbl.npy', tgt_lbl)

# construct adj_matrix (shared by source and target domain)
src_data_reshape = np.concatenate((src_data1_lbl1.squeeze().transpose(), src_data2_lbl1.squeeze().transpose(), src_data3_lbl1.squeeze().transpose(), src_data4_lbl1.squeeze().transpose(), src_data5_lbl1.squeeze().transpose(), src_data1_lbl0.squeeze().transpose(), src_data2_lbl0.squeeze().transpose(), src_data3_lbl0.squeeze().transpose(), src_data4_lbl0.squeeze().transpose(), src_data5_lbl0.squeeze().transpose()), axis=1)
tgt_data_reshape = np.concatenate((tgt_data1_lbl1.squeeze().transpose(), tgt_data2_lbl1.squeeze().transpose(), tgt_data3_lbl1.squeeze().transpose(), tgt_data4_lbl1.squeeze().transpose(), tgt_data5_lbl1.squeeze().transpose(), tgt_data1_lbl0.squeeze().transpose(), tgt_data2_lbl0.squeeze().transpose(), tgt_data3_lbl0.squeeze().transpose(), tgt_data4_lbl0.squeeze().transpose(), tgt_data5_lbl0.squeeze().transpose()), axis=1)
src_tgt_data = np.concatenate((src_data_reshape, tgt_data_reshape), axis=1)  # (NodeNum, src&tgtNum * T)

A = np.zeros((NodeNum, NodeNum))
for i in range(NodeNum):
    for j in range(i, NodeNum):
        if i == j:
            A[i][j] = 0
        else:
            A[i][j] = abs(np.corrcoef(src_tgt_data[i, :], src_tgt_data[j, :])[0][1])  # get value from corrcoef matrix
            A[j][i] = A[i][j]
np.save(data_dir + 'adj_matrix.npy', A)
