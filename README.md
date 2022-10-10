# STGCN
This is a code implemention of the UFA-Net method proposed in the manuscipt "Unsupervised Cross-Domain Functional MRI Adaptation for Automated Major Depressive Disorder Identification".

1. Data Construction
We construct a demo example consisting of 10 source examples and 10 target examples:

run: synthesize_data.py

src_data.shape (SrcNum, 1, T, NodeNum, 1)
src_lbl.shape (SrcNum, )
tgt_data.shape (TgtNum, 1, T, NodeNum, 1)
tgt_lbl.shape (TgtNum, )
adj_matrix.shape (NodeNum, NodeNum)

where
SrcNum: the number of subjects in the source domain
TgtNum: the number of subjects in the target domain
T: the number of time points of a fMRI scan (here is 200)
NodeNum: the number of brain nodes/ROIs (here is 116, corresponding to AAL116 atlas)

2. Model Training and Validation
This is a two-step optimization method.

2.1. using L_cls to initialize the network parameter (this step does not involve domain adaptation)
run: ../codes/main_pretrain.py
The pretrained model is saved in: ../codes/checkpoints_pretrain/

2.2. using L_cls and L_mmd to train the whole network.
run: ../codes/main_UDA.py
The classification results are saved in: ../codes/checkpoints/

