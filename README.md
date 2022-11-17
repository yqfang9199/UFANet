# UFANet

#### This is a code implemention of the UFA-Net method proposed in the manuscipt "Unsupervised Cross-Domain Functional MRI Adaptation for Automated Major Depressive Disorder Identification".

### 0. Dependencies
`torch==1.10.0`<br>
`torchvision==0.2.1`<br>
`numpy==1.21.2`<br>
`scikit_learn==1.1.3`<br>

### 1. Data Construction
We construct a demo consisting of 10 source examples and 10 target examples.

Run: `synthesize_data.py`<br>

The shape of the constructed data and label:<br>
> src_data.npy `(SrcNum, 1, T, NodeNum, 1)`<br>
> src_lbl.npy `(SrcNum, )`<br>
> tgt_data.npy `(TgtNum, 1, T, NodeNum, 1)`<br>
> tgt_lbl.npy `(TgtNum, )`<br>
> adj_matrix.npy `(NodeNum, NodeNum)`<br>

where<br>
`SrcNum` is the number of subjects in the source domain<br>
`TgtNum` is the number of subjects in the target domain<br>
`T` is the number of time points of a fMRI scan (here is 200)<br>
`NodeNum` is the number of brain nodes/ROIs (here is 116, corresponding to AAL116 atlas)<br>

### 2. Model Training and Validation
This is a two-step optimization method.

#### 2.1. Using $L_{C}$ to initialize the network parameter (not involve domain adaptation)<br>
Run: `main_pretrain.py`<br>
The pretrained model is saved in: `checkpoints_pretrain`

#### 2.2. Using $L_{C}$ and $L_{MMD}$ to train the whole network<br>
Run: `main_UDA.py`<br>
The classification results are saved in: `checkpoints`


