from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import math
import torch
from dataset_prep import Dataset_MMD
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
from net.st_gcn import pre_Model, Model
import random
import numpy as np
from sklearn.metrics import roc_auc_score


def step_decay(epoch, learning_rate, drop, epochs_drop):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train_ddcnet(epoch, model, learning_rate, source_loader, target_loader, drop, epochs_drop):
    """
    train source and target domain on ddcnet
    :param epoch: current training epoch
    :param model: defined ddcnet
    :param learning_rate: initial learning rate
    :param source_loader: source loader
    :param target_loader: target train loader
    :return:
    """
    log_interval = 1  # original: 10
    LEARNING_RATE = step_decay(epoch, learning_rate, drop, epochs_drop)
    print(f'Learning Rate: {LEARNING_RATE}')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_DECAY)
    clf_criterion = nn.BCELoss()

    model.train()

    train_correct = 0
    train_loss = 0
    TN = 0
    FP = 0
    FN = 0
    TP = 0

    tr_auc_y_gt = []
    tr_auc_y_pred = []

    len_dataloader = min(len(source_loader), len(target_loader))
    for step, (source_sample_batch, target_sample_batch) in enumerate(zip(source_loader, target_loader)):
        optimizer.zero_grad()
        source_data_batch = source_sample_batch['data'].to(device).float()
        source_label_batch = source_sample_batch['label'].to(device).float()
        target_data_batch = target_sample_batch['data'].to(device).float()

        source_preds, mmd_loss = model(source_data_batch, target_data_batch)

        clf_loss = clf_criterion(source_preds[:, 0], source_label_batch)
        loss = clf_loss + mmd_loss * 1
        loss.backward()
        optimizer.step()
        train_loss += loss

        source_pred_cpu = source_preds.data.cpu().numpy() > 0.5
        train_correct += sum(source_pred_cpu[:, 0] == source_label_batch.cpu().numpy())  # correct num in one batch

        TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(source_label_batch.cpu().numpy(), source_pred_cpu[:, 0], labels=[0, 1]).ravel()
        TN += TN_tmp
        FP += FP_tmp
        FN += FN_tmp
        TP += TP_tmp

        # tr auc
        tr_auc_y_gt.extend(source_label_batch.cpu().numpy())
        tr_auc_y_pred.extend(source_preds.detach().cpu().numpy())

        if (step + 1) % log_interval == 0:
            print("Train Epoch [{:4d}/{}] Step [{:2d}/{}]: src_cls_loss={:.6f}, mmd_loss={:.6f}, loss={:.6f}".format(
                    epoch, TRAIN_EPOCHS, step + 1, len_dataloader, clf_loss.data, mmd_loss.data, loss.data))

    train_loss /= len_dataloader
    train_acc = (TP+TN) / (TP + FP + TN + FN)
    train_AUC = roc_auc_score(tr_auc_y_gt, tr_auc_y_pred)

    print('Train set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), train_AUC: {:.5}'.format(
        train_loss, train_correct, (len_dataloader * BATCH_SIZE), train_acc, train_AUC))

    # save checkpoint.pth, save train loss and acc to a txt file
    if epoch == 100:
        torch.save(model.state_dict(), SAVE_PATH + 'fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth')
    with open(SAVE_PATH + 'fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
        f.write('epoch {}, loss {:.5}, train acc {:.5}, train_AUC {:.5}\n'.format(epoch, train_loss, train_acc, train_AUC))


def test_ddcnet(model, target_loader) -> object:
    """
    test target data on fine-tuned alexnet
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """
    clf_criterion = nn.BCELoss()

    model.eval()
    test_loss = 0
    test_correct = 0
    TN = 0
    FP = 0
    FN = 0
    TP = 0

    te_auc_y_gt = []
    te_auc_y_pred = []

    for step, test_sample_batch in enumerate(target_loader):
        test_data_batch = test_sample_batch['data'].to(device).float()
        test_label_batch = test_sample_batch['label'].to(device).float()

        test_preds, _ = model(test_data_batch, test_data_batch)  # source and target share the encoder
        test_loss += clf_criterion(test_preds[:, 0], test_label_batch)

        test_pred_cpu = test_preds.data.cpu().numpy() > 0.5
        test_correct += sum(test_pred_cpu[:, 0] == test_label_batch.cpu().numpy())
        TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(test_label_batch.cpu().numpy(), test_pred_cpu[:, 0], labels=[0, 1]).ravel()
        TN += TN_tmp
        FP += FP_tmp
        FN += FN_tmp
        TP += TP_tmp
        te_auc_y_gt.extend(test_label_batch.cpu().numpy())
        te_auc_y_pred.extend(test_preds.detach().cpu().numpy())

    test_loss /= len(target_loader)

    TPR = TP / (TP + FN)  # Sensitivity
    TNR = TN / (TN + FP)  # Specificity
    PPV = TP / (TP + FP)  # Precision
    test_acc = (TP+TN) / (TP + FP + TN + FN)

    test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_pred)

    print('Test set: Correct_num: {}, test_acc: {:.4f}, test_AUC: {:.4f}, TPR: {:.4f}, TNR: {:.4f}, PPV:{:.4f}\n'.format(
        test_correct, test_acc, test_AUC, TPR, TNR, PPV))

    # save test loss and acc to a txt file
    with open(SAVE_PATH + 'fold_' + str(fold) + '_test_results.txt', 'a') as f:
        f.write('epoch {}, test_acc {:.5}, test_AUC {:.5}, TPR {:.5}, TNR {:.5}, PPV {:.5}\n'.format(epoch, test_acc, test_AUC, TPR, TNR, PPV))


if __name__ == '__main__':

    ROOT_PATH = '../data/'
    SAVE_PATH = '../codes/checkpoints/'

    BATCH_SIZE = 10  # original: 128
    TRAIN_EPOCHS = 100
    L2_DECAY = 5e-4
    MOMENTUM = 0.9
    learning_rate = 0.01
    drop = 0.5
    epochs_drop = 30.0

    for fold in [0]:
        seed = fold
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Load data begin:')
        source_data = ROOT_PATH + 'src_data.npy'  # (SrcNum, 1, T, NodeNum, 1)
        source_label = ROOT_PATH + 'src_lbl.npy'  # (SrcNum,)
        target_data = ROOT_PATH + 'tgt_data.npy'  #  (TgtNum, 1, T, NodeNum, 1)
        target_label = ROOT_PATH + 'tgt_lbl.npy'  # (TgtNum,)

        source_dataset = Dataset_MMD(source_data, source_label, transform=None)
        source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        target_dataset = Dataset_MMD(target_data, target_label, transform=None)
        target_train_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        target_test_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        print('Load data complete!')
        # load pretrained model
        ddcnet_pre = pre_Model(in_channels=1, num_class=1, edge_importance_weighting=True)
        ddcnet_pre.load_state_dict(torch.load('../codes/checkpoints_pretrain/fold_'+str(fold)+'_epoch_20.pth', map_location=device))
        ddcnet_pre_dict = ddcnet_pre.state_dict()
        # construct new model
        ddcnet = Model(in_channels=1, num_class=1, edge_importance_weighting=True)
        ddcnet_dict = ddcnet.state_dict()
        ddcnet_pre_dict = {k: v for k, v in ddcnet_pre_dict.items() if k in ddcnet_dict}
        ddcnet_dict.update(ddcnet_pre_dict)
        ddcnet.load_state_dict(ddcnet_dict)
        ddcnet.to(device)

        with open(SAVE_PATH + 'fold_' + str(fold) + '_train_loss_and_acc.txt', 'a') as f:
            f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch: {}\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))
        with open(SAVE_PATH + 'fold_' + str(fold) + '_test_results.txt', 'a') as f:
            f.write('total_epoch: {}, batch_size: {}, initial_lr {:.8}, drop_lr: {:.5}, drop_lr_per_epoch: {}\n'.format(TRAIN_EPOCHS, BATCH_SIZE, learning_rate, drop, epochs_drop))

        for epoch in range(1, TRAIN_EPOCHS + 1):
            print(f'Train Epoch {epoch}:')
            train_ddcnet(epoch, ddcnet, learning_rate, source_loader, target_train_loader, drop, epochs_drop)
            correct = test_ddcnet(ddcnet, target_test_loader)
