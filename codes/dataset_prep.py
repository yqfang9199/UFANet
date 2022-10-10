# -*- coding: UTF-8 -*-
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random


class Dataset_MMD(Dataset):
    def __init__(self, data_file, label_file, transform=None):
        self.data_file = data_file
        self.label_file = label_file
        self.transform = transform

    def __len__(self):
        subj_num = np.load(self.data_file).shape[0]
        return subj_num

    def __getitem__(self, idx):
        data = np.load(self.data_file)[idx, :, :, :, :]
        label = np.load(self.label_file)[idx]

        data = torch.from_numpy(data)
        label = torch.tensor(label)

        # return dictionary
        sample = {'data': data, 'label': label}
        return sample
