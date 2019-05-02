from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import plotly
import pickle
import sys


###############################################################
#                     CIFAR NET
###############################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.conv2_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

###############################################################
#              CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(61326, 1000,bias = False)
        self.fc2 = nn.Linear(1000,200,bias = False)
        self.fc3 = nn.Linear(200,1000,bias = False)
        self.fc4 = nn.Linear(1000, 61326,bias = False)
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 61326)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


###############################################################
#           SPLIT   CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################

# class SPLIT_CIFAR_CAE_TWO_SKILLS(nn.Module):
#     def __init__(self):
#         super(SPLIT_CIFAR_CAE_TWO_SKILLS, self).__init__()
#         self.fc1 = nn.Linear(20442, 20, bias=True)
#         self.fc2 = nn.Linear(20, 20442, bias=True)
#         self.relu = nn.Tanh()
#         self.sigmoid = nn.Tanh()

#     def encoder(self, x):
#         h1 = self.relu(self.fc1(x.view(-1, 20442)))
#         return h1

#     def decoder(self, z):
#         h2 = self.sigmoid(self.fc2(z))
#         return h2

#     def forward(self, x):
#         h1 = self.encoder(x)
#         h2 = self.decoder(h1)
#         return h1, h2


class SPLIT_CIFAR_CAE_TWO_SKILLS(nn.Module):
    def __init__(self):
        super(SPLIT_CIFAR_CAE_TWO_SKILLS, self).__init__()
        self.fc1 = nn.Linear(20442, 1000,bias = True)
        self.fc2 = nn.Linear(1000,50,bias = True)
        self.fc3 = nn.Linear(50,1000,bias = True)
        self.fc4 = nn.Linear(1000, 20442,bias = True)
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 20442)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
