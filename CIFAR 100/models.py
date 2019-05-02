from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
#import plotly
import pickle
import sys


###############################################################
#                      HELPER FUNCTIONS
###############################################################
def dictionary_save_obj(obj, name ):
    with open('Distributions_mu_logvar/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def dictionary_load_obj(name ):
    with open('Distributions_mu_logvar/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

###############################################################
#                      MNIST MODEL 
###############################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
#                     CIFAR NET
###############################################################
class CIAFR_Net(nn.Module):
    def __init__(self):
        super(CIAFR_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



###############################################################
#                   MNIST- CAE -CONTACITED AUTO ENCODER
###############################################################
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(62006, 2000)
        self.fc12 = nn.Linear(2000, 200)
        self.fc22 = nn.Linear(200, 2000)
        self.fc23 = nn.Linear(2000, 10000)
        self.fc2 = nn.Linear(10000, 62006)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Tanh()
        self.dropout = nn.Dropout(0.05)

    def encoder(self, x):
        h11 = self.relu(self.fc1(x.view(-1, 62006)))
        #h11= F.dropout(h11, training=self.training)
        h1=self.relu(self.fc12(h11))
        return h1

    def decoder(self, z):
        h22 = self.sigmoid(self.fc22(z))
        #h22 = F.dropout(h22, training=self.training)
        h22 = self.sigmoid(self.fc23(h22))
        h2 = self.sigmoid(self.fc2(h22))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
# class CAE(nn.Module):
#     def __init__(self):
#         super(CAE, self).__init__()
#         self.fc1 = nn.Linear(21432, 200)
#         self.fc2 = nn.Linear(200, 21432)
#         self.relu = nn.Tanh()
#         self.sigmoid = nn.Tanh()

#     def encoder(self, x):
#         h1 = self.relu(self.fc1(x.view(-1, 21432)))
#         return h1

#     def decoder(self, z):
#         h2 = self.sigmoid(self.fc2(z))
#         return h2

#     def forward(self, x):
#         h1 = self.encoder(x)
#         h2 = self.decoder(h1)
#         return h1, h2

###############################################################
#              CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################

class CIFAR_CAE(nn.Module):
    def __init__(self):
        super(CIFAR_CAE, self).__init__()
        self.fc1 = nn.Linear(62006, 600, bias=False)
        self.fc11 = nn.Linear(62006, 600, bias=False)
        self.fc2 = nn.Linear(600, 62006, bias=False)
        self.fc21 = nn.Linear(62006, 600, bias=False)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 62006)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

###############################################################
#              CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################

class CIFAR_CAE_TWO_SKILLS(nn.Module):
    def __init__(self):
        super(CIFAR_CAE_TWO_SKILLS, self).__init__()
        self.fc1 = nn.Linear(61326, 400, bias=True)
        self.fc2 = nn.Linear(400, 61326, bias=True)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 61326)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

###############################################################
#           SPLIT   CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################

class SPLIT_CIFAR_CAE_TWO_SKILLS(nn.Module):
    def __init__(self):
        super(SPLIT_CIFAR_CAE_TWO_SKILLS, self).__init__()
        self.fc1 = nn.Linear(20669, 2000)
        self.fc2 = nn.Linear(2000,50)
        self.fc3 = nn.Linear(50,2000)
        self.fc4 = nn.Linear(2000, 20669)
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 20669)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.fc4(h2)

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
###############################################################
#           MULTI  MNIST CIFAR---CAE -CONTACITED AUTO ENCODER
###############################################################

class MULTI_CIFAR_CAE_TWO_SKILLS(nn.Module):
    def __init__(self):
        super(MULTI_CIFAR_CAE_TWO_SKILLS, self).__init__()
        self.fc1 = nn.Linear(21432, 400, bias=True)
        self.fc2 = nn.Linear(400, 21432, bias=True)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 21432)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2
###############################################################
#                    CAE -CONTACITED VARIATIONAL AUTO ENCODER
###############################################################
class VAE_CAE(nn.Module):
    def __init__(self):
        super(VAE_CAE, self).__init__()
        self.fc1 = nn.Linear(21432, 200, bias=False)
        self.fc11 = nn.Linear(21432, 200, bias=False)
        self.fc2 = nn.Linear(200, 21432, bias=False)
        self.fc21 = nn.Linear(21432, 200, bias=False)
        self.relu = nn.Tanh()
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 21432)))
        h2 = self.relu(self.fc11(x.view(-1, 21432)))
        return h1,h2

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2


    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, x):
        h1,h2 = self.encoder(x)
        z = self.reparameterize(h1, h2)
        h3 = self.decoder(z)
        return z,h3,h2




    
