
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CAE_91(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(21432, 2000)
        self.fc2 = nn.Linear(2000,200)
        self.fc3 = nn.Linear(200,2000)
        self.fc4 = nn.Linear(2000, 21432)
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 21432)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(21432, 5)
        
        self.fc2 = nn.Linear(5, 21432)
        self.relu = nn.ELU()
        self.sigmoid = nn.ELU()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 21432)))
        return h1

    def decoder(self, z):
        h2 = (self.fc2(z))
        return h2

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

class CAE_multiLayer(nn.Module):
    def __init__(self):
        super(CAE_multiLayer, self).__init__()
        self.fc1 = nn.Linear(21432, 4)
        self.fc11 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc22 = nn.Linear(4, 21432)
        self.elu = nn.ELU()
        

    def encoder(self, x):
        h1 = self.elu(self.fc1(x.view(-1, 21432)))
        h2 = self.fc11(h1)
        return h2

    def decoder(self, z):
        h3 = self.elu(self.fc2(z))
        h4 = self.fc22(h3)
        return h4

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2