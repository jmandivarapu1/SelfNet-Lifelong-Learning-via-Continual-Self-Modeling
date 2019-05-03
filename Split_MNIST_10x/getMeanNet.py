import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import models
import helper_functions
import pandas as pd
import os
import sys
from scipy.stats import geom
import torchvision
import time
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from PIL import Image
import itertools
import pickle
from numpy import dot
from numpy.linalg import norm
from sklearn.utils import shuffle

def getMeanNet(start_idx, end_idx):
    num_models = end_idx - start_idx
    nets = [models.Net() for i in range(num_models)]
    
    
    net1 = models.Net()
    net2 = models.Net()
    net3 = models.Net()
    net4 = models.Net()
    net5 = models.Net()
    net6 = models.Net()
    net7 = models.Net()
    net8 = models.Net()
    net9 = models.Net()
    net10 = models.Net()

    for idx,net in enumerate(nets): 
        net_model = torch.load("task_net_models/mnist_digit_solver_"+str(idx+start_idx)+".pt")
        net.load_state_dict(net_model)



    net1_model = torch.load("task_net_models/mnist_digit_solver_0.pt")
    net1.load_state_dict(net1_model)

    net2_model = torch.load("task_net_models/mnist_digit_solver_1.pt")
    net2.load_state_dict(net2_model)

    net3_model = torch.load("task_net_models/mnist_digit_solver_2.pt")
    net3.load_state_dict(net3_model)

    net4_model = torch.load("task_net_models/mnist_digit_solver_3.pt")
    net4.load_state_dict(net4_model)

    net5_model = torch.load("task_net_models/mnist_digit_solver_4.pt")
    net5.load_state_dict(net5_model)

    net6_model = torch.load("task_net_models/mnist_digit_solver_5.pt")
    net6.load_state_dict(net6_model)

    net7_model = torch.load("task_net_models/mnist_digit_solver_6.pt")
    net7.load_state_dict(net7_model)

    net8_model = torch.load("task_net_models/mnist_digit_solver_7.pt")
    net8.load_state_dict(net8_model)

    net9_model = torch.load("task_net_models/mnist_digit_solver_8.pt")
    net9.load_state_dict(net9_model)

    net10_model = torch.load("task_net_models/mnist_digit_solver_9.pt")
    net10.load_state_dict(net10_model)

    flatNets = [[] for i in range(num_models)]
    net_shapes = []
    for idx,net in enumerate(nets):
        flatNets[idx], net_shapes = helper_functions.flattenNetwork(net)

    flat1, net_shapes=helper_functions.flattenNetwork(net1)
    flat2, net_shapes=helper_functions.flattenNetwork(net2)
    flat3, net_shapes=helper_functions.flattenNetwork(net3)
    flat4, net_shapes=helper_functions.flattenNetwork(net4)
    flat5, net_shapes=helper_functions.flattenNetwork(net5)
    flat6, net_shapes=helper_functions.flattenNetwork(net6)
    flat7, net_shapes=helper_functions.flattenNetwork(net7)
    flat8, net_shapes=helper_functions.flattenNetwork(net8)
    flat9, net_shapes=helper_functions.flattenNetwork(net9)
    flat10, net_shapes=helper_functions.flattenNetwork(net10)

    all = torch.Tensor()
    for idx, flatNet in enumerate(flatNets):
        all = torch.cat((all, torch.Tensor(flatNet).view(-1,len(flatNet))), dim=0)


    all = torch.cat((torch.Tensor([flat1]), torch.Tensor([flat2])), dim=0)
    all = torch.cat((all, torch.Tensor([flat3])), dim=0)
    all = torch.cat((all, torch.Tensor([flat4])), dim=0)
    all = torch.cat((all, torch.Tensor([flat5])), dim=0)
    all = torch.cat((all, torch.Tensor([flat6])), dim=0)
    all = torch.cat((all, torch.Tensor([flat7])), dim=0)
    all = torch.cat((all, torch.Tensor([flat8])), dim=0)
    all = torch.cat((all, torch.Tensor([flat9])), dim=0)
    all = torch.cat((all, torch.Tensor([flat10])), dim=0)
    # # print(all)

    def loadWeights_mnsit(weights_to_load, net):
        net.conv1.weight.data = torch.from_numpy(weights_to_load[0]).cuda()
        net.conv1.bias.data =   torch.from_numpy(weights_to_load[1]).cuda()
        net.conv2.weight.data = torch.from_numpy(weights_to_load[2]).cuda()
        net.conv2.bias.data =   torch.from_numpy(weights_to_load[3]).cuda()
        net.fc1.weight.data =   torch.from_numpy(weights_to_load[4]).cuda()
        net.fc1.bias.data =     torch.from_numpy(weights_to_load[5]).cuda()
        net.fc2.weight.data =   torch.from_numpy(weights_to_load[6]).cuda()
        net.fc2.bias.data =     torch.from_numpy(weights_to_load[7]).cuda()
        return net

    mean = torch.mean(all, dim=0)
    meanNet = models.Net()

    mean_weights=helper_functions.unFlattenNetwork(mean.data.numpy(), net_shapes)
    meanNet=loadWeights_mnsit(mean_weights,meanNet)
    torch.save(meanNet.state_dict(),'meanNet.pt')


