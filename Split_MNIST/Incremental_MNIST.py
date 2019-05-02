##############################################################################################################################
# Aim of the Experiment: Incrmental MNIST For Fixed Iterations
# Output: All the collected data for individual skill is present in the folder -- Collected_Data
# Task Nets: All the trained Inc-MNIST weights saved in the folder -- task_net_models
# Test & Train Image Samples: All the train,test Inc-MNIST task images sample saved in the folder -- Test_Train_Images
# Plots : If any plots you can save in folder -- Inc_MNIST_plots (Currently Not Used)
# Conclusion : Working very well you can see all the outputs in Visoom or in folder Permuated_MNIST_plots with output.pdf
##############################################################################################################################

from __future__ import print_function
import argparse
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
import pickle
from numpy import dot
from numpy.linalg import norm
import visdom
vis = visdom.Visdom()
vis.delete_env('Inc_MNIST_Fixed_Iter') #If you want to clear all the old plots for this python Experiments.Resets the Environment
vis = visdom.Visdom(env='Inc_MNIST_Fixed_Iter')
# Execution flags
Flags = {}
Flags['mnist_train'] =  True #This will run the mnist experiment then run the incremental approach
#else it will load the existing weights in the directory and run the experiment
Flags['regular_mnist_train']= False # For choosing the  incase
Flags['normalize']= True # For choosing the  incase

nSamples = 10
nBiased = min(nSamples,10)
trainBias = 0.5
minReps = 1
nReps = 20
stage=0
addRepsTotal = nReps*minReps
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--vaelr', type=float, default=0.001, metavar='VAE_LR',
                    help='vae learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--schedule', type=int, nargs='+', default=[20,80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
trainset=datasets.MNIST(root='../data', train=True,download=True, transform=transform)
testset = datasets.MNIST(root='../data', train=False,download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True,**kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,shuffle=True, **kwargs)
global_trainset_train_labels=trainset.train_labels
global_trainset_train_data=trainset.train_data
global_testset_test_labels=testset.test_labels
global_testset_test_data=testset.test_data
global_train_loader=train_loader
global_test_loader=test_loader


train_Indexs=[[]] * 10
test_Indexs = [[]] * 10
Skill_Mu=[]
#####################################################################################################################
# Calculating all the Train Indexes for Each Digit
#####################################################################################################################
def CALUCULATE_TRAIN_INDEXES():
    train_Indexs[0]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 0]
    train_Indexs[1]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 1]
    train_Indexs[2]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 2]
    train_Indexs[3]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 3]
    train_Indexs[4]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 4]
    train_Indexs[5]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 5]
    train_Indexs[6]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 6]
    train_Indexs[7]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 7]
    train_Indexs[8]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 8]
    train_Indexs[9]=[num for num in range(0,len(trainset.train_labels)) if trainset.train_labels[num]  == 9]
#####################################################################################################################
# Calculating all the Test Indexes for Each Digit
#####################################################################################################################
def CALUCULATE_TEST_INDEXES():
    test_Indexs[0]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 0]
    test_Indexs[1]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 1]
    test_Indexs[2]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 2]
    test_Indexs[3]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 3]
    test_Indexs[4]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 4]
    test_Indexs[5]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 5]
    test_Indexs[6]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 6]
    test_Indexs[7]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 7]
    test_Indexs[8]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 8]
    test_Indexs[9]=[num for num in range(0,len(testset.test_labels)) if testset.test_labels[num]  == 9]
######################################################################################################
# RELOD THE DATASET WHEN NEEDED
######################################################################################################
def RELOAD_DATASET():
    #print("RELOADING THE DATA")
    global train_loader,test_loader,trainset,testset

    train_loader=global_train_loader
    test_loader=global_test_loader

    trainset.train_data=global_trainset_train_data
    testset.test_data=global_testset_test_data

    trainset.train_labels=global_trainset_train_labels
    testset.test_labels=global_testset_test_labels
######################################################################################################
#MODIFYING DATSET FOR INDIVIDUAL CLASSES
######################################################################################################
def load_individual_class(postive_class,negative_classes):
    RELOAD_DATASET()
    global train_loader,test_loader,train_Indexs,test_Indexs

    index_train_postive=[]
    index_test_postive=[]
    index_train_negative=[]
    index_test_negative=[]
    #print(postive_class)
    for i in range(0,len(postive_class)):
        index_train_postive=index_train_postive+train_Indexs[postive_class[i]]
        index_test_postive=index_test_postive+test_Indexs[postive_class[i]]
        
    for i in range(0,len(negative_classes)):
        index_train_negative=index_train_negative+train_Indexs[negative_classes[i]][0:int(0.2*(len(train_Indexs[negative_classes[i]])))]
        index_test_negative=index_test_negative+test_Indexs[negative_classes[i]][0:int(0.2*(len(test_Indexs[negative_classes[i]])))]
    
    index_train=index_train_postive+index_train_negative
    index_test=index_test_postive+index_test_negative
    modified_train_labels = [1 if (trainset.train_labels[x] in postive_class) else 0 for x in index_train]
    modified_test_labels  = [1 if (testset.test_labels[x] in postive_class) else 0 for x in index_test]
    trainset.train_labels=modified_train_labels#train set labels
    trainset.train_data=trainset.train_data[index_train]#train set data
    testset.test_labels=modified_test_labels#testset labels
    testset.test_data=testset.test_data[index_test]#testset data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=200,shuffle=True) 

    return train_loader,test_loader

CALUCULATE_TRAIN_INDEXES()
CALUCULATE_TEST_INDEXES()

#####################################################################################################################
# For Viewing the test/train images-just for confirmation
#####################################################################################################################
classes = ('0','1', '2', '3', '4','5','6','7','8','9')
def imshow(img,permuatation):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('Test_Train_Images/train_data_image'+str(permuatation)+'.png')

def imshow_test(img,permuatation):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('Test_Train_Images/test_data_image'+str(permuatation)+'.png')

def SHOW_TEST_TRAIN_IMAGES_SAMPLE(permuatation):
    # get some random training images
    global train_loader
    global test_loader
    dataiter = iter(train_loader)
    dataiter_test = iter(test_loader)
    print("length of testset loader is",len(test_loader))
    print("length if train set loader is",len(train_loader))
    images, labels = dataiter.next()
    images_test, labels_test = dataiter_test.next()
    imshow(torchvision.utils.make_grid(images),permuatation)
    imshow_test(torchvision.utils.make_grid(images_test),permuatation)
    #img = Image.open('train_data_image.png')
    #img.show()
    #time.sleep(5)
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #img = Image.open('test_data_image.png')
    print(' '.join('%5s' % classes[labels_test[j]] for j in range(4)))

######################################################################################################
# LOADING OF VAE OUTPUT SKILL WEIGHTS BACK INTO THE MNSIT NETWORK
######################################################################################################
def loadWeights_mnsit(weights_to_load, net):
    model.conv1.weight.data = torch.from_numpy(weights_to_load[0]).cuda()
    model.conv1.bias.data =   torch.from_numpy(weights_to_load[1]).cuda()
    model.conv2.weight.data = torch.from_numpy(weights_to_load[2]).cuda()
    model.conv2.bias.data =   torch.from_numpy(weights_to_load[3]).cuda()
    model.fc1.weight.data =   torch.from_numpy(weights_to_load[4]).cuda()
    model.fc1.bias.data =     torch.from_numpy(weights_to_load[5]).cuda()
    model.fc2.weight.data =   torch.from_numpy(weights_to_load[6]).cuda()
    model.fc2.bias.data =     torch.from_numpy(weights_to_load[7]).cuda()
    return model

student_model=models.CAE().to(device)#nn.DataParallel(models.CAE().to(device))
teacher_model=models.CAE().to(device)#nn.DataParallel(models.CAE().to(device))
vae_optimizer = optim.Adam(student_model.parameters(), lr = 0.0001)
lam = 0.001
Actual_Accuracy=[]
threshold_batchid=[]
threshold_net_updates=[]
SUM=[]
Actual_task_net_weights=[]
######################################################################################################
# Adjusting Learning rate
######################################################################################################
def adjust_learning_rate(optimizer, epoch):
    global state
    #global vae_optimizer
    if epoch in args.schedule:
        state['vaelr'] *= args.gamma
        print("state lr is",state['vaelr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['vaelr']
        print("VAE OPTIMIZER IS",vae_optimizer)
######################################################################################################
# TRAIN
######################################################################################################
def train(model,train_loader,test_loader,optimizer, epochs,task_number):
    options = dict(fillarea=True,width=400,height=400,xlabel='Batch_ID(Iterations)',ylabel='Loss',title=task_number)
    acc_options = dict(fillarea=True,width=400,height=400,xlabel='Epoch',ylabel='Accuracy',title=task_number)
    win = vis.line(X=np.array([2]),Y=np.array([1]),win=task_number,name=task_number,opts=options)
    model.to(device)
    model.train()
    total_batchid=0
    task_test_accuracy=[0]
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            vis.line(X=np.array([total_batchid+batch_idx]),Y=np.array([loss.item()]),win=win,update='append')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        total_batchid=total_batchid+batch_idx
        acc=test(model,test_loader)
        acc=test(model,test_loader)
        task_test_accuracy.append(acc)
        vis.bar(X=np.array(task_test_accuracy),win='ACC'+task_number,opts=acc_options)
    return acc

######################################################################################################
# TEST
######################################################################################################
def test(model, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

######################################################################################################
# LOADING OF VAE OUTPUT SKILL WEIGHTS BACK INTO THE MNSIT NETWORK AND TEST MNIST
######################################################################################################
print("VAE SCHEDULER IS ", vae_optimizer)
def CAE_AE_TRAIN(shapes,task_samples,iterations):
    global stage
    options = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='Loss',title='CAE_skills'+str(len(task_samples)))
    options_2 = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='Accuracy',title='CAE_skills'+str(len(task_samples)))
    options_mse = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='MSE',title='CAE_MSE_skills'+str(len(task_samples)))
    options_mse_org = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='MSE_Orginal',title='CAE_MSE_WRT_Orginal_skills'+str(len(task_samples)))
    win = vis.line(X=np.array([0]),Y=np.array([0.005]),win='CAE_skills '+str(len(task_samples)),name='CAE_skills'+str(len(task_samples)),opts=options)
    win_2 = vis.line(X=np.array([0]),Y=np.array([0]),win='CAE_Acc_skills '+str(len(task_samples)),name='CAE_skills'+str(len(task_samples)),opts=options_2)
    win_mse = vis.line(X=np.array([0]),Y=np.array([0]),win='CAE_MSE_skills '+str(len(task_samples)),name='CAE_MSE_skills'+str(len(task_samples)),opts=options_mse)
    win_mse_org = vis.line(X=np.array([0]),Y=np.array([0]),win='CAE_MSE_Org_skills '+str(len(task_samples)),name='CAE_MSE_Orgskills'+str(len(task_samples)),opts=options_mse_org)
    total_resend=0
    total=len(task_samples)
    accuracies = np.zeros((iterations,len(task_samples)))
    #task_samples=task_samples.to(device)
    student_model.train()
    final_dataframe=pd.DataFrame()
    final_dataframe_1=pd.DataFrame()
    for i in range(0,len(task_samples)-len(Skill_Mu)):
        Skill_Mu.append([])
    for batch_idx in range(1,iterations):
        train_loss = 0
        randPerm = np.random.permutation(len(task_samples))
        #adjust_learning_rate(vae_optimizer, batch_idx)
        #print("after lr change",vae_optimizer)
        #randPerm,nReps = helper_functions.biased_permutation(stage+1,nBiased,trainBias,len(task_samples),minReps)
        print(randPerm)
        resend=[]
        for s in randPerm:
            #collect_data=[]
            skill=s#randint(0,len(task_samples)-1)
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill])).to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc2.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, data.view(-1, 21432), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            loss.backward()
            vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win=win,name='Loss_Skill_'+str(skill),update='append')#,opts=options_lgnd)
            vae_optimizer.step()
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
            if float(loss.data[0] * 1000000)>=1.00:
                resend.append(s)
        print("resending skills",resend)
        total_resend=total_resend+len(resend)
        for s in resend:
            skill=s
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill])).to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc2.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, 
                data.view(-1, 21432), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            loss.backward()
            vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win=win,name='Loss_Skill_'+str(skill),update='append')#,opts=options_lgnd)
            vae_optimizer.step()
        if batch_idx %1==0:
            values=0
            for i in range(0,len(task_samples)):
                collect_data_1=[]
                mu1=Skill_Mu[i][0]
                task_sample = student_model.decoder(mu1).cpu()
                sample=task_sample.data.numpy().reshape(21432)
                #print('MSE IS',mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy()))
                #mse=mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy())
                mse=mean_squared_error(sample,Variable(torch.FloatTensor(task_samples[i])).data.numpy())
                mse_orginal=mean_squared_error(sample,Actual_task_net_weights[i])
                final_weights=helper_functions.unFlattenNetwork(sample, shapes)
                model_x=loadWeights_mnsit(final_weights,model)
                if i>=6:
                    Train_loader,Test_loader=load_individual_class([i],[1,2,3,4])
                else:
                    Train_loader,Test_loader=load_individual_class([i],[6,7,8,9])
                Avg_Accuracy= test(model_x, Test_loader)
                b=torch.FloatTensor(task_samples[i]).data.numpy()
                b1=torch.FloatTensor(Actual_task_net_weights[i]).data.numpy()
                COSINE_SIMILARITY=dot(sample, b)/(norm(sample)*norm(b))
                COSINE_SIMILARITY_wrt_orginal=dot(sample, b1)/(norm(sample)*norm(b1))
                collect_data_1.extend([total,batch_idx,i,mse,mse_orginal,Avg_Accuracy,Actual_Accuracy[i],len(resend),sample,torch.FloatTensor(task_samples[i]).data.numpy(),COSINE_SIMILARITY,COSINE_SIMILARITY_wrt_orginal])
                final_dataframe_1=pd.concat([final_dataframe_1, pd.DataFrame(collect_data_1).transpose()])
                accuracies[batch_idx,i] = Avg_Accuracy
                if COSINE_SIMILARITY>0.99:
                    values=values+1
                # if len(task_samples)>6:
                #     if round(Avg_Accuracy+0.5)>=int(Actual_Accuracy[i]-1):
                #         values=values+1
                # else:
                #     if round(Avg_Accuracy+0.5)>=int(Actual_Accuracy[i]):
                #         values=values+1
                vis.line(X=np.array([batch_idx]),Y=np.array([Avg_Accuracy]),win=win_2,name='Acc_Skill_'+str(i),update='append')
                vis.line(X=np.array([batch_idx]),Y=np.array([mse]),win=win_mse,name='MSE_Skill_'+str(i),update='append')
                vis.line(X=np.array([batch_idx]),Y=np.array([mse_orginal]),win=win_mse_org,name='MSE_Org_Skill_'+str(i),update='append')#,opts=options_lgnd)
            if values==len(task_samples):
                print("########## \n Batch id is",batch_idx,"\n#########")
                threshold_batchid.append(batch_idx)
                threshold_net_updates.append((batch_idx*total)+total_resend)
                break

    stage=stage+1
    final_dataframe_1.columns=['no_of_skills','batch_idx','skill','caluclated_mse','mse_wrt_orginal','Accuracy','Actual_Accuracy','Resend_len','sample','task_sample','COSINE_SIMILARITY','COSINE_SIMILARITY_wrt_orginal']
    final_dataframe_1.to_hdf('Collected_Data/'+str(len(task_samples))+'_data','key1')
    return accuracies
 
#####################################################################################################
# FLATTEN THE MNIST WEIGHTS AND FEED IT TO VAE
######################################################################################################
def FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model):
    final_skill_sample=[]
    Flat_input,net_shapes=helper_functions.flattenNetwork(model.cpu())
    final_skill_sample.append(Flat_input)
    Actual_task_net_weights.append(Flat_input)
    if len(task_samples)==0:
        accuracies=CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,100)
    else:
        accuracies=CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,100)
    return accuracies

#####################################################################################################
# FMNSIT TRAINING
######################################################################################################
allAcc = []
for mnsit_class in range(0,10):
    task_samples=[]
    print("device is ",device)
    model = models.Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print("########## \n Threshold id is",threshold_batchid,"\n#########")
    if mnsit_class>=6:
        Train_loader,Test_loader=load_individual_class([mnsit_class],[1,2,3,4])
    else:
        Train_loader,Test_loader=load_individual_class([mnsit_class],[6,7,8,9])
    SHOW_TEST_TRAIN_IMAGES_SAMPLE(mnsit_class)
    accuracy=train(model,Train_loader,Test_loader,optimizer,3,'MNIST Skill '+str(mnsit_class))
    Actual_Accuracy.append(int(accuracy))
    print("Actual acc",Actual_Accuracy)
    torch.save(model.state_dict(),'task_net_models/mnist_digit_solver_'+str(mnsit_class)+'.pt')
    if mnsit_class==0 :#or mnsit_class==1:
        accuracies=FLATTEN_WEIGHTS_TRAIN_VAE([],model)
    else:
        for i in range(0,mnsit_class):  
            recall_memory_sample=Skill_Mu[i][0]#Tasks_MU_LOGVAR[i]['mu']#[len(Tasks_MU_LOGVAR[i]['mu'])-1]
            generated_sample=student_model.decoder(recall_memory_sample).cpu()
            task_samples.append(generated_sample)
        accuracies=FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model) 
    allAcc.append(accuracies)

    with open('Collected_Data/allAcc_'+str(mnsit_class), 'wb') as fp:
        pickle.dump(allAcc, fp)

    with open('Collected_Data/Actual_Accuracy_'+str(mnsit_class), 'wb') as fp:
        pickle.dump(Actual_Accuracy, fp)
    

    if mnsit_class>=1:
        options = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Actual_Accuracy',title='Actual_Accuracy')
        options_threshold = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Threshold_Cutoff',title='Cutoff_Threshold')
        options_threshold_net_updates = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Threshold_Cutoff',title='Cutoff_Threshold_Net_Updates')
        vis.bar(X=Actual_Accuracy,opts=options,win='acc_viz')
        vis.bar(X=threshold_batchid,opts=options_threshold,win='threshold_viz')
        vis.bar(X=threshold_net_updates,opts=options_threshold_net_updates,win='threshold_viz_net_updates')



