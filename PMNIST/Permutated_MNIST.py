##############################################################################################################################
# Aim of the Experiment: Permutated MNIST For Fixed Iterations
# Output: All the collected data for individual skill is present in the folder -- Collected_Data
# Task Nets: All the trained Permutated MNIST weights saved in the folder -- task_net_models
# Test & Train Image Samples: All the train,test Permutated MNIST task images sample saved in the folder -- Test_Train_Images
# Plots : If any plots you can save in folder -- Permuated_MNIST_plots (Currently Not Used)
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
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import models
import helper_functions
import pandas as pd
import os
import sys
from scipy.stats import geom
import torchvision
import time
from PIL import Image
import pickle
from numpy import dot
from numpy.linalg import norm
import visdom
vis = visdom.Visdom()
Epoch='Cosine_3_'
vis.delete_env('Permutated_MNIST_Threshold_'+Epoch+'_epoch') #If you want to clear all the old plots for this python Experiments.Resets the Environment
vis = visdom.Visdom(env='Permutated_MNIST_Threshold_'+Epoch+'_epoch')
plt.rcParams["figure.figsize"] = (12,10)
plt.rcParams["font.size"] = 16


# Execution flags
Flags = {}
Flags['mnist_train'] =  True #This will run the mnist experiment then run the incremental approach
#else it will load the existing weights in the directory and run the experiment
Flags['regular_mnist_train']= False # For choosing the  incase
Flags['load_mnist_and_train_cae']=False
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
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
#idx_permute = torch.from_numpy(np.random.permutation(784), dtype=torch.int64)
# idx_permute = [np.random.permutation(28**2) for _ in range(1)] 
# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
#               transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28) )])
              
              
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
trainset=datasets.MNIST(root='../data', train=True,download=True, transform=transform)
testset = datasets.MNIST(root='../data', train=False,download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True,**kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,shuffle=True, **kwargs)



######################################################################################################
# RELOD THE DATASET WHEN NEEDED
######################################################################################################
def RELOAD_DATASET(ptransform):
    print("RELOADING THE DATA")
    global train_loader,test_loader
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
              transforms.Lambda(lambda x: x.view(-1)[ptransform].view(1, 28, 28) )])
    trainset=datasets.MNIST(root='../data', train=True,download=True, transform=transform)
    testset = datasets.MNIST(root='../data', train=False,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True,**kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,shuffle=True, **kwargs)
    return train_loader,test_loader


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

print("device is ",device)
#model = models.Net().to(device)
#model_reset= models.Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
student_model=models.CAE().to(device)#nn.DataParallel(models.CAE().to(device))
teacher_model=models.CAE().to(device)#nn.DataParallel(models.CAE().to(device))
vae_optimizer = optim.Adam(student_model.parameters(), lr = 0.0001)
lam = 0.001
Actual_Accuracy=[]
threshold_batchid=[]
threshold_net_updates=[]
Actual_task_net_weights=[]
# win = vis.line(
# X=np.array([0]),
# Y=np.array([0]),
# win="test",
# name='Line1',
# )
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
#                                       PLOTTING                                                     #
######################################################################################################
def plotAccuracies(allAcc):
    for s,acc in enumerate(allAcc):
        x = np.array(acc)
        f = plt.figure()
        for i in range(x.shape[1]):
            plt.plot(x[:,i],'.-', markersize=5, label='Skill: '+str(i))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Encoded Skills -- Stage ' + str(s))
        plt.legend(loc='lower right')
        plt.show()    
        f.savefig('./Permuated_MNIST_plots_1/acc_'+str(s)+'.pdf', bbox_inches='tight')
        plt.close()


def plotAccuracyDiff(allAcc,Actual_Accuracies):
    for s,acc in enumerate(allAcc):
        x = np.array(acc)
        f = plt.figure()
        for i in range(x.shape[1]):
            plt.plot(Actual_Accuracies[i] - x[:,i],'.-', markersize=5, 
                label='Skill: '+str(i))
        plt.xlabel('Iterations')
        plt.ylabel('True Acc - Reconst. Acc')
        plt.title('Encoded Skills -- Stage ' + str(s))
        plt.legend(loc='upper right')
        plt.show()    
        f.savefig('./Permuated_MNIST_plots_1/acc_diff_'+str(s)+'.pdf', bbox_inches='tight')
        plt.close()


def plotAccuracyDiffPoint(allAcc,Actual_Accuracies,testPoint=25):
    for s,acc in enumerate(allAcc):
        x = np.array(acc)
        f = plt.figure()
        for i in range(x.shape[1]):
            plt.plot(i,Actual_Accuracies[i] - x[testPoint,i],'.-', markersize=10, 
                label='Skill: '+str(i))
        plt.xlabel('Iterations')
        plt.ylabel('True Acc - Reconst. Acc')
        plt.title('Encoded Skills -- Stage ' + str(s) + ', Iter ' + str(testPoint))
        plt.legend(loc='lower right')
        plt.show()    
        f.savefig('./Permuated_MNIST_plots_1/acc_diff_pt_'+str(s)+'_'+str(testPoint)+'.pdf', bbox_inches='tight')
        plt.close()

def plotAccuracyDiffPointAll(allAcc,Actual_Accuracies,testPoint=25):
    f = plt.figure()
    yAll = []
    for i in range(10):
        yAll.append([])
        for s in range(i,10):
            yAll[i].append(Actual_Accuracies[i] - allAcc[s][testPoint][i])
    
    for i,y in enumerate(yAll):
        plt.plot(range(i,10),y,'.-', markersize=5,label='Skill: '+str(i))
    plt.xlabel('Stages')
    plt.ylabel('True Acc - Reconst. Acc')
    plt.title('Encoded Skills -- Iter ' + str(testPoint))
    plt.legend(loc='upper left')
    plt.show()    
    f.savefig('./Permuated_MNIST_plots_1/acc_diff_pt_all_'+str(s)+'_'+str(testPoint)+'.pdf', bbox_inches='tight')
    plt.close()

######################################################################################################
# LOADING OF VAE OUTPUT SKILL WEIGHTS BACK INTO THE MNSIT NETWORK AND TEST MNIST
######################################################################################################
print("VAE SCHEDULER IS ", vae_optimizer)
Skill_Mu=[]
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
    student_model.train()
    final_dataframe_1=pd.DataFrame()
    total_resend=0
    total=len(task_samples)
    accuracies = np.zeros((iterations,len(task_samples)))
    for i in range(0,len(task_samples)-len(Skill_Mu)):
        Skill_Mu.append([])
    for batch_idx in range(1,iterations):
        randPerm = np.random.permutation(len(task_samples))
        #randPerm,nReps = helper_functions.biased_permutation(stage+1,nBiased,trainBias,len(task_samples),minReps)
        #print(randPerm,nReps)
        resend=[]
        for s in randPerm:
            skill=s
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill])).to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc2.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, 
                data.view(-1, 21840), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            loss.backward()
            #options_lgnd = dict(fillarea=True, legend=['Loss_Skill_'+str(skill)])
            vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win=win,name='Loss_Skill_'+str(skill),update='append')#,opts=options_lgnd)
            vae_optimizer.step()
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
            if float(loss.data[0] * 10000)>=1.00:
                resend.append(s)
        print("RESEND List",resend)
        total_resend=total_resend+len(resend)
        for s in resend:
            skill=s
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill])).to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc2.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, 
                data.view(-1, 21840), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            loss.backward()
            #options_lgnd = dict(fillarea=True, legend=['Loss_Skill_'+str(skill)])
            vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win=win,name='Loss_Skill_'+str(skill),update='append')#,opts=options_lgnd)
            vae_optimizer.step()
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
            
           
        if batch_idx %1==0:
            values=0
            for i in range(0,len(task_samples)):
                collect_data_1=[]
                mu1=Skill_Mu[i][0]
                task_sample = student_model.decoder(mu1).cpu()
                sample=task_sample.data.numpy().reshape(21840)
                #print('MSE IS',i,mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy()))
                #mse=mean_squared_error(task_sample.data.numpy(),Variable(torch.FloatTensor(task_samples[i])).data.numpy())
                final_weights=helper_functions.unFlattenNetwork(sample, shapes)
                model_x=loadWeights_mnsit(final_weights,model)
                Train_loader,Test_loader=RELOAD_DATASET(idx_permute[i])
                mse=mean_squared_error(sample,Variable(torch.FloatTensor(task_samples[i])).data.numpy())
                mse_orginal=mean_squared_error(sample,Actual_task_net_weights[i])
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
                #     if round(Avg_Accuracy+0.5)>=int(Actual_Accuracy[i]):
                #         values=values+1
                # else:
                #     if round(Avg_Accuracy+0.5)>=int(Actual_Accuracy[i]):
                #         values=values+1
                # if len(task_samples)>6:
                #     if mse<=0.00015:
                #         print('Mse is {} Below Threshold for skill {}'.format(i,mse))
                #         values=values+1
                # else:
                #     if mse<=0.00015:
                #         print('Mse is {} Below Threshold for skill {}'.format(i,mse))
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
    final_dataframe_1.to_hdf('Collected_Data/'+str(len(task_samples))+'_testset_data_'+Epoch+'__epoch','key1')
    return accuracies


######################################################################################################
# FLATTEN THE MNIST WEIGHTS AND FEED IT TO VAE
######################################################################################################
def FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model):
    final_skill_sample=[]
    Flat_input,net_shapes=helper_functions.flattenNetwork(model.cpu())
    final_skill_sample.append(Flat_input)
    Actual_task_net_weights.append(Flat_input)
    vis.line(X=np.array(range(0,len(Flat_input))),Y=Flat_input,win='win_task_network',name='Task_net_'+str(len(task_samples)),opts=options_task_network,update='append')
    if len(task_samples)==0:
        accuracies = CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,300)
    else:
        accuracies = CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,300)
    return accuracies
######################################################################################################
#                                   CREATING THE PERMUTATIONS
######################################################################################################
#np.random.seed(0)
idx_permute = [np.random.permutation(28**2) for _ in range(10)] 
#=vis.bar(X=[1,2,3])
options_task_network = dict(fillarea=True,width=400,height=400,xlabel='Task Network',ylabel='T',title='TASK_NETWORK')
#win_task_network = vis.line(Y=np.array([0.001,0.001]),win='Task Net Dist',name='TASK NET DIST',opts=options_task_network)
######################################################################################################
#                                   TRAINING STARTS HERE - MNIST + CAE
######################################################################################################

allAcc = []
for permuatation in range(0,10):
    model = models.Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    task_samples=[]
    print("########## \n Threshold id is",threshold_batchid,"\n#########")
    Train_loader,Test_loader=RELOAD_DATASET(idx_permute[permuatation])
    SHOW_TEST_TRAIN_IMAGES_SAMPLE(permuatation)
    accuracy=train(model,Train_loader,Test_loader,optimizer,3,'MNIST Skill '+str(permuatation+1))
    options = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Actual_Accuracy',title='Actual_Accuracy')
    Actual_Accuracy.append(int(accuracy))
    print("Actual acc",Actual_Accuracy)
    torch.save(model.state_dict(),'task_net_models/mnist_reset_digit_solver_'+str(permuatation)+'.pt')
    if permuatation==0 :
        accuracies = FLATTEN_WEIGHTS_TRAIN_VAE([],model)
    else:
        for i in range(0, permuatation):
            recall_memory_sample=Skill_Mu[i][0]
            generated_sample=student_model.decoder(recall_memory_sample).cpu()
            task_samples.append(generated_sample)
        accuracies = FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model)
    allAcc.append(accuracies)

    with open('Collected_Data/allAcc_testset_'+Epoch+'_epoch_'+str(permuatation), 'wb') as fp:
        pickle.dump(allAcc, fp)

    with open('Collected_Data/Actual_Accuracy_testset_'+Epoch+'_epoch_'+str(permuatation), 'wb') as fp:
        pickle.dump(Actual_Accuracy, fp)

    with open('Collected_Data/tasknet_'+Epoch+'_epoch_'+str(permuatation), 'wb') as fp:
        pickle.dump(Actual_task_net_weights, fp)
    

    if permuatation>=1:
        options = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Actual_Accuracy',title='Actual_Accuracy')
        options_threshold = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Threshold_Cutoff',title='Cutoff_Threshold')
        options_threshold_net_updates = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Threshold_Cutoff',title='Cutoff_Threshold_Net_Updates')
        vis.bar(X=Actual_Accuracy,opts=options,win='acc_viz')
        vis.bar(X=threshold_batchid,opts=options_threshold,win='threshold_viz')
        vis.bar(X=threshold_net_updates,opts=options_threshold_net_updates,win='threshold_viz_net_updates')



#plotAccuracies(allAcc)
#plotAccuracyDiff(allAcc,Actual_Accuracy)
#plotAccuracyDiffPoint(allAcc,Actual_Accuracy,testPoint=25)
#plotAccuracyDiffPointAll(allAcc,Actual_Accuracy,testPoint=25)


