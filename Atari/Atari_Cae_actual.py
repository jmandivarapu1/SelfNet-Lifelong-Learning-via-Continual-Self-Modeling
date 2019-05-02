##############################################################################################################################
# Aim of the Experiment: Continual Learning on CIFAR-100 with Breakup
# Output: All the collected data for individual skill is present in the folder -- Collected_Data
# Task Nets: All the trained CIFAR 100 weights saved in the folder -- task_net_models
# Test & Train Image Samples: All the train,test CIFAR 10 task images sample saved in the folder -- Test_Train_Images
# Plots : If any plots you can save in folder -- Permuated_MNIST_plots (Currently Not Used)
# Conclusion : 
##############################################################################################################################

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import helper_functions
import models
import pandas as pd
import os
import numpy as np
import pickle
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import visdom
vis = visdom.Visdom()
vis.delete_env('Atari_rerun_lr_0.0001_lam_0.0001') #If you want to clear all the old plots for this python Experiments.Resets the Environment
vis = visdom.Visdom(env='Atari_rerun_lr_0.0001_lam_0.0001')
#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
################################################################################
# EXECUTION FLAGS                                 #                             
################################################################################
Flags = {}
Flags['CIFAR_100'] = False
Flags['CIFAR_10_Incrmental'] = True
Flags['NC_NET'] = False
Flags['NC_NET_CUM'] = False
Flags['CAE_Train'] = True
Flags['SPLIT_INPUT']=True
best_acc = 0  # best test accuracy

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda =  torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


######################################################################################################
#                                   RELOD THE WEIGHTS INTO NET                                       #
######################################################################################################
def loadWeights_cifar(weights_to_load, net):
    net.conv1.weight.data = torch.from_numpy(weights_to_load[0])
    net.conv1.bias.data =   torch.from_numpy(weights_to_load[1])
    net.conv2.weight.data = torch.from_numpy(weights_to_load[2])
    net.conv2.bias.data =   torch.from_numpy(weights_to_load[3])
    net.conv3.weight.data = torch.from_numpy(weights_to_load[4])
    net.conv3.bias.data =   torch.from_numpy(weights_to_load[5])
    net.conv4.weight.data = torch.from_numpy(weights_to_load[6])
    net.conv4.bias.data =   torch.from_numpy(weights_to_load[7])
    net.gru.weight_ih.data = torch.from_numpy(weights_to_load[8])
    net.gru.weight_hh.data = torch.from_numpy(weights_to_load[9])
    net.gru.bias_ih.data =   torch.from_numpy(weights_to_load[10])
    net.gru.bias_hh.data =   torch.from_numpy(weights_to_load[11])
    net.critic_linear.weight.data = torch.from_numpy(weights_to_load[12])
    net.critic_linear.bias.data = torch.from_numpy(weights_to_load[13])
    net.actor_linear.weight.data = torch.from_numpy(weights_to_load[14])
    net.actor_linear.bias.data =torch.from_numpy(weights_to_load[15])
    
    return net
#####################################################################################################################
#                                               Adjusting Learning Rate                                             #
#####################################################################################################################
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


######################################################################################################
#                                   GLOBAL VARIABLES
######################################################################################################
student_model=models.CAE_Big().to(device)
#teacher_model=models.CAE_Big().to(device)
vae_optimizer =optim.Adam(student_model.parameters(), lr = 0.00001)#, amsgrad=True)
lam = 0.0001
input_size=76863
split_size=11
diff_count=[]
Actual_Accuracy=[]
threshold_batchid=[]
threshold_net_updates=[]
Actual_task_net_weights=[]
#biased training variables
nSamples = 10
nBiased = min(nSamples,10)
trainBias = 0.5
minReps = 1
nReps = 20
stage=2
addRepsTotal = nReps*minReps
games=['boxing.40.tar','stargunner.40.tar','kangaroo.40.tar','pong-v4.40.tar','spaceinvaders-v4.40.tar']#, 'breakout-v4.40.tar']
games_config=[18,18,18,6,6]#,4]
Skill_Mu=[[] for _ in range(0,split_size*len(games))]
#####################################################################################################################
#                                              CAE TRAIN ON ATARI                                                   #
#####################################################################################################################
def CAE_AE_TRAIN(shapes,task_samples,iterations):
    global stage
    global Skill_Mu
    global student_model
    global vae_optimizer
    options = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='Loss',title='CAE_skills'+str(len(task_samples)))
    options_2 = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='Accuracy',title='Acc_skills_'+str(len(task_samples)))
    options_mse = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='MSE',title='MSE_skills_'+str(len(task_samples)))
    options_mse_org = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='Cosine_Similarity',title='Cosine_Similarity_'+str(len(task_samples)))
    options_cosine_org_orginal = dict(fillarea=True,width=400,height=400,xlabel='Iterations',ylabel='Cosine_Similarity',title='Cosine_Sim_wrt_org_'+str(len(task_samples)))
    win = vis.line(X=np.array([0]),Y=np.array([0.005]),win='Loss_skills_'+str(len(task_samples)),name='Loss_skills'+str(len(task_samples)),opts=options)
    # win_2 = vis.line(X=np.array([0]),Y=np.array([0]),win='Acc_skills_'+str(len(task_samples)),name='Acc_skills_'+str(len(task_samples)),opts=options_2)
    win_mse = vis.line(X=np.array([0]),Y=np.array([0]),win='MSE_skills_'+str(len(task_samples)),name='MSE_skills_'+str(len(task_samples)),opts=options_mse)
    win_mse_org = vis.line(X=np.array([0]),Y=np.array([0]),win='Cosine_Sim '+str(len(task_samples)),name='Cosine_similarity_'+str(len(task_samples)),opts=options_mse_org)
    win_cosine_org_orginal = vis.line(X=np.array([0]),Y=np.array([0]),win='Cosine_Sim_orginal '+str(len(task_samples)),name='Cosine_sim_wrt_org'+str(len(task_samples)),opts=options_cosine_org_orginal)

    splitted_input=[]
    skills=[]
    total_resend=0
    task_samples_copy=task_samples
    total=len(task_samples)
    final_dataframe_1=pd.DataFrame()
    accuracies = np.zeros((iterations,len(task_samples)))
    for t_input in range(0,len(task_samples)):
        task_sample_append=np.concatenate((task_samples[t_input],task_samples[t_input][-diff_count[t_input]:]))
        splitted_input=np.array_split(task_sample_append,split_size)
        for i in range(0,len(splitted_input)):
            # if(len(task_samples[t_input])==842407):
            #     splitted_input[i]=np.concatenate((splitted_input[i], [0.05]))
            # else:
            #     splitted_input[i]=np.concatenate((splitted_input[i], [0.05]))
            print(len(splitted_input[i]))
            skills.append(splitted_input[i])
        
    task_samples=skills
    student_model.train()
    iterations=iterations
    for batch_idx in range(1,iterations):
        randPerm=np.random.permutation(len(task_samples))
        #randPerm,nReps = helper_functions.biased_permutation(stage+1,nBiased,trainBias,total*3,minReps)
        resend=[]
        for s in randPerm:
            skill=s
            vae_optimizer.zero_grad()
            data=Variable(torch.FloatTensor(task_samples[skill])).to(device)
            hidden_representation, recons_x = student_model(data)
            W = student_model.state_dict()['fc2.weight']
            loss,con_mse,con_loss,closs_mul_lam = helper_functions.Contractive_loss_function(W, 
                data.view(-1, input_size), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win=win,name='Loss_Skill_'+str(skill),update='append')
            loss.backward()
            vae_optimizer.step()
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
            cosine=F.cosine_similarity(recons_x, data.view(-1, input_size))
            if cosine<=0.991:
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
                data.view(-1, input_size), recons_x,hidden_representation, lam)
            Skill_Mu[skill]=hidden_representation
            vis.line(X=np.array([batch_idx]),Y=np.array([loss.item()]),win=win,name='Loss_Skill_'+str(skill),update='append')
            loss.backward()
            vae_optimizer.step()
            print('Train Iteration: {},tLoss: {:.6f},picked skill {}'.format(batch_idx,loss.data[0],skill ))
        if batch_idx %1==0:
            m=0
            n=split_size
            values=0
            for i in range(0,int(len(task_samples_copy))):
                collect_data_1=[]
                sample=[]
                for k in range(m,n):
                    mu1=Skill_Mu[k][0]
                    mini_task_sample = student_model.decoder(mu1).cpu()
                    task_sample=mini_task_sample.data.numpy().reshape(input_size)
                    sample=np.concatenate([sample,task_sample])
                #print("in cae test len is",len(sample))
                    # if len(Actual_task_net_weights[i])==841893:
                    #     sample=np.concatenate([sample,task_sample[0:-3]])
                    # else:
                sample=sample[0:-diff_count[i]]
                m=m+split_size
                n=n+split_size
                model_z=models.NNPolicy(1,256,games_config[i])
                FI,net_shapes=helper_functions.flattenNetwork(model_z.cpu())
                final_weights=helper_functions.unFlattenNetwork(torch.from_numpy(sample).float(), net_shapes)
                #print("lksfkfjf",diff_count,len(sample),len(task_samples_copy[i]))
                model_x=loadWeights_cifar(final_weights,model_z)
                mse=mean_squared_error(sample,Variable(torch.FloatTensor(task_samples_copy[i])).data.numpy())
                mse_orginal=mean_squared_error(sample,Actual_task_net_weights[i])
                # sample=sample[:-diff_count[i]]
                b=torch.FloatTensor(task_samples_copy[i]).data.numpy()#[:-diff_count[i]]
                b1=torch.FloatTensor(Actual_task_net_weights[i]).data.numpy()
                COSINE_SIMILARITY=dot(sample, b)/(norm(sample)*norm(b))
                COSINE_SIMILARITY_wrt_orginal=dot(sample, b1)/(norm(sample)*norm(b1))
                if COSINE_SIMILARITY>=0.998:
                    torch.save(model_x.state_dict(),'./New_Games/'+str(games[i][0:10])+'_'+str(total)+'_'+str(COSINE_SIMILARITY)+'_'+str(batch_idx)+'.pt')
                    values=values+1
                # else:
                #     if COSINE_SIMILARITY>=0.998:
                #         torch.save(model_x.state_dict(),'./Latest_Cae/'+str(games[i][0:10])+'_'+str(total)+'_'+str(COSINE_SIMILARITY)+'_'+str(batch_idx)+'.pt')
                #         values=values+1
                #torch.save(model_x.state_dict(),'./Latest_Cae/'+str(games[i][0:10])+'_'+str(total)+'_'+str(COSINE_SIMILARITY)+'.pt')
                vis.line(X=np.array([batch_idx]),Y=np.array([mse]),win=win_mse,name='MSE_Skill_'+str(i),update='append')
                vis.line(X=np.array([batch_idx]),Y=np.array([COSINE_SIMILARITY]),win=win_mse_org,name='Cos_Sim_Skill_'+str(i),update='append')#,opts=options_lgnd)
                vis.line(X=np.array([batch_idx]),Y=np.array([COSINE_SIMILARITY_wrt_orginal]),win=win_cosine_org_orginal,name='Cos_wrt_org_Sim_Skill_'+str(i),update='append')#,opts=options_lgnd)
        if values==len(task_samples_copy):
            print("########## \n Batch id is",batch_idx,"\n#########")
            threshold_batchid.append(batch_idx+total_resend)
            break


    return accuracies

######################################################################################################
#                           FLATTEN THE CIFAR WEIGHTS AND FEED IT TO CAE                             #
######################################################################################################
def FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model):
    final_skill_sample=[]
    Flat_input,net_shapes=helper_functions.flattenNetwork(model.cpu())
    print("llll",len(Flat_input))
    final_skill_sample.append(Flat_input)
    Actual_task_net_weights.append(Flat_input)
    if len(task_samples)==0:
        accuracies = CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,600)
    else:
        accuracies = CAE_AE_TRAIN(net_shapes,task_samples+final_skill_sample,600)
    return accuracies

#####################################################################################################################
#                                  CIFAR AND CAE TRAIN START HERE ALL START HERE                                    #
#####################################################################################################################
#if Flags['CIFAR_10_Incrmental']:

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
for cifar_class in range(0,len(games)):
    allAcc = []
    task_samples=[]
    model=models.NNPolicy(1,256,games_config[cifar_class])
    model.load_state_dict(torch.load('./Trained_nets/'+ str(games[cifar_class])))
    count = count_parameters(model)
    print (count)
    diff_count.append(845493-count)
    if cifar_class==0:
        accuracies=FLATTEN_WEIGHTS_TRAIN_VAE([],model)
    else:
        m=0
        n=split_size
        #print(Skill_Mu)
        #print(len(Skill_Mu))
        for i in range(0,cifar_class):
            generated_sample=[]
            for j in range(m,n):  
                recall_memory_sample=Skill_Mu[j][0]
                sample=student_model.decoder(recall_memory_sample).cpu()
                sample=sample.data.numpy()#.reshape(20442)
                generated_sample=np.concatenate([generated_sample,sample])
            print("Len of generated sample is",len(generated_sample))
            task_samples.append(generated_sample[0:-diff_count[i]])
            m=m+split_size
            n=n+split_size
        accuracies=FLATTEN_WEIGHTS_TRAIN_VAE(task_samples,model) 
    allAcc.append(accuracies)
    if cifar_class>=1:
        options_threshold = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Threshold_Cutoff',title='Cutoff_Threshold')
        options_threshold_net_updates = dict(fillarea=True,width=400,height=400,xlabel='Skill',ylabel='Threshold_Cutoff',title='Cutoff_Threshold_Net_Updates')
        vis.bar(X=threshold_batchid,opts=options_threshold,win='threshold_viz')
        vis.bar(X=threshold_net_updates,opts=options_threshold_net_updates,win='threshold_viz_net_updates')

vis.save(vis)



#python a3c.py --env SpaceInvaders-v4 --test True --input Orginal --ipname spaceinvaders-v4.40.tar
#python a3c.py --env Pong-v4 --test True --input Orginal --ipname pong-v4.40.tar
#python a3c.py --env StarGunner-v4 --test True --input Orginal --ipname stargunner.40.tar
#python a3c.py --env Breakout-v4 --test True --input Orginal --ipname breakout-v4.40.tar
#python a3c.py --env Boxing-v4 --test True --input Orginal --ipname boxing.40.tar
#python a3c.py --env Kangaroo-v4 --test True --input Orginal --ipname kangaroo.40.tar

#python a3c.py --env SpaceInvaders-v4 --test True --input recollect --ipname spaceinvad_100_0.9864520836481616.pt
#python a3c.py --env Pong-v4 --test True --input Recollection --ipname pong-v4.40_100_0.9903461392026515.pt
#python a3c.py --env Breakout-v4 --test True --input recollect --ipname breakout-v_100_0.9962183854382421.pt
#python a3c.py --env Boxing-v4 --test True --input recollection --ipname boxing.40._100_0.9962899174301124.pt
#python a3c.py --env Kangaroo-v4 --test True --input recollection --ipname kangaroo.4_100_0.9968678176531669.pt
#python a3c.py --env StarGunner-v4 --test True --input recollection --ipname


