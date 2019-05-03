import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from scipy.stats import geom

#############################################################################
# Contractive Loss
#############################################################################
def Contractive_loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss
    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.
    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term
    Returns:
        Variable: the (scalar) CAE loss
    """

    mse = F.mse_loss(recons_x, x)
    # cos = F.cosine_similarity(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)

    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    #print(w_sum.shape,(dh**2).shape)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)



    


    return mse + contractive_loss.mul_(lam) ,mse,contractive_loss,contractive_loss.mul_(lam)







def Layer_Weighted_MSE(target, recons):
    # print(target)
    # print("-------------------")
    # print(recons)
    # print("--------------------")
    both = torch.cat((target, recons), dim=1)
    both_mean = both.mean(dim=1)
    both_std = both.std(dim=1)
    target = (target - both_mean)/both_std
    recons = (recons - both_mean)/both_std

    # target_mean = target.mean(dim=1)
    # target_std = target.std(dim=1)
    # target = (target - target_mean)/target_std

    # recon_mean = recons.mean(dim=1)
    # recon_std = recons.std(dim=1)
    # recons = (recons - recon_mean)/recon_std

    conv1_target = target[0][0:260].reshape(-1,260)
    conv1_recons = recons[0][0:260].reshape(-1,260)
    concatenated_conv1 = torch.cat((conv1_target, conv1_recons), dim=1)
    conv1_std = concatenated_conv1.std(dim=1)

    # print(conv1_target)
    # print("-------------------")
    # print(conv1_recons)


    conv2_target = target[0][260:5280].reshape(-1,5020)
    conv2_recons = recons[0][260:5280].reshape(-1,5020)
    concatenated_conv2 = torch.cat((conv2_target, conv2_recons), dim=1)
    conv2_std = concatenated_conv2.std(dim=1)

    fc1_target = target[0][5280:21330].reshape(-1,16050)
    fc1_recons = recons[0][5280: 21330].reshape(-1,16050)
    concatenated_fc1 = torch.cat((fc1_target, fc1_recons), dim=1)
    fc1_std = concatenated_fc1.std(dim=1)

    fc2_target = target[0][21330:].reshape(-1,102)
    fc2_recons = recons[0][21330:].reshape(-1,102)
    concatenated_fc2 = torch.cat((fc2_target, fc2_recons), dim=1)
    fc2_std = concatenated_fc2.std(dim=1)

    # print(Variable(torch.FloatTensor(slice_orig.reshape(-1,260)), requires_grad=True))
    # sys.exit()

    conv1_mse = F.mse_loss(conv1_recons, conv1_target)
    # conv1_mse = conv1_mse/260
    conv1_cos = F.cosine_similarity(conv1_recons, conv1_target)
    conv2_mse = F.mse_loss(conv2_recons, conv2_target)
    # conv2_mse = conv2_mse/5020
    conv2_cos = F.cosine_similarity(conv2_recons, conv2_target)
    fc1_mse = F.mse_loss(fc1_recons, fc1_target)
    # fc1_mse = fc1_mse/16050
    fc1_cos = F.cosine_similarity(fc1_recons, fc1_target)
    fc2_mse = F.mse_loss(fc2_recons, fc2_target)
    # fc2_mse = fc2_mse/102
    fc2_cos = F.cosine_similarity(fc2_recons, fc2_target)
    #normalized coeeficients = .9844, .132, .0399, .1090
    #probability distribution (sums to 1) = 0.77595628, 0.10382514, 0.03278689, 0.08743169
    layer_stds = [conv1_std, conv2_std, fc1_std, fc2_std]
    coefficients = [ 0.67595628, 0.25382514, 0.03278689, 0.03743169]
    CL_coeffs = [10, 5, .2, .2]
    # coefficients = CL_coeffs
    combined_mse = conv1_mse.mul_(coefficients[0]) + conv2_mse.mul_(coefficients[1]) + fc1_mse.mul_(coefficients[2]) + fc2_mse.mul_(coefficients[3]) + conv1_cos*coefficients[0]*.1 + conv2_cos*coefficients[1]*.1  + fc1_cos*coefficients[2]*.1  + fc2_cos*coefficients[3]*.1



    return combined_mse, conv1_mse, conv2_mse, fc1_mse, fc2_mse, conv1_cos, conv2_cos, fc1_cos, fc2_cos
#############################################################################
# Flattening the NET
#############################################################################
def flattenNetwork(net):
    flatNet = []
    shapes = []
    for param in net.parameters():
        #if its WEIGHTS
        curr_shape = param.cpu().data.numpy().shape
        shapes.append(curr_shape)
        if len(curr_shape) == 2:
            param = param.cpu().data.numpy().reshape(curr_shape[0]*curr_shape[1])
            flatNet.append(param)
        elif len(curr_shape) == 4:
            param = param.cpu().data.numpy().reshape(curr_shape[0]*curr_shape[1]*curr_shape[2]*curr_shape[3])
            flatNet.append(param)
        else:
            param = param.cpu().data.numpy().reshape(curr_shape[0])
            flatNet.append(param)
    finalNet = []
    for obj in flatNet:
        for x in obj:
            finalNet.append(x)
    finalNet = np.array(finalNet)
    return finalNet,shapes


#############################################################################
# UN-Flattening the NET
#############################################################################
def unFlattenNetwork(weights, shapes):
    #this is how we know how to slice weights

    begin_slice = 0
    end_slice = 0
    finalParams = []
    #print(len(weights))
    for idx,shape in enumerate(shapes):
        if len(shape) == 2:
            end_slice = end_slice+(shape[0]*shape[1])
            curr_slice = weights[begin_slice:end_slice]
            param = np.array(curr_slice).reshape(shape[0], shape[1])
            finalParams.append(param)
            begin_slice = end_slice
        elif len(shape) == 4:
            end_slice = end_slice+(shape[0]*shape[1]*shape[2]*shape[3])
            curr_slice = weights[begin_slice:end_slice]
            #print("shape: "+str(shape))
            #print("curr_slice: "+str(curr_slice.shape))
            param = np.array(curr_slice).reshape(shape[0], shape[1], shape[2], shape[3])
            finalParams.append(param)
            begin_slice = end_slice
        else:
            end_slice = end_slice+shape[0]
            curr_slice = weights[begin_slice:end_slice]
            param = np.array(curr_slice).reshape(shape[0],)
            finalParams.append(param)
            begin_slice = end_slice
    finalArr = np.array(finalParams)
    return np.array(finalArr)


#############################################################################
# Load Net
#############################################################################
def loadWeights_mnsit(weights_to_load, model):
    j=0
    for i in model.features:
        #print(i.weight)
        i.weight= nn.Parameter(torch.from_numpy(unflaten_weights[j]))
        j=j+1
    return model

#############################################################################
# Biased Training
#############################################################################
def biased_permutation(nItems=20,nBiased=10,bias=0.35,addRepsTotal=10,minReps=1):
    nBiased = min(nItems,nBiased)
    excess = np.round(geom.pmf(np.arange(1,nBiased+1),bias)*addRepsTotal)

    perm = []
    nReps = np.ones(nItems,dtype=np.int32)*minReps
    for i in range(nItems):
        if i < nBiased:
            nReps[i] += excess[i].astype(np.int32)
        perm.extend(list(np.ones(nReps[i],dtype=np.int32)*i))
    revPerm = [(nItems - 1) - x for x in perm]

    return np.random.permutation(revPerm),nReps


#############################################################################
#               COUNTING NET PARAMETERS LAYER BY LAYER                      #
#############################################################################
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

#############################################################################
#               COUNTING NET PARAMETERS LAYER BY LAYER                      #
#############################################################################
def estimate_fisher(self, dataset, sample_size, batch_size=32):
    # sample loglikelihoods from the dataset.
    data_loader = utils.get_data_loader(dataset, batch_size)
    loglikelihoods = []
    for x, y in data_loader:
        x = x.view(batch_size, -1)
        x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
        y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
        loglikelihoods.append(
            F.log_softmax(self(x))[range(batch_size), y.data]
        )
        if len(loglikelihoods) >= sample_size // batch_size:
            break
    # estimate the fisher information of the parameters.
    loglikelihood = torch.cat(loglikelihoods).mean(0)
    loglikelihood_grads = autograd.grad(loglikelihood, self.parameters())
    parameter_names = [
        n.replace('.', '__') for n, p in self.named_parameters()
    ]
    return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}
