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
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam),mse,contractive_loss,contractive_loss.mul_(lam)
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