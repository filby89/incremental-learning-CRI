import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable, Function



def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target.long())

def bce_loss_niki(output, target):
    loss = F.binary_cross_entropy_with_logits(output, target)
    return loss

def get_corr():
    import pickle
    result = pickle.load(open('adj_corr.pickle', 'rb'))
    return torch.Tensor(result).float()

def get_ccc():
    res = np.load('ccc.npy')
    return torch.Tensor(res).float()


def combined_loss(output, target):
    l = F.mse_loss(output, target)

    l += bce_loss(output, target)
    # from model.bpmll_pytorch import BPMLLLoss
    # l += BPMLLLoss()(output, target)

    return l

def cross_entropy(output, target):
    t = target.clone().detach()
    output = torch.sigmoid(output)
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0
    return F.cross_entropy(output, t.long())/torch.sum(t,dim=(0,1))


