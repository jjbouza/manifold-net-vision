import torch as to
import numpy as np
import torch.nn as nn
from msqrt import sqrtm
import torch.nn.functional as F

#For grass geodesic set iterative_stat_function=weightedFrechetMeanUpdate
def weightedIterativeStatistic(point_list, weights, iterative_stat_function):
    mean = iterative_stat_function(point_list[0], point_list[1], weights[0])
    for point,weight in zip(point_list[2:], weights[2:]):
        mean = iterative_stat_function(mean, point, weight)
    return mean

#Geodesic approximation
def stiefelGeodesicApprox(X,Y,t):
    #t = to.abs(t)
    lift = Y - 0.5*X@(Y.t()@X+X.t()@Y)
    scale = t*lift
    retract = (X+scale)@matrixRootInverse(to.eye(scale.shape[1])+scale.t()@scale)
    return retract

def grassmanGeodesic(X,Y,t):
    svd_term = Y@to.inverse(X.t()@Y)-X
    U,s,V = to.svd(svd_term)
    theta = to.atan(s).float()
    qr_term = X@V@to.diag(to.cos(theta*t))+U@to.diag(to.sin(theta*t))
    return qr_term

def matrixRootInverse(X):
    return to.inverse(sqrtm(X))

def weightedFrechetMeanUpdate(previous_mean, new_point, weight, geodesic_generator=stiefelGeodesicApprox):
    return geodesic_generator(previous_mean, new_point, weight.float())

#Does a single conv. in 1D. Use for weighted FM.
class fullConv1d(nn.Module):
    #Init with iterative_mean_function = weightedFrechetMeanUpdate
    #num_frame = number of blocks.
    def __init__(self, num_frames, iterative_mean_function = weightedFrechetMeanUpdate):
            super(fullConv1d, self).__init__()
            self.iterative_mean_function = iterative_mean_function
            #Weights: By default we init weights so that we compute the unweighted
            #FM:
            self.weight = nn.Parameter(to.Tensor([1/n for n in range(2,num_frames+2)]), requires_grad=True)
            self.weight_reference = to.sum(to.Tensor([1/n for n in range(2, num_frames+2)]))

    def forward(self, block_list):
        #Computes weighted FM:
        #self.weight = self.weight/to.sum(self.weight)
        out = weightedIterativeStatistic(block_list, self.weight, weightedFrechetMeanUpdate)
        weight_penalty = (self.weight_reference - to.sum(self.weight))**2

        return out, weight_penalty


