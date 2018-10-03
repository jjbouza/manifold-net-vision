import torch as to
from torch import nn

import fm_ops as ops


class GrassmannAverageProjection(nn.Module):
    def __init__(self, in_frames, out_frames):
        super(GrassmannAverageProjection, self).__init__()
        self.temporal_mean = GrassmannAverage(in_frames, out_frames)

    def forward(self, x):
        y, weight_penalty = self.temporal_mean(x)
        x = temporalProjection(x,y)
        return x,weight_penalty

class GrassmannAverageBottleneck(nn.Module):
    def __init__(self, in_frames, out_frames):
        super(GrassmannAverageBottleneck, self).__init__()
        self.temporal_mean = GrassmannAverage(in_frames, out_frames)

    def forward(self, x):
        y, weight_penalty = self.temporal_mean(x)
        x = temporalReconstruction(x,y)
        return x,weight_penalty

#Given a video with dimensions [frames, channels, height, width]
#this module reduces the first dimenion by taking a weighted FM.
class GrassmannAverage(nn.Module):
    def __init__(self, in_frames, out_frames):
        super(GrassmannAverage, self).__init__()
        self.out_frames = out_frames
        self.num_blocks = int(in_frames/out_frames)
        self.weights = nn.Parameter(to.Tensor([1/n for n in range(2, self.num_blocks+2)]), requires_grad=True)
        self.weight_reference = to.sum(to.Tensor([1/n for n in range(2, self.num_blocks+2)]))

    def forward(self, x):
        blocks = orthogonalizeBlocks(x, self.num_blocks)
        self.fm = ops.weightedIterativeStatistic(blocks, self.weights, ops.weightedFrechetMeanUpdate)
        weight_penalty = (self.weight_reference - to.sum(to.abs(self.weights)))**2

        return self.fm, weight_penalty

    def run(self, x):
        return temporalProjection(x, self.fm)


#Given a video and a temporal subspace returned by TemporalMean, calculates the reconstruction.
def temporalProjection(video_original, subspace):
    principal_proj_coords = video_original@subspace
    return principal_proj_coords

def temporalReconstruction(video_original, subspace):
    principal_proj_coords = video_original@subspace
    reconstruction = principal_proj_coords@subspace.t()
    return reconstruction

def orthogonalizeBlocks(video, num_blocks):
    block_size = int(video.shape[0]/num_blocks)

    orth_blocks = []
    previous_index = 0
    for i in range(0, num_blocks):
        block = video[previous_index:previous_index+block_size]
        orth_blocks.append(chol_orthogonalize(block))
        previous_index += block_size

    return to.stack(orth_blocks)

def chol_orthogonalize(vector_matrix):
    VV = vector_matrix@vector_matrix.t()+0.01*to.eye(vector_matrix.shape[0])
    R = to.potrf(VV, upper=True)
    U = vector_matrix.t()@to.inverse(R)

    return U

def gram_schmidt(vector_matrix, eps=1e-10):
    vector_matrix = vector_matrix
    basis = [vector_matrix[0]]

    for v in (vector_matrix)[1:]:
        coeff_vec = to.sum(to.stack([b*to.dot(v,b)/to.dot(b,b) for b in basis]),0)
        w = v - coeff_vec

        if w.norm() < eps:
            print('Gram Schmidt Error: A matrix passed to gram schmidt was not full rank.')
            quit()

        basis.append(w/w.norm())


    return to.stack([b for b in basis]).t()

        



