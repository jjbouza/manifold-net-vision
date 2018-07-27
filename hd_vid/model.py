import torch
import time
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import numpy as np
import GrassmannAverage as gm
import cv2

#Loads video into numpy array
def getVideoArray(name):
    #Load video into numpy array.
    cap = cv2.VideoCapture(name)
    #frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCount = 800
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))
    fc = 0
    ret = True
    while (fc < 800 and ret):
        ret, val = cap.read()
        if ret:
            buf[fc] = val.astype('float32')/255.0
        fc += 1

    cap.release()
    return buf


class autoencoder(nn.Module):
    def __init__(self, video_length, channel_factor, latent_variable_size):
        super(autoencoder, self).__init__()

        self.channel_factor = channel_factor
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(3, channel_factor, 3, 3)
        self.bn1 = nn.BatchNorm2d(channel_factor)

        self.e2 = nn.Conv2d(channel_factor, channel_factor*2, 3, 3)
        self.bn2 = nn.BatchNorm2d(channel_factor*2)

        self.e3 = nn.Conv2d(channel_factor*2, channel_factor*4, 3, 3)
        self.bn3 = nn.BatchNorm2d(channel_factor*4)

        self.e4 = nn.Conv2d(channel_factor*4, channel_factor*8, 3, 3)
        self.bn4 = nn.BatchNorm2d(channel_factor*8)
        
        #Grassmann average - PCA approximation in latent space.
        self.pca = gm.GrassmannAverageProjection(video_length, 20)

        #self.fc1 = nn.Linear(channel_factor*8*8*15, latent_variable_size)
        #self.fc1 = nn.Linear(20, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, channel_factor*8*8*15)
        self.d2 = nn.ConvTranspose2d(channel_factor*8, channel_factor*4, 3, 3)
        self.bn6 = nn.BatchNorm2d(channel_factor*4, 1.e-3)

        self.d3 = nn.ConvTranspose2d(channel_factor*4, channel_factor*2, 3, 3)
        self.bn7 = nn.BatchNorm2d(channel_factor*2, 1.e-3)

        self.d4 = nn.ConvTranspose2d(channel_factor*2, channel_factor, 3, 3)
        self.bn8 = nn.BatchNorm2d(channel_factor, 1.e-3)

        self.d5 = nn.ConvTranspose2d(channel_factor, 3, 3, 3)
        self.bn9 = nn.BatchNorm2d(1, 1.e-3)


        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        shape_outputs = []
        shape_outputs.append(x.shape)
        h1 = self.elu(self.bn1(self.e1(x)))
        shape_outputs.append(h1.shape)
        h2 = self.elu(self.bn2(self.e2(h1)))
        shape_outputs.append(h2.shape)
        h3 = self.elu(self.bn3(self.e3(h2)))
        shape_outputs.append(h3.shape)
        h4 = self.elu(self.bn4(self.e4(h3)))
        h5 = h4.view(-1, self.channel_factor*8*8*15)

        y, self.weight_penalty = self.pca(h5)

        return self.fc1(y), shape_outputs

    def decode(self, z, shape_outputs):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.channel_factor*8, 8, 15)
        h2 = self.elu(self.bn6(self.d2(h1, output_size=shape_outputs[-1])))
        h3 = self.elu(self.bn7(self.d3(h2, output_size=shape_outputs[-2])))
        h4 = self.elu(self.bn8(self.d4(h3, output_size=shape_outputs[-3])))

        return self.sigmoid(self.d5(h4, output_size=shape_outputs[-4]))

    def forward(self, x):
        latent, shape_outputs = self.encode(x)
        res = self.decode(latent, shape_outputs)
        return res, self.weight_penalty

model = autoencoder(video_length=300,channel_factor=16, latent_variable_size=50)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Model Parameters: ", params)

#MSE loss
def loss_function(recon_x, x):
    mse_loss = torch.mean((recon_x-x)**2)
    return mse_loss


optimizer = optim.Adam(model.parameters(), lr=1e-2)

def train():
    video = torch.from_numpy(getVideoArray('hd2.mp4')[450:750]).permute(0,3,1,2)

    loss_track = []
    recon_batch = None

    try:
        for i in range(1000):
            start = time.time() 
            optimizer.zero_grad()
            recon_batch, weight_penalty = model(video)
            loss = loss_function(recon_batch, video)
            loss = loss+weight_penalty
            print('Iteration ', i, ': ', loss-weight_penalty)
            loss.backward()
            optimizer.step()
            end = time.time()
            print('Iteration ', i, ' time: ', end-start)

    except KeyboardInterrupt:
        pass
            
    return recon_batch, loss_track


vid,loss = train()
np.save('out_m2', vid.detach().numpy())
np.save('loss_m2', np.array(loss))
torch.save(model, 'model2')
