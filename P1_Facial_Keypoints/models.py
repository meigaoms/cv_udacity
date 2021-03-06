## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import math


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #output size = (W-F+2P)/S + 1 = 224
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2) #output size = W/S = 112
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) #output size = (W-F+2P)/S + 1 = 112
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) #output size = W/S = 56
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) #output size = (W-F+2P)/S + 1 = 56
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) #output size = W/S = 28
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #output size = (W-F+2P)/S + 1 = 28
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2) #output size = W/S = 14

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # self.init_weights()

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(0.0, 1.0)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                I.xavier_uniform(m.weight)