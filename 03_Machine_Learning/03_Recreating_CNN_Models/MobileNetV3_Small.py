"""
Title: MobileNetV3: Searching for the Next Generation of MobileNet Models

Description: Implementing MobileNetV3 Small in Pytorch

"""

import torch
import torch.nn as nn

# ConvBlock
class ConvBlock(nn.Module):
    """ Convolution Block with Conv2d, BatchNorm2d, ReLU6 """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
# Squeeze-and-Excitation Block
class SEBlock(nn.module):
    """ Squeeze and Excitation Block """
    def __init__(self):
        super(SEBlock, self).__init__()    
    
# Bottleneck Block
class BottleneckBlock(nn.Module):
    """ MobileNetV3 Bottleneck Block """
    
class MobileNetV3(nn.Module):
    """ MobileNetV3 """
    def __init__(self):
        super(MobileNetV3, self).__init__()
        
        # First Convolution Layer
        
        # Bottleneck Blocks
        
        # Last Convolution Layer
        
        # Global Average Pooling
        
        # Fully Connected Layer
        
        
        
    def forward(self, x):
        pass