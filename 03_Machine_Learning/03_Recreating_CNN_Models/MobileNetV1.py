"""
Title: MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Applications

Description: Implementing MobileNetV1 in Pytorch. This implementation does not include alpha and rho values

"""
import torch
import torch.nn as nn
from torchsummary import summary

class BottleneckBlock(nn.Module):
    """ MobileNetV1 Bottleneck Block """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        ):
        super(BottleneckBlock, self).__init__()
        
        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
            )
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pw(x)
        return x
        
class MobileNetV1(nn.Module):
    """ MobileNetV1 """
    def __init__(self, config: list, 
            in_channels: int = 3, 
            classes: int = 1000
            ):
        super(MobileNetV1, self).__init__()
        
        # First Convolution Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

        # Bottleneck Blocks
        self.blocks = nn.ModuleList([])
        for i in range(len(config)):
            kernel_size, stride, input_channels, out_channels = config[i]
            
            if i == 6:
                for i in range(5):
                    self.blocks.append(BottleneckBlock(input_channels, out_channels, stride))
            else:
                self.blocks.append(BottleneckBlock(input_channels, out_channels, stride))
            
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(out_channels, classes)
        
        # Softmax
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # for block in self.blocks:
        for i, block in enumerate(self.blocks):
            # print(i)
            x = block(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        # print(x.shape)
        x = self.fc(x)
        return x
                         
layers_config= [
    # kerenl_size, stride, in_channels, out_channels
    [3, 1, 32, 64],     
    [3, 2, 64, 128],
    [3, 1, 128, 128],
    [3, 2, 128, 256],
    [3, 1, 256, 256],
    [3, 2, 256, 512],
    [3, 1, 512, 512],   # 5 times
    [3, 2, 512, 1024],
    [3, 2, 1024, 1024],
]


# 
# print("Build Model")
# model = MobileNetV1(layers_config, 3, 1000)
# summary(model, (3, 224, 224), device='cpu')
# print("Done Building")
