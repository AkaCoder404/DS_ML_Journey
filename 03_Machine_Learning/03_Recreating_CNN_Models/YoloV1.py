"""
Title: You Only Look Once: Unified, Real-Time Object Detection
Description: Implementation of YoloV1

Notes:
- Batch Normalization (wasn't used in the original paper)
- Darknet-19 - Joseph Redmon, but wasn't used in the original paper
- Fully Connected is 496 neurons (Paper uses 4096 neurons)
"""

import torch
import torch.nn as nn
import torchsummary as summary


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):  
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class YOLOV1(nn.Module):
    def __init__(self, config, in_channels=3, classes=20, grid_size=7, num_boxes=2):
        super(YOLOV1, self).__init__()
        self.in_channels = in_channels
        self.config = config
        self.darknet = self._create_conv_layers(config) # Darknet-19
        self.fcs = self._create_fcs(S=grid_size, num_boxes=num_boxes, num_classes=classes)
        
    def _create_conv_layers(self, config):
        layers = []
        in_channels = self.in_channels
        
        for layer in config:
            if type(layer) == tuple:
                layers.append(CNNBlock(in_channels=in_channels, 
                                       out_channels=layer[1], 
                                       kernel_size=layer[0], 
                                       stride=layer[2], 
                                       padding=layer[3]))
                in_channels = layer[1]  # Update in_channels for next layer
            elif type(layer) == list:
                num_repeats = layer[-1]
                for _ in range(num_repeats):    # Repeat layer num_repeats times
                    for sublayer in layer[:-1]: # Iterate through all sublayers
                        if type(sublayer) == int: 
                            continue
                        layers.append(CNNBlock(in_channels=in_channels,
                                                out_channels=sublayer[1],
                                                kernel_size=sublayer[0],
                                                stride=sublayer[2],
                                                padding=sublayer[3]))
                        in_channels = sublayer[1] # Update in_channels for next layer
            elif type(layer) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = in_channels # Update in_channels for next layer
                
        return nn.Sequential(*layers)
    
    def _create_fcs(self, S, num_boxes, num_classes):
        S, B, C = S, num_boxes, num_classes # B is 2 in paper, C is 20 in paper
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # Paper uses 4096 neurons
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)) # (S, S, 30) where 30 = (C + B * 5)
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

class YoloV1Loss(nn.Module):
    """ YoloV1 Loss Function"""
    def __init__(self, S=7, B=2, C=20):
        super(YoloV1Loss, self).__init__()
        
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def _intersection_over_union(self, boxes_preds, boxes_labels):
        pass
        
    def forward(self, predictions, target):
        # target is of shape (N, S, S, 30)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) # (N, S, S, 30)
        
        b1_pred_xywh = predictions[..., 21:25] # (N, S, S, 4)
        b2_pred_xywh = predictions[..., 26:30] # (N, S, S, 4)
        targ_xywh = target[..., 21:25] # (N, S, S, 4)
        
        iou_b1 = self._intersection_over_union(b1_pred_xywh, targ_xywh)
        iou_b2 = self._intersection_over_union(b2_pred_xywh, targ_xywh)
    
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # (2, N, S, S)
        iou_maxes, best_box = torch.max(ious, dim=0) # (N, S, S)
        # Iobj_i, tells if object exists in cell i
        exists_box = target[..., 20].unsqueeze(3) # (N, S, S, 1)
        
# Architecture configuration
config = [
    # Tuple: (kernel_size, filters, stride, padding)
    (7, 64, 2, 3),
    "M",                # MaxPool2d(kernel_size=2, stride=2)
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: [(kernel_size, filters, stride, padding), # of repeats]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# model = YOLOV1(config=config, in_channels=3, classes=20, grid_size=7, num_boxes=2)
# summary.summary(model, input_size=(3, 448, 448), device='cpu')