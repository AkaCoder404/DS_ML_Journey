"""
Title: U-Net: Convolutional Networks for Biomedical Image Segmentation

Description: Implementation of UNet Model on different datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image


class UNetNoSkip(nn.Module):
    """ UNet Model without Skip Connections """
    def __init__(self, in_channels, out_channels):
        super(UNetNoSkip, self).__init__()

        # Encoder -> Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder -> Upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # Peform Sigmoid at the end of the model to get values between 0 and 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)     
        # Decoder
        x2 = self.decoder(x1)
        return x2

class UNet(nn.Module):
    """ Original UNet Model with Skip Connections """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # With Skip Connections
        def conv_block(in_channels, out_channels, padding1=0, padding2=0):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding2)
            )
            
        self.enc_conv0 = conv_block(in_channels=in_channels, out_channels=64)
        self.enc_conv1 = conv_block(64, 128)
        self.enc_conv2 = conv_block(128, 256)
        self.enc_conv3 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bottleneck_conv0 = conv_block(512, 1024, padding2=1)
        self.bottleneck_conv1 = conv_block(1024, 1024, padding2=1)
        
        self.relu = nn.ReLU()
        
        self.dec_conv3 = conv_block(1024, 512)
        self.dec_conv2 = conv_block(512, 256)
        self.dec_conv1 = conv_block(256, 128)
        self.dec_conv0 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
            
    def crop(self, enc, crop_size, targt_size):
        # Peform copy and crop  
        start =  crop_size // 2  
        end = start + targt_size          
        print(enc[:, :, start:end, start:end].shape)
        return enc[:, :, start:end, start:end]
        
    def conv_half(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)      

    def forward(self, x):
        print("Input:", x.shape)                            # (3, 572, 572)
        enc0 = self.enc_conv0(x)                            # (64, 568, 568)
        print("enc0", enc0.shape)
        enc1 = self.enc_conv1(self.pool(enc0))              # (128, 280, 280)
        print("enc1", enc1.shape)
        enc2 = self.enc_conv2(self.pool(enc1))              # (256, 136, 136)
        print("enc2", enc2.shape)
        enc3 = self.enc_conv3(self.pool(enc2))              # (512, 64, 64)
        print("enc3", enc3.shape)

        bottleneck0 = self.pool(enc3)                       # (512, 32, 32)
        print("Bottleneck0", bottleneck0.shape)
        bottleneck1 = self.bottleneck_conv0(bottleneck0)    # (1024, 30, 30)
        print("Bottleneck1", bottleneck1.shape)
        bottleneck2 = self.bottleneck_conv1(bottleneck1)    # (1024, 28, 28)
        print("Bottleneck2", bottleneck2.shape)

        dev_conv3_crop      = self.crop(enc3, 4, 56)                          # (512, 56, 56)
        dev_conv3_up_conv   = self.conv_half(1024, 512)(self.up(bottleneck2)) # (512, 56, 56)
    
    
        dec3 = self.dec_conv3(torch.cat([dev_conv3_crop, dev_conv3_up_conv], 1))
        print("dec3", dec3.shape)
        
        dev_conv2_crop      = self.crop(enc2, 16, 104)                         # (256, 104, 104)
        dev_conv2_up_conv   = self.conv_half(512, 256)(self.up(dec3))          # (256, 104, 104)
        
        dec2 = self.dec_conv2(torch.cat([dev_conv2_crop, dev_conv2_up_conv], 1))
        print("dec2", dec2.shape)
        
        dev_conv1_crop      = self.crop(enc1, 40, 200)                         # (128, 200, 200)
        dev_conv1_up_conv   = self.conv_half(256, 128)(self.up(dec2))          # (128, 200, 200)
        
        dec1 = self.dec_conv1(torch.cat([dev_conv1_crop, dev_conv1_up_conv], 1))
        print("dec1", dec1.shape)
        
        dev_conv0_crop      = self.crop(enc0, 88, 392)                              # (64, 392, 392)
        dev_conv0_up_conv   = self.conv_half(128, 64)(self.up(dec1))                # (64, 392, 392)
        
        dec0 = self.dec_conv0(torch.cat([dev_conv0_crop, dev_conv0_up_conv], 1))    #（64， 392， 392）
        print("dec0", dec0.shape)

        return self.final_conv(dec0)                                                # (2, 388, 388)


class UNetPadded(nn.Module):
    """UNet Model but with padding to keep the same size as input """
    def __init__(self, in_channels, out_channels):
        super(UNetPadded, self).__init__()
        
        # With Skip Connections
        def conv_block(in_channels, out_channels, padding1=0, padding2=0):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding2)
            )
                        
        self.enc_conv0 = conv_block(in_channels=in_channels, out_channels=64, padding1=1, padding2=1)
        self.enc_conv1 = conv_block(64, 128, 1, 1)
        self.enc_conv2 = conv_block(128, 256, 1, 1)
        self.enc_conv3 = conv_block(256, 512, 1, 1)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bottleneck_conv0 = conv_block(512, 1024, padding1=1, padding2=1)
        self.bottleneck_conv1 = conv_block(1024, 1024, padding1=1, padding2=1)
        
        self.relu = nn.ReLU()
        
        self.dec_conv3 = conv_block(1024, 512, padding1=1, padding2=1)
        self.dec_conv2 = conv_block(512, 256, padding1=1, padding2=1)
        self.dec_conv1 = conv_block(256, 128, padding1=1, padding2=1)
        self.dec_conv0 = conv_block(128, 64, padding1=1, padding2=1)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
            
    def crop(self, enc, crop_size, targt_size):
        # Peform copy and crop  
        start =  crop_size // 2  
        end = start + targt_size          
        # print(enc[:, :, start:end, start:end].shape)
        return enc[:, :, start:end, start:end]
        
    def conv_half(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)      

    def forward(self, x):
        # print("Input:", x.shape)                          # (3, 224, 224)
        enc0 = self.enc_conv0(x)                            # (64, 112, 112)
        # print("enc0", enc0.shape)
        enc1 = self.enc_conv1(self.pool(enc0))              # (128, 56, 56)
        # print("enc1", enc1.shape)
        enc2 = self.enc_conv2(self.pool(enc1))              # (256, 28, 28)
        # print("enc2", enc2.shape)
        enc3 = self.enc_conv3(self.pool(enc2))              # (512, 14, 14)
        # print("enc3", enc3.shape)

        bottleneck0 = self.pool(enc3)                       # (512, 14, 14)
        # print("Bottleneck0", bottleneck0.shape)
        bottleneck1 = self.bottleneck_conv0(bottleneck0)    # (1024, 14, 14)
        # print("Bottleneck1", bottleneck1.shape)
        bottleneck2 = self.bottleneck_conv1(bottleneck1)    # (1024, 14, 14)
        # print("Bottleneck2", bottleneck2.shape)

        dec3_conv_half = self.conv_half(1024, 512)(self.up(bottleneck2))
        dec3 = self.dec_conv3(torch.cat([enc3, dec3_conv_half], 1)) # ()
        # print("dec3", dec3.shape)

        dec2_conv_half = self.conv_half(512, 256)(self.up(dec3))
        dec2 = self.dec_conv2(torch.cat([enc2, dec2_conv_half ], 1))
        # print("dec2", dec2.shape)
        
        dec1_conv_half = self.conv_half(256, 128)(self.up(dec2))
        dec1 = self.dec_conv1(torch.cat([enc1, dec1_conv_half], 1))
        # print("dec1", dec1.shape)
        
        dec0_conv_half = self.conv_half(128, 64)(self.up(dec1))
        dec0 = self.dec_conv0(torch.cat([enc0, dec0_conv_half], 1))    #（64， 392， 392）
        # print("dec0", dec0.shape)

        return self.final_conv(dec0)                                                # (2, 388, 388)


class TGSCustomDataset(Dataset):
    """
    Custom Dataset for TGS Salt Identification Challenge
    
    4000 Training Images
    18000 Test Images
    """
    def __init__(self, root, image_folder, mask_folder, transform=None):
        """
        Args:
            root (string): Root directory path.
            image_folder (string): Folder containing the images.
            mask_folder (string): Folder containing the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.images = list(sorted(os.listdir(os.path.join(root, image_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_folder))))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_folder, self.images[idx])
        mask_path = os.path.join(self.root, self.mask_folder, self.masks[idx])
        
        # Image open without alpha channel
        image = Image.open(img_path).convert("RGB")
        # Mask open black and white
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]
    
    
if __name__ == "__main__":
    # Pass random tensor through model
    from torchsummary import summary
    random_tensor = torch.randn(3, 572, 572)
    model = UNetPadded(3, 1)
    summary(model, (3, 224, 224))
    
    