"""
Title: Image Data Loaders
Description: Image data loaders for many different types of datasets

Notes:
- ImageNet
- CIFAR-10
- CIFAR-100
- MNIST
- FashionMNIST
- STL-10
- SVHN
"""

import torch
import torchvision
import torchvision.transforms as transforms

# TODO - Add more datasets
# TODO - Data augmentation options


class MNISTDataLoader():
    def __init__(self, batch_size):
        """
        Constructor for the MNISTDataLoader class.
        @param batch_size: The batch size for the data loader.
        """
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def get_data_loaders(self):
        """
        Get the training and test data loaders for MNIST.
        @return: The training and test data loaders.
        """
        train_loader = self.get_train_loader()
        test_loader = self.get_test_loader()
        return train_loader, test_loader
    
    def get_train_loader(self):
        """
        Get the training data loader for MNIST.
        @return: The training data loader.
        """
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader
    
    def get_test_loader(self):
        """
        Get the test data loader for MNIST.
        @return: The test data loader.
        """
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
    

