"""
Title: PyTorch Train Template
Description: Template for training a PyTorch models
"""
import sys

import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
import tqdm
import logging

def train_step(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Perform a single training step.
    """
    model.train()
    total_acc, total_loss, total_batch_time  = 0.0, 0.0, 0.0
    for batch_idx, (data, targets) in enumerate(tqdm.tqdm(train_loader)):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (outputs.argmax(dim=1) == targets).sum().item()
        
    return total_acc / len(train_loader), total_loss / len(train_loader)
    
                
def test_step(model, test_loader, criterion, device='cpu'):
    """
    Perform a single testing step.
    """
    model.eval()
    total_acc, total_loss, total_batch_time = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm.tqdm(test_loader)):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_acc += (outputs.argmax(dim=1) == targets).sum().item()
            
    return total_acc / len(test_loader), total_loss / len(test_loader)
    


def training(model, train_loader, test_loader, criterion, optimizer, logger, num_epochs=5):
    """
    Perform the training loop.
    """
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_acc, train_loss = train_step(model, train_loader, criterion, optimizer)
        test_acc, test_loss = test_step(model, test_loader, criterion)
        
        logger.info(f"Train Accuracy: {train_acc}, Train Loss: {train_loss}")
        logger.info(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")
        
        # TODO - Add tensorboard logging
        # TODO - Add model checkpointing
        # TODO - Add early stopping
        # TODO - Add learning rate scheduling
        # TODO - Add gradient clipping
        # TODO - Add model saving
        # TODO - Add time taken per epoch
        
    logger.info("Training finished")
    logger.info("-" * 50)
                

class Training():
    def __init__(self, 
                 model, 
                 train_loader, 
                 test_loader, 
                 criterion, 
                 optimizer, 
                 num_epochs, 
                 new_train_dir,
                 device='cpu'
        ):
        """
        Constructor for the Training class.
        @param model: The model to train.
        @param train_loader: The training data loader.
        @param test_loader: The test data loader.
        @param criterion: The loss function.
        @param optimizer: The optimizer.
        @param num_epochs: The number of epochs to train.
        @param new_train_dir: The directory to save the training logs.
        @param device: The device to train on.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.new_train_dir = new_train_dir
        
        # Logging
    

    def start_training(self):
        """
        Start the training process.
        """
        logger = logging.getLogger("Model")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.addHandler(logging.FileHandler(self.new_train_dir + '/train.log'))
        
        training(model = self.model, 
            train_loader = self.train_loader,
            test_loader = self.test_loader,
            criterion = self.criterion,
            optimizer = self.optimizer,
            logger = logger,
            num_epochs = self.num_epochs
        )