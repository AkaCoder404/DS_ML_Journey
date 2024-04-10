import os
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
import time
from tqdm import tqdm

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# Import the model
from MobileNetV1 import MobileNetV1, layers_config

imagenette_dict = {
    'n01440764': 0,  # Class 1
    'n02102040': 1,  # Class 2
    'n02979186': 2,  # Class 3
    'n03000684': 3,  # Class 4
    'n03028079': 4,  # Class 5
    'n03394916': 5,  # Class 6
    'n03417042': 6,  # Class 7
    'n03425413': 7,  # Class 8
    'n03445777': 8,  # Class 9
    'n03888257': 9   # Class 10
}

imagenet_classes = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute'
)

# Load the dataset
class CustomDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        # label = int(self.data.iloc[idx, 1][1:])
        label = imagenette_dict[self.data.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)

        return image, label

data_dir = os.path.expanduser('~/Developer/Datasets/imagenette2')
csv_file = os.path.join(data_dir, 'noisy_imagenette.csv')  # Replace 'your_dataset.csv' with the actual CSV file name
transform = T.Compose([
    T.Resize((224, 224)),
    lambda image: image.convert('RGB') if image.mode == 'L' else image,
    T.ToTensor()
    ])
custom_dataset = CustomDataset(csv_path=csv_file, data_dir=data_dir, transform=transform)

print("Dataset Size", len(custom_dataset))

# TODO Show more images

# Show an example image and its corresponding label
image, label = custom_dataset[13393]
print(image.shape, label)
# Print out original path of the image
print(custom_dataset.data.iloc[13393, 0])

numpy_array = image.numpy()
plt.imshow(numpy_array.transpose(1, 2, 0))  
plt.title(label)
plt.savefig("test.png")


# # Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
print(len(train_dataset), len(test_dataset))

# # Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32)

sample = next(iter(train_loader))

# Load the model
model = MobileNetV1(layers_config, 3, 10)
summary(model, (3, 224, 224), device='cpu')

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 10
loss_fn = nn.CrossEntropyLoss()
criterion = torch.optim.Adam(model.parameters(), lr=learning_rate)


print(f"Training using {device}")
model.to(device)

# Calculate accuracy
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def training_step(model : nn.Module, 
        train_loader : DataLoader,
        loss_fn: nn.Module,
        optimizer : torch.optim.Optimizer,
        epoch : int,
        ):
    
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)               # Forward pass
        loss = loss_fn(y_pred, y)       # Calculate loss
        optimizer.zero_grad()           # Zero out gradients
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        train_acc +=  (y_pred.argmax(dim=1) == y).sum().item() / len(y) * 100
        train_loss += loss.item()
        
        # Clear memory
        # torch.mps.empty_cache()
        torch.cuda.empty_cache()
        
    return train_loss / len(train_loader), train_acc / len(train_loader)
    
def testing_step(model : nn.Module, 
        test_loader : DataLoader,
        loss_fn: nn.Module,
        ):
    
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        model.eval()
        for batch, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)               # Forward pass
            loss = loss_fn(y_pred, y)       # Calculate loss 
            test_acc += (y_pred.argmax(dim=1) == y).sum().item() / len(y) * 100
            test_loss += loss.item()
            
    return test_loss / len(test_loader), test_acc / len(test_loader)
    
# Train the model
def train():
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(epochs):
        print("Start Epoch 1...")
        train_loss, train_acc = training_step(model, train_loader, loss_fn, criterion, epoch)
        test_loss, test_acc = testing_step(model, val_loader, loss_fn)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss {train_loss}, Train Acc {train_acc}, Test Loss {test_loss}, Test Acc {test_acc}")
    
    return history
        

train()

# Save the model
torch.save(model.state_dict(), "MobileNetV1_imagenette_2.pth")

# Save history
