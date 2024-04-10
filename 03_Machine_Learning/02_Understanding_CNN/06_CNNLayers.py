import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
# Print out the model's architecture png
from torchviz import make_dot


from PIL import Image

# Load a pre-trained model (e.g., VGG16)
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
print(model)
model.eval()

# Specify the layer from which you want to visualize feature maps
layer = 7  # Change this to the desired layer
target_layer = model.features[layer]  # Change this to the desired layer



# Load and preprocess an example image
# Replace 'example_image.jpg' with the path to your image
image_path = 'images/lena_256.jpg'
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = Image.open(image_path)
image = preprocess(image).unsqueeze(0) # Unsqueeze to add artificial first dimension

# Register a forward hook to capture feature maps
activation = {}

def hook_fn(module, input, output):
    activation['value'] = output

hook = target_layer.register_forward_hook(hook_fn)

# Forward pass to compute feature maps
with torch.no_grad():
    print(image.shape)
    y = model(image)
    print(y.shape)
    _, predicted = torch.max(y.data, 1)
    
    
    
    # Print out the label of the predicted class
    print('Predicted class: {}'.format(y.argmax(dim=1).item()))
    # Print out name of predicted class
    print('Label Name: ', _, predicted)
 
    # Print the NN architecture
    make_dot(y.mean(), params=dict(model.named_parameters())).render("images/06_CNNLayers", format="png")
    

# Remove the hook
hook.remove()

# Get the feature maps from the hook
feature_maps = activation['value'][0]

# Plot the feature maps
num_feature_maps = feature_maps.size(0)
rows = int(num_feature_maps**0.5)
cols = int(num_feature_maps / rows)
fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

# Plot the feature maps in the grid
for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        if index < num_feature_maps:
            ax = axs[i, j]
            ax.imshow(feature_maps[index].cpu(), cmap='viridis')
            ax.axis('off')
            ax.set_title(f'FM {index+1}')

# Remove any empty subplots
for i in range(num_feature_maps, rows * cols):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.savefig('images/06_feature_maps_{}.png'.format(layer))