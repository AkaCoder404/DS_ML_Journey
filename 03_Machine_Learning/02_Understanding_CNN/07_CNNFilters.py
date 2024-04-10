import torch
import matplotlib.pyplot as plt
from torchvision import models
from torchviz import make_dot

# Load a pre-trained model (e.g., VGG16)
layer = 28
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.eval()

# Specify the layer from which you want to visualize filters
target_layer = model.features[layer]  # Change this to the desired layer

# Get the filter weights from the specified layer
filter_weights = target_layer.weight.data

# Create a grid of subplots to display the filters
num_filters = filter_weights.size(0)
rows = int(num_filters**0.5)
cols = int(num_filters / rows)
fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

# Use torchviz to visualize the network architecture
sample_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 input size
# Pass the input through the model to create a computation graph
output = model(sample_input)
# Use torchviz to create a visualization of the computation graph
dot = make_dot(output, params=dict(model.named_parameters()))
# Save the visualization to a file (e.g., PNG)
dot.render("model_graph", format="png")

# Plot the filter weights in the grid
for i in range(rows):
    for j in range(cols):
        index = i * cols + j
        if index < num_filters:
            ax = axs[i, j]
            filter_data = filter_weights[index].cpu().numpy()
            ax.imshow(filter_data[0], cmap='gray')
            ax.axis('off')
            ax.set_title(f'F {index+1}')

# Remove any empty subplots
for i in range(num_filters, rows * cols):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.savefig('images/07_CNNFilters_L{}.png'.format(layer))
