import matplotlib.pyplot as plt
from torchvision import utils
from torch import nn
from torchvision import models
import numpy as np




def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    print(tensor.shape)
    n,c,w,h = tensor.shape # number of filters, channels, width, height

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


layer = 0
model = models.AlexNet()
print(model)


filter = model.features[layer].weight.data.clone()
visTensor(filter, ch=0, allkernels=False)


plt.axis("off")
plt.ioff()
plt.savefig("images/08_CNNDeconvAct_L{}.png".format(layer), bbox_inches="tight")

