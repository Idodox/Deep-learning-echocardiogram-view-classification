import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter


from network_architectures import TestNetwork
from helper_methods import get_num_correct


from PIL import Image


torch.set_printoptions(linewidth=120)
torch.set_num_threads = 16

# import torchvision
# from collections import OrderedDict
# from collections import namedtuple
# from itertools import product
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt


ROOT_PATH = '../Data/frames'

network = TestNetwork()

data_transforms = transforms.Compose([
    transforms.CenterCrop(300)
    , transforms.Resize(100)
    , transforms.ToTensor()
])

# Load the dataset with ImageFolder:
data_set = datasets.ImageFolder(ROOT_PATH
                                , transform=data_transforms
                                , loader=Image.open
                                )

# Create a dataloader object so we can get the data in batches:
loader = torch.utils.data.DataLoader(data_set
                                     , batch_size=1000
                                     , shuffle=True
                                     # ,batch_sampler =  # TODO: add stratified sampling
                                     , num_workers=0
                                     , drop_last=False
                                     )

optimizer = optim.Adam(network.parameters(), lr=0.01)


# Overfitting single batch

network = TestNetwork()
batch = next(iter(loader))
images, labels = batch




for epoch in range(10):

    preds = network(images) # Pass Batch
    loss = F.cross_entropy(preds, labels) # Calculate Loss

    optimizer.zero_grad() # Whenever pytorch calculates gradients it always adds it to whatever it has, so we need to reset it each batch.
    loss.backward() # Calculate Gradients - the gradient is the direction we need to move towards the loss function minimum (LR will tell us how far to step)
    optimizer.step() # Update Weights - the optimizer is able to update the weights because we passed it the weights as an argument in line 4.

    print('Epoch:', epoch, 'num correct:', get_num_correct(preds, labels), 'Batch loss:', loss.item())