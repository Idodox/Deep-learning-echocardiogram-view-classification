import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from helper_methods import pickle_loader, get_num_correct
from functools import partial
import numpy as np
# import matplotlib.pyplot as plt
from network_architectures import cnn_3d_1
from tqdm import tqdm

from pathlib import Path
from comet_ml import Experiment

# Create an experiment
experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1"
                        ,project_name="thesis"
                        ,workspace="idodox")

# Exp
# some_param = "some value"
# experiment.log_parameter("param name", some_param)

# train_accuracy = "some value"
# experiment.log_metric("acc", train_accuracy)

ROOT_PATH = str(Path.home()) + "/Documents/Thesis/Data/frames/5frame_steps"

network = cnn_3d_1()



n_epochs = 50
lr = 1e-06
batch_size = 30

data_transforms = transforms.Compose([
    # TODO: normalize
    transforms.ToTensor()
])

# Load the dataset with ImageFolder:
data_set = torchvision.datasets.DatasetFolder(ROOT_PATH
                                # , transform = data_transforms
                                , loader = partial(pickle_loader, max_frames = 10)
                                , extensions = ('.pickle')
                                )

# Create a dataloader object so we can get the data in batches:
loader = torch.utils.data.DataLoader(data_set
                                     , batch_size=batch_size
                                     , shuffle=True
                                     # ,batch_sampler =  # TODO: add stratified sampling
                                     , num_workers=0
                                     , drop_last=False
                                     )

# Experiment hyperparameter log
hyper_params = {"learning_rate": lr
               ,"n_epochs": n_epochs
               ,"batch_size": batch_size
               ,"normalized_data": False
               ,"stratified": False
               ,"max_frames": 10
               ,"dataset": ""}
experiment.log_parameters(hyper_params)


batch = next(iter(loader))
images, labels = batch
print(labels)
images = torch.unsqueeze(images, 1) # added channel dimensions (grayscale)

optimizer = optim.Adam(network.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

for epoch in tqdm(range(n_epochs)):

    preds = network(images) # Pass Batch
    loss = F.cross_entropy(preds, labels) # Calculate Loss

    optimizer.zero_grad() # Whenever pytorch calculates gradients it always adds it to whatever it has, so we need to reset it each batch.
    loss.backward() # Calculate Gradients - the gradient is the direction we need to move towards the loss function minimum (LR will tell us how far to step)
    optimizer.step() # Update Weights - the optimizer is able to update the weights because we passed it the weights as an argument in line 4.

    # print(preds, labels)
    print('Epoch:', epoch, 'num correct:', get_num_correct(preds, labels), 'Percent correct:', "{:.2%}".format(get_num_correct(preds, labels)/len(labels)), 'Batch loss:', loss.item())