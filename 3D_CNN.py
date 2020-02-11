from comet_ml import Experiment
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

# Create an experiment
experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1"
                        ,project_name="thesis"
                        ,workspace="idodox")

# Experiment hyperparameters
hyper_params = {"learning_rate": 1e-06
               ,"n_epochs": 50
               ,"batch_size": 30
               ,"normalized_data": False
               ,"stratified": False
               ,"max_frames": 10
               ,"dataset": "5frame_steps"
               ,"resolution": 100
               ,"conv1_in_ch": 1
               ,"conv1_out_ch": 16
               ,"conv1_kernel": 3
               ,"bn1_n_features": 16
               ,"conv2_in_ch": 16
               ,"conv2_out_ch": 16
               ,"conv2_kernel": 3
               ,"bn2_n_features": 16
               ,"conv3_in_ch": 16
               ,"conv3_out_ch": 16
               ,"conv3_kernel": 3
               ,"bn3_n_features": 16
               ,"maxpool1_kernel": 2
               ,"fc1_size": 1000
               ,"fc2_size": 1000
                }

experiment.log_parameters(hyper_params)


data_transforms = transforms.Compose([
    # TODO: normalize
    transforms.ToTensor()
])

# Load the dataset with ImageFolder:
# noinspection PyTypeChecker

ROOT_PATH = str(Path.home()) + "/Documents/Thesis/Data/frames/" + hyper_params['dataset']

data_set = torchvision.datasets.DatasetFolder(ROOT_PATH
                                # , transform = data_transforms
                                , loader = partial(pickle_loader, max_frames = hyper_params['max_frames'])
                                , extensions = '.pickle'
                                )

# Create a dataloader object so we can get the data in batches:
loader = torch.utils.data.DataLoader(data_set
                                     , batch_size=hyper_params['batch_size']
                                     , shuffle=True
                                     # ,batch_sampler =  # TODO: add stratified sampling
                                     , num_workers=0
                                     , drop_last=False
                                     )


network = cnn_3d_1(hyper_params['max_frames'], hyper_params['resolution'], hyper_params['conv1_in_ch'],
                   hyper_params['conv1_out_ch'], hyper_params['conv1_kernel'], hyper_params['bn1_n_features'],
                   hyper_params['conv2_in_ch'], hyper_params['conv2_out_ch'], hyper_params['conv2_kernel'],
                   hyper_params['bn2_n_features'], hyper_params['conv3_in_ch'], hyper_params['conv3_out_ch'],
                   hyper_params['conv3_kernel'], hyper_params['bn3_n_features'], hyper_params['maxpool1_kernel'],
                   hyper_params['fc1_size'], hyper_params['fc2_size'])

batch = next(iter(loader))
images, labels = batch
print(labels)
images = torch.unsqueeze(images, 1) # added channel dimensions (grayscale)

optimizer = optim.Adam(network.parameters(), lr=hyper_params['learning_rate'])
criterion = nn.CrossEntropyLoss()


# train_accuracy = "some value"
# experiment.log_metric("acc", train_accuracy)

with experiment.train():
    for epoch in tqdm(range(hyper_params["n_epochs"])):

        optimizer.zero_grad() # Whenever pytorch calculates gradients it always adds it to whatever it has, so we need to reset it each batch.
        preds = network(images) # Pass Batch
        loss = criterion(preds, labels) # Calculate Loss
        loss.backward() # Calculate Gradients - the gradient is the direction we need to move towards the loss function minimum (LR will tell us how far to step)
        optimizer.step() # Update Weights - the optimizer is able to update the weights because we passed it the weights as an argument in line 4.

        num_correct = get_num_correct(preds, labels)

        experiment.log_metric("accuracy", num_correct/len(labels)*100)
        experiment.log_metric("CrossEntropyLoss", loss.item())


        # print(preds, labels)
        print('Epoch:', epoch, 'num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct/len(labels)), 'Batch loss:', loss.item())