from comet_ml import Experiment
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms
from functools import partial
from tqdm import tqdm
from pathlib import Path

from network_architectures import cnn_3d_1
from helper_methods import pickle_loader, get_num_correct, get_train_val_idx


# Comet ML experiment
experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1" ,project_name="thesis" ,workspace="idodox")

hyper_params = {"learning_rate": 0.00001
               ,"n_epochs": 50
               ,"batch_size": 30
               ,"num_workers": 3
               ,"normalized_data": False
               ,"stratified": True
               ,"max_frames": 10
               ,"dataset": "5frame_steps"
               ,"resolution": 100
               ,"conv1_in_ch": 1
               ,"conv1_out_ch": 32
               ,"conv1_kernel": 3
               ,"bn1_n_features": 32
               ,"conv2_in_ch": 32
               ,"conv2_out_ch": 16
               ,"conv2_kernel": 3
               ,"bn2_n_features": 16
               ,"conv3_in_ch": 16
               ,"conv3_out_ch": 8
               ,"conv3_kernel": 3
               ,"bn3_n_features": 8
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

network = cnn_3d_1(hyper_params['max_frames'], hyper_params['resolution'], hyper_params['conv1_in_ch'],
                   hyper_params['conv1_out_ch'], hyper_params['conv1_kernel'], hyper_params['bn1_n_features'],
                   hyper_params['conv2_in_ch'], hyper_params['conv2_out_ch'], hyper_params['conv2_kernel'],
                   hyper_params['bn2_n_features'], hyper_params['conv3_in_ch'], hyper_params['conv3_out_ch'],
                   hyper_params['conv3_kernel'], hyper_params['bn3_n_features'], hyper_params['maxpool1_kernel'],
                   hyper_params['fc1_size'], hyper_params['fc2_size'])

ROOT_PATH = str(Path.home()) + "/Documents/Thesis/Data/frames/" + hyper_params['dataset']

master_data_set = torchvision.datasets.DatasetFolder(ROOT_PATH
                                # , transform = data_transforms
                                , loader = partial(pickle_loader, max_frames = hyper_params['max_frames'])
                                , extensions = '.pickle'
                                )

train_idx, val_idx = get_train_val_idx(master_data_set)


train_set = torch.utils.data.Subset(master_data_set, train_idx)
val_set = torch.utils.data.Subset(master_data_set, val_idx)

train_loader = torch.utils.data.DataLoader(train_set
                                     , batch_size=hyper_params['batch_size']
                                     , shuffle=True
                                     # ,batch_sampler =  # TODO: add stratified sampling
                                     , num_workers=hyper_params['num_workers']
                                     , drop_last=False
                                     )

val_loader = torch.utils.data.DataLoader(val_set
                                     , batch_size=hyper_params['batch_size']
                                     , shuffle=True
                                     # ,batch_sampler =  # TODO: add stratified sampling
                                     , num_workers=hyper_params['num_workers']
                                     , drop_last=False
                                     )


optimizer = optim.Adam(network.parameters(), lr=hyper_params['learning_rate'])
criterion = nn.CrossEntropyLoss()

log_number_train = log_number_val = 0


for epoch in tqdm(range(hyper_params["n_epochs"])):

    total_train_loss = 0
    total_train_correct = 0

    network.train()
    for batch_number, (images, labels) in tqdm(enumerate(train_loader)):

        images = torch.unsqueeze(images, 1)  # added channel dimensions (grayscale)

        optimizer.zero_grad() # Whenever pytorch calculates gradients it always adds it to whatever it has, so we need to reset it each batch.
        preds = network(images) # Pass Batch
        loss = criterion(preds, labels) # Calculate Loss
        total_train_loss += loss.item()
        loss.backward() # Calculate Gradients - the gradient is the direction we need to move towards the loss function minimum (LR will tell us how far to step)
        optimizer.step() # Update Weights - the optimizer is able to update the weights because we passed it the weights as an argument in line 4.

        num_correct = get_num_correct(preds, labels)
        total_train_correct += num_correct

        experiment.log_metric("Train batch accuracy", num_correct/len(labels)*100, step = log_number_train)
        experiment.log_metric("Train batch CrossEntropyLoss", loss.item(), step = log_number_train)
        log_number_train += 1

        print('Train: Batch number:', batch_number, 'Num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct/len(labels)), 'Loss:', loss.item())

    experiment.log_metric("Train epoch accuracy", total_train_correct/len(train_loader.dataset)*100, step = epoch)
    experiment.log_metric("Train epoch CrossEntropyLoss", total_train_loss, step = epoch)
    print('Train: Epoch:', epoch, 'num correct:', total_train_correct, 'Accuracy:', "{:.2%}".format(total_train_correct/len(train_loader.dataset)), 'Batch loss:', total_train_loss)


    total_val_loss = 0
    total_val_correct = 0

    network.eval()
    with torch.no_grad():
        for batch_number, (images, labels) in tqdm(enumerate(val_loader)):
            images = torch.unsqueeze(images, 1)  # added channel dimensions (grayscale)

            preds = network(images)  # Pass Batch
            loss = criterion(preds, labels)  # Calculate Loss
            total_val_loss += loss.item()

            num_correct = get_num_correct(preds, labels)
            total_val_correct += num_correct

            experiment.log_metric("Val batch accuracy", num_correct / len(labels) * 100, step=log_number_val)
            experiment.log_metric("Val batch CrossEntropyLoss", loss.item(), step=log_number_val)
            log_number_val += 1

            print('Val: Batch number:', batch_number, 'Num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct / len(labels)), 'Loss:', loss.item())

        experiment.log_metric("Val epoch accuracy", total_val_correct / len(val_loader.dataset) * 100, step=epoch)
        experiment.log_metric("Val epoch CrossEntropyLoss", total_val_loss, step=epoch)
        print('Val Epoch:', epoch, 'num correct:', total_val_correct, 'Accuracy:',
              "{:.2%}".format(total_val_correct / len(val_loader.dataset)), 'Batch loss:', total_val_loss)
