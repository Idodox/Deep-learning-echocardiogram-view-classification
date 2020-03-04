from comet_ml import Experiment
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms
from functools import partial
from tqdm import tqdm
from pathlib import Path

from model_architectures import cnn_3d_1
from torchsummary import summary
from torchutils import pickle_loader, get_num_correct, get_train_val_idx, DatasetFolderWithPaths, get_mistakes, online_mean_and_std, calc_accuracy, get_cross_val_idx

torch.backends.cudnn.benchmark=True
print('CUDA available:', torch.cuda.is_available())
print('CUDA enabled:', torch.backends.cudnn.enabled)

log_data = True

#TODO: update get_cross_val_idx with "random_state" (from hyper params)
hyper_params = {"learning_rate": 0.00001
               ,"n_epochs": 100
               ,"batch_size": 50
               ,"num_workers": 4
               ,"normalized_data": True
               ,"stratified": True
               ,"max_frames": 10
               ,"dataset": "5frame_steps"
               ,"resolution": 100
               ,"conv1_ch": 128
               ,"conv1_kernel": (3, 9, 9)
               ,"conv2_ch": 64
               ,"conv2_kernel": (3, 9, 9)
               ,"conv3_ch": 32
               ,"conv3_kernel": (3, 5, 5)
               ,"conv4_ch": 16
               ,"conv4_kernel": (1, 3, 3)
               ,"last_maxpool_kernel": 3
               ,"fc1_size": 512
               ,"dropout1_ratio": 0.6
               ,"fc2_size": 256
               ,"dropout2_ratio": 0.6
               ,"fc3_size": 128
               ,"dropout3_ratio": 0.6
                }


model = cnn_3d_1(hyper_params)
model = model.cuda()


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

# Log number of parameters
hyper_params['trainable_params'] = sum(p.numel() for p in model.parameters())
print('N_trainable_params:', hyper_params['trainable_params'])

data_transforms = transforms.Compose([
    transforms.ToTensor()
    ,transforms.Normalize([53.91100598, 54.00132478, 54.09308712, 54.09459359, 54.12711804, 54.13030674, 54.09839364, 54.03708794, 53.8983994, 53.75836842]
                          , [54.49093735, 54.5142583, 54.56545188, 54.56279357, 54.56717128, 54.57554804, 54.55975601, 54.55826991, 54.53283708, 54.50516662])
])

ROOT_PATH = str("/home/ido/data/" + hyper_params['dataset'])

master_data_set = DatasetFolderWithPaths(ROOT_PATH
                                # , transform = data_transforms
                                , loader = partial(pickle_loader, max_frames = hyper_params['max_frames'])
                                , extensions = '.pickle'
                                )

if log_data:
    # Comet ML experiment
    experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1" ,project_name="thesis" ,workspace="idodox")
    experiment.log_parameters(hyper_params)

fold_indexes= get_cross_val_idx(master_data_set)
curr_fold = 0
acc_list_train = []
acc_list_val = []

for train_idx, val_idx in fold_indexes:


    train_set = torch.utils.data.Subset(master_data_set, train_idx)
    val_set = torch.utils.data.Subset(master_data_set, val_idx)

    train_loader = torch.utils.data.DataLoader(train_set
                                         , batch_size=hyper_params['batch_size']
                                         , shuffle=True
                                         # ,batch_sampler =  # TODO: add stratified sampling
                                         , num_workers=hyper_params['num_workers']
                                         , drop_last=False
                                         )

    # online_mean_and_std(train_loader)

    val_loader = torch.utils.data.DataLoader(val_set
                                         , batch_size=hyper_params['batch_size']
                                         , shuffle=True
                                         # ,batch_sampler =  # TODO: add stratified sampling
                                         , num_workers=hyper_params['num_workers']
                                         , drop_last=False
                                         )


    optimizer = optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    log_number_train = log_number_val = 0

    summary(model, (1, 10, 100, 100))








    for epoch in tqdm(range(hyper_params["n_epochs"])):

        total_train_loss = 0
        total_train_correct = 0
        incorrect_classifications_train = []
        incorrect_classifications_val = []
        epoch_classifications_train = []
        epoch_classifications_val = []
        max_train_acc = 0
        max_val_acc = 0

        model.train()
        for batch_number, (images, labels, paths) in enumerate(train_loader):

            images = torch.unsqueeze(images, 1).double().cuda()  # added channel dimensions (grayscale)
            labels = labels.long().cuda()

            optimizer.zero_grad() # Whenever pytorch calculates gradients it always adds it to whatever it has, so we need to reset it each batch.
            preds = model(images) # Pass Batch

            loss = criterion(preds, labels) # Calculate Loss
            total_train_loss += loss.item()
            loss.backward() # Calculate Gradients - the gradient is the direction we need to move towards the loss function minimum (LR will tell us how far to step)
            optimizer.step() # Update Weights - the optimizer is able to update the weights because we passed it the weights as an argument in line 4.

            num_correct = get_num_correct(preds, labels)
            total_train_correct += num_correct

            if log_data:
                experiment.log_metric("Train batch accuracy fold %s" % curr_fold, num_correct/len(labels)*100, step = log_number_train)
                experiment.log_metric("Train batch loss fold %s" % curr_fold, loss.item(), step = log_number_train)
            log_number_train += 1

            # print('Train: Batch number:', batch_number, 'Num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct/len(labels)), 'Loss:', loss.item())
            incorrect_classifications_train.append(get_mistakes(preds, labels, paths))
            for prediction in zip(preds, labels, paths):
                epoch_classifications_train.append(prediction)

        epoch_accuracy = calc_accuracy(epoch_classifications_train)
        if epoch_accuracy > max_train_acc:
            max_train_acc = epoch_accuracy

        if log_data:
            experiment.log_metric("Train epoch accuracy fold %s" % curr_fold, epoch_accuracy, step = epoch)
            experiment.log_metric("Train epoch loss fold %s" % curr_fold, total_train_loss, step = epoch)
        print('Train: Epoch:', epoch, 'num correct:', total_train_correct, 'Accuracy:', str(epoch_accuracy) + '%', 'Batch loss:', total_train_loss)


        total_val_loss = 0
        total_val_correct = 0

        model.eval()
        with torch.no_grad():
            for batch_number, (images, labels, paths) in enumerate(val_loader):
                images = torch.unsqueeze(images, 1).double().cuda()  # added channel dimensions (grayscale)
                labels = labels.long().cuda()

                preds = model(images)  # Pass Batch
                loss = criterion(preds, labels)  # Calculate Loss
                total_val_loss += loss.item()

                num_correct = get_num_correct(preds, labels)
                total_val_correct += num_correct

                if log_data:
                    experiment.log_metric("Val batch accuracy fold %s" % curr_fold, num_correct / len(labels) * 100, step=log_number_val)
                    experiment.log_metric("Val batch loss fold %s" % curr_fold, loss.item(), step=log_number_val)
                log_number_val += 1

                # print('Val: Batch number:', batch_number, 'Num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct / len(labels)), 'Loss:', loss.item())
                # print_mistakes(preds, labels, paths)

                incorrect_classifications_val.append(get_mistakes(preds, labels, paths))

                for prediction in zip(preds, labels, paths):
                    epoch_classifications_val.append(prediction)

            epoch_accuracy = calc_accuracy(epoch_classifications_val)
            if epoch_accuracy > max_val_acc:
                max_val_acc = epoch_accuracy

            if log_data:
                experiment.log_metric("Val epoch accuracy fold %s" % curr_fold, epoch_accuracy, step=epoch)
                experiment.log_metric("Val epoch loss fold %s" % curr_fold, total_val_loss, step=epoch)
            print('Val Epoch:', epoch, 'num correct:', total_val_correct, 'Accuracy:', str(epoch_accuracy) + '%' , 'Batch loss:', total_val_loss)

        if epoch >= hyper_params['n_epochs']-1:
            print('TRAIN MISCLASSIFICATIONS:')
            print(incorrect_classifications_train)
            print('TEST MISCLASSIFICATIONS:')
            print(incorrect_classifications_val)

    acc_list_train.append(max_train_acc)
    acc_list_val.append(max_val_acc)
    curr_fold += 1

if log_data:
    experiment.log_metric("average_train_accuracy", np.average(acc_list_train))
    experiment.log_metric("average_val_accuracy", np.average(acc_list_val))

