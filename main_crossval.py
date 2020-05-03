from comet_ml import Experiment
import os
from tqdm import tqdm
from run_state import Run
from torchutils import *

print('CUDA available:', torch.cuda.is_available())
print('CUDA enabled:', torch.backends.cudnn.enabled)
# torch.cuda.empty_cache()
torch.cuda.get_device_capability(0)

save_experiment = True

HP1 = {"learning_rate": 0.001
      ,"lr_scheduler": {'step_size': 5, 'gamma': 0.8}
      ,"n_epochs": 80
      ,"batch_size": 64
      ,"num_workers": 5
      ,"normalized_data": True
      ,"stratified": False
      ,"horizontal_flip": False
      ,"max_frames": 10
      ,"random_seed": 43
      ,"flip_prob": 0.5
      ,"dataset": "10frame_5steps_100px"
      ,"classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
      ,"model_type": "3dCNN"
      ,"resolution": 100
      ,"adaptive_pool": [7,5,5]
      ,"features": [16,16,"M",32,32,"M",32,32,"M",64,64,64,"M"]
      ,"classifier": [0.5,200,0.5,150,0.4,100]
                }


run = Run(disable_experiment = not save_experiment,
      machine = 'server',
      hyper_params= hyper_params)

for epoch in tqdm(range(hyper_params["n_epochs"])):

    train(epoch, run)
    evaluate(epoch, run)





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

