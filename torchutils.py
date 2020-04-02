from torch import nn as nn
from torchvision import datasets
import pickle
import torch
import shutil
import random
from comet_ml import Experiment
from modular_cnn import ModularCNN, make_layers
from utils import get_num_correct, get_mistakes, calc_accuracy
import numpy as np
from resnext_util import generate_resnext_model


class DatasetFolderWithPaths(datasets.DatasetFolder):
    """Custom dataset that includes file paths. Extends
    torchvision.datasets.DatasetFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what DatasetFolder normally returns
        original_tuple = super(DatasetFolderWithPaths, self).__getitem__(index)
        # data file path
        path = self.samples[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    """

    def __call__(self, arr):
        """
        Args:
            numpy array to be converted to tensor.

        Returns:
            Tensor: Converted array.
        """
        return torch.from_numpy(arr)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        new_tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        for i, image in enumerate(tensor):
            new_tensor[i] = image.sub_(mean).div_(std)
        return new_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomZoom(object):
   #TODO: Make this work for our case
    def __init__(self,
                 zoom_range,
                 fill_mode='constant',
                 fill_value=0,
                 target_fill_mode='nearest',
                 target_fill_value=0.,
                 lazy=False):
        """Randomly zoom in and/or out on an image
        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom.
            Anything less than 1.0 will zoom in on the image,
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in,
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(),
                zoom_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), zoom_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed

class RandomHorizontalFlip(object):
    """Horizontally flip every frame in the set pending probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (sequence of grayscale images): clip to be flipped.

        Returns:
            Randomly flipped image.
        """
        if random.random() < self.p:
            for i, image in enumerate(clip):
                clip[i] = torch.flip(image, [1])

        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def pickle_loader(path, min_frames = None, shuffle_frames = False):
    """
    :param path: path to pickle file
    :return: opens the file and returns the un-pickled file
    """
    # try:
    with open(path, 'rb') as handle:
        file = pickle.load(handle)
    file = file / 255
    if min_frames is not None:
        assert len(file) >= min_frames # Assert file has at least (max_frames) number of frames.
        file = file[:min_frames]
    # this option was added to ascertain weather the net uses frame order to determine classification.
    if shuffle_frames:
        np.random.shuffle(file)
    return file
    # except:
    #     print('Loading pickle file failed. path:', path)


def save_checkpoint(state, is_best, filename='checkpoint.pt.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt.tar')



def train(epoch, run):
    total_train_loss = 0
    total_train_correct = 0
    incorrect_classifications_train = []
    epoch_classifications_train = []
    run.model.train()
    for batch_number, (images, labels, paths) in enumerate(run.train_loader):

        # for i, (image, label, path) in enumerate(zip(images, labels, paths)):
        #     save_plot_clip_frames(image, label, path, added_info_to_path = epoch)

        if run.grayscale:
            images = torch.unsqueeze(images, 1).double()  # added channel dimensions (grayscale)
        else:
            images = images.float().permute(0, 4, 1, 2, 3).float()
        labels = labels.long()

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        run.optimizer.zero_grad()  # Whenever pytorch calculates gradients it always adds it to whatever it has, so we need to reset it each batch.
        preds = run.model(images)  # Pass Batch

        loss = run.criterion(preds, labels)  # Calculate Loss
        total_train_loss += loss.item()
        loss.backward()  # Calculate Gradients - the gradient is the direction we need to move towards the loss function minimum (LR will tell us how far to step)
        run.optimizer.step()  # Update Weights - the optimizer is able to update the weights because we passed it the weights as an argument in line 4.

        num_correct = get_num_correct(preds, labels)
        total_train_correct += num_correct

        run.experiment.log_metric("Train batch accuracy", num_correct / len(labels) * 100, step=run.log_number_train)
        run.experiment.log_metric("Avg train batch loss", loss.item(), step=run.log_number_train)
        run.log_number_train += 1

        # print('Train: Batch number:', batch_number, 'Num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct/len(labels)), 'Loss:', loss.item())
        incorrect_classifications_train.append(get_mistakes(preds, labels, paths))
        for prediction in zip(preds, labels, paths):
            epoch_classifications_train.append(prediction)
    epoch_accuracy = calc_accuracy(epoch_classifications_train)

    run.experiment.log_metric("Train epoch accuracy", epoch_accuracy, step=epoch)
    run.experiment.log_metric("Avg train epoch loss", total_train_loss / batch_number, step=epoch)
    print('\nTrain: Epoch:', epoch, 'num correct:', total_train_correct, 'Accuracy:', str(epoch_accuracy) + '%')


def evaluate(epoch, run):
    incorrect_classifications_val = []
    total_val_loss = 0
    total_val_correct = 0
    best_val_acc = 0
    epoch_classifications_val = []
    run.model.eval()
    with torch.no_grad():
        for batch_number, (images, labels, paths) in enumerate(run.val_loader):

            if run.grayscale:
                images = torch.unsqueeze(images, 1).double()  # added channel dimensions (grayscale)
            else:
                images = images.float().permute(0, 4, 1, 2, 3).float()
            labels = labels.long()

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            preds = run.model(images)  # Pass Batch
            loss = run.criterion(preds, labels)  # Calculate Loss
            total_val_loss += loss.item()

            num_correct = get_num_correct(preds, labels)
            total_val_correct += num_correct

            run.experiment.log_metric("Val batch accuracy", num_correct / len(labels) * 100, step=run.log_number_val)
            run.experiment.log_metric("Avg val batch loss", loss.item(), step=run.log_number_val)
            run.log_number_val += 1

            # print('Val: Batch number:', batch_number, 'Num correct:', num_correct, 'Accuracy:', "{:.2%}".format(num_correct / len(labels)), 'Loss:', loss.item())
            # print_mistakes(preds, labels, paths)

            incorrect_classifications_val.append(get_mistakes(preds, labels, paths))

            for prediction in zip(preds, labels, paths):
                epoch_classifications_val.append(prediction)

        epoch_accuracy = calc_accuracy(epoch_classifications_val)

        run.experiment.log_metric("Val epoch accuracy", epoch_accuracy, step=epoch)
        run.experiment.log_metric("Avg val epoch loss", total_val_loss / batch_number, step=epoch)
        print('Val Epoch:', epoch, 'num correct:', total_val_correct, 'Accuracy:', str(epoch_accuracy) + '%')

    # if epoch >= hyper_params['n_epochs'] - 1:
    #     print('TRAIN MISCLASSIFICATIONS:')
    #     print(incorrect_classifications_train)
    #     print('TEST MISCLASSIFICATIONS:')
    #     print(incorrect_classifications_val)
    is_best = epoch_accuracy > run.best_val_acc
    if is_best:
        print("Best run so far! updating params...")
        run.best_val_acc = epoch_accuracy
        run.best_model_preds = epoch_classifications_val
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': run.model.state_dict(),
        'best_acc1': run.best_val_acc,
        'optimizer': run.optimizer.state_dict(),
    }, is_best)


def get_resnext(hyper_params):
    model = generate_resnext_model('score') # in score, last_ft=True, in feature, last_fc=False
    model = nn.DataParallel(model).cuda(0)
    print('loading resnext model')
    model_data = torch.load('resnext-101-64f-kinetics.pth')
    model.load_state_dict(model_data['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_classes = len(hyper_params['classes'])
    model.module.fc = nn.Linear(2048, num_classes)
    model.to(torch.device('cuda:0'))
    return model


def get_modular_3dCNN(hyper_params):
    num_classes = len(hyper_params['classes'])
    model = ModularCNN(make_layers(hyper_params["features"], batch_norm=True), classifier = hyper_params["classifier"], adaptive_pool=hyper_params["adaptive_pool"], num_classes = num_classes)
    if torch.cuda.is_available():
        model = model.cuda(0)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda(0)
    model.to(torch.device('cuda:0'))
    return model