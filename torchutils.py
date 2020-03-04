import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets
import os
import re
import pickle
from tqdm import tqdm
from torch import max
from sklearn.model_selection import KFold
import torch
import shutil
import random

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
        inplace(bool,optional): Bool to make this operation in-place.

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


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_frame_sequences(frames_folder_path):
    """
    This function accepts the path for the frames folder (which should contain the frames for
    all videos separated into different folders.

    returns a nested dictionary of classes, in each class the keys are video name and value is max number of frames.
    """

    class_folders = ['apex', 'papillary', 'mitral']

    files = {}
    for class_name in class_folders:
        # print(class_name)
        files[class_name] = {}
        file_names = os.listdir(frames_folder_path + "/" + class_name)
        # print(frames_folder_path + "/" + class_name)
        # print(file_names)
        try:
            for file in file_names:
                file_name = re.match(".+?(?=_)", file).group()

                frame_number = re.search("(?<=_)[\d]+", file).group()
                assert(frame_number.isdigit())
                frame_number = int(frame_number)

                # see if the current frame number is greater than the one in the dictionary.
                # If it is, replace it.
                # If the file name doesn't exist in the dict, add it.
                if file_name in files[class_name]:
                    if frame_number > files[class_name][file_name]:
                        files[class_name][file_name] = frame_number
                else:
                    files[class_name][file_name] = frame_number

        except():
            print("get_frame_sequences function error")
            return

    return files


def pickle_loader(path, min_frames = None):
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
        return file[:min_frames]
    return file
    # except:
    #     print('Loading pickle file failed. path:', path)


def get_mistakes(preds, labels, paths):
    preds_check = preds.argmax(dim=1).eq(labels)
    mistake_paths = [path for (i, path) in enumerate(paths) if preds_check[i] == False]
    predicted_view = [view.item() for (i, view) in enumerate(preds.argmax(dim=1)) if preds_check[i] == False]
    ground_truth_view = [view.item() for (i, view) in enumerate(labels) if preds_check[i] == False]
    mistakes = []
    for (true_v, predicted_v, path) in zip(ground_truth_view, predicted_view, mistake_paths):
        mistakes.append((predicted_v, true_v, path))
    return mistakes


def get_train_val_idx(data_set, random_state):
    # Separate data set into movie names so we can split by movie:
    movie_list = list()
    label_list = list()

    for (path, label) in data_set.samples:
        movie_name = re.search('[ \w-]+?(?=_\d)', path).group()
        if movie_name not in movie_list:
            movie_list.append(movie_name)
            label_list.append(label)

    X_train, X_val, y_train, y_val = train_test_split(movie_list, label_list, stratify=label_list, test_size=0.2,
                                                      random_state=random_state)

    train_idx = list()
    val_idx = list()
    for i, (path, label) in enumerate(data_set.samples):
        movie_name = re.search('[ \w-]+?(?=_\d)', path).group()
        if movie_name in X_train:
            train_idx.append(i)
        elif movie_name in X_val:
            val_idx.append(i)
        else:
            raise NameError("movie not in X_train or in X_val")

    # Make sure train and val indexes aren't mixed up
    for index in train_idx:
        assert(index not in val_idx)

    return train_idx, val_idx


def get_cross_val_idx(data_set, random_state, n_splits = 5):
    # TODO: add assertion to make sure indexes in train are not in val.

    movie_list = np.array([])
    label_list = np.array([])

    for (path, label) in data_set.samples:
        movie_name = re.search('[ \w-]+?(?=_\d)', path).group()
        if movie_name not in list(movie_list):
            movie_list = np.append(movie_list, movie_name)
            label_list = np.append(label_list, label)

    fold_indexes = []

    cv = KFold(n_splits=5, random_state=random_state, shuffle=True)
    for train_index, val_index in cv.split(movie_list):


        X_train, X_val = movie_list[train_index], movie_list[val_index]
        y_train, y_val = label_list[train_index], label_list[val_index]

        train_idx = list()
        val_idx = list()

        for i, (path, label) in enumerate(data_set.samples):
            movie_name = re.search('[ \w-]+?(?=_\d)', path).group()
            if movie_name in X_train:
                train_idx.append(i)
            elif movie_name in X_val:
                val_idx.append(i)
            else:
                raise NameError("movie not in X_train or in X_val")
        fold_indexes.append((train_idx, val_idx))

    return fold_indexes


def online_mean_and_std(loader):
    """Compute the mean and std in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    pixel_mean = np.zeros(10)
    pixel_std = np.zeros(10)
    k = 1
    for image, _, __ in tqdm(loader):
        image = np.array(image)
        pixels = image.reshape((-1, image.shape[1]))

        for pixel in pixels:
            diff = pixel - pixel_mean
            pixel_mean += diff / k
            pixel_std += diff * (pixel - pixel_mean)
            k += 1

    pixel_std = np.sqrt(pixel_std / (k - 2))
    print(pixel_mean)
    print(pixel_std)


def calc_accuracy(prediction_list):
    predictions_dict = {}
    # preprocessing
    for (pred, true, path) in prediction_list:
        # get index of max prediction
        pred = max(pred, 0)[1].item()
        true = true.item()
        file_name = re.search('[ \w-]+?(?=_\d)', path).group()

        if file_name not in predictions_dict.keys():
            predictions_dict[file_name] = [(pred, true)]
        else:
            predictions_dict[file_name].append((pred, true))

    num_correct = 0
    video_count = len(predictions_dict.keys())
    misclassified_videos = []
    for video, pred_list in predictions_dict.items():
        count_t = 0
        count_c = 0
        for (pred, true) in pred_list:
            count_t += 1
            if pred == true:
                count_c += 1
        if count_c > np.floor(count_t/2):
            num_correct += 1
        else:
            misclassified_videos.append(video)

    # print(num_correct, video_count, misclassified_videos)

    return np.round(num_correct/video_count*100, 23)


def save_checkpoint(state, is_best, filename='checkpoint.pt.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt.tar')