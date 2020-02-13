import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Sampler
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import re
import pickle


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def plot_batch(batch, ncolumns = 10, show_labels = False):
    images, labels = batch

    # grid row argument says rows but varies columns.
    grid = torchvision.utils.make_grid(images, nrow = ncolumns)

    plt.figure(figsize=(15, 7))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    if show_labels:
        print('labels:', labels)


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


def get_frames(folder, file_name, frame_numbers, frames_folder_path = str(Path.home()) + "/Documents/Thesis/Data/frames"):
    """
    :param folder: (str) class name
    :param file_name: name of video we want the frames for
    :param frame_numbers: frame numbers
    :param frames_folder_path: path for frames folder
    :return: array of requested frames
    """
    pass


# files = get_frame_sequences(str(Path.home()) + "/Documents/Thesis/Data/frames")

def pickle_loader(path, max_frames = None):
    """
    :param path: path to pickle file
    :return: opens the file and returns the un-pickled file
    """
    try:
        with open(path, 'rb') as handle:
            file = pickle.load(handle)
        if max_frames is not None:
            assert len(file) >= max_frames # Assert file has at least (max_frames) number of frames.
            return file[:max_frames]
        return file
    except:
        print('Loading pickle file failed. path:', path)


def get_train_val_idx(data_set):
    # Separate data set into movie names so we can split by movie:
    movie_list = list()
    label_list = list()

    for (path, label) in data_set.samples:
        movie_name = re.search('[ \w-]+?(?=_\d)', path).group()
        if movie_name not in movie_list:
            movie_list.append(movie_name)
            label_list.append(label)

    X_train, X_val, y_train, y_val = train_test_split(movie_list, label_list, stratify=label_list, test_size=0.2,
                                                      random_state=42)

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

    return train_idx, val_idx