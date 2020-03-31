import errno
import os.path as osp
import os
import re
import warnings
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from torch import max
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import use
use('agg')
from matplotlib import gridspec
import math



def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.
    Args:
        fpath (str): file path.
    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_frame_sequences(frames_folder_path, class_folders = ['apex', 'papillary', 'mitral']):
    """
    This function accepts the path for the frames folder (which should contain the frames for
    all videos separated into different folders.

    returns a nested dictionary of classes, in each class the keys are video name and value is max number of frames.
    """

    files = {}
    for class_name in class_folders:
        # print(class_name)
        files[class_name] = {}
        file_names = os.listdir(frames_folder_path + "/" + class_name)
        # print(frames_folder_path + "/" + class_name)
        # print(file_names)
        try:
            for file in file_names:
                if file == '.DS_Store': continue
                file_name = re.match(r".+(?=_\d+\.jpg)", file).group()

                frame_number = re.search(r"(?<=_)[\d]+(?=\.jpg)", file).group()
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


def get_mistakes(preds, labels, paths):
    preds_check = preds.argmax(dim=1).eq(labels)
    mistake_paths = [path for (i, path) in enumerate(paths) if preds_check[i] == False]
    predicted_view = [view.item() for (i, view) in enumerate(preds.argmax(dim=1)) if preds_check[i] == False]
    ground_truth_view = [view.item() for (i, view) in enumerate(labels) if preds_check[i] == False]
    mistakes = []
    for (true_v, predicted_v, path) in zip(ground_truth_view, predicted_view, mistake_paths):
        mistakes.append((predicted_v, true_v, path))
    return mistakes


def get_train_val_idx(data_set, random_state, test_size = 0.2):
    # Separate data set into movie names so we can split by movie:
    movie_list = list()
    label_list = list()

    for (path, label) in data_set.samples:
        movie_name = re.search('.+(?=_\d+\.pickle)', path).group()
        if movie_name not in movie_list:
            movie_list.append(movie_name)
            label_list.append(label)

    X_train, X_val, y_train, y_val = train_test_split(movie_list, label_list, stratify=label_list, test_size=test_size,
                                                      random_state=random_state)

    train_idx = list()
    val_idx = list()
    for i, (path, label) in enumerate(data_set.samples):
        movie_name = re.search('.+(?=_\d+\.pickle)', path).group()
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
        movie_name = re.search('.+(?=_\d+\.pickle)', path).group()
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
            movie_name = re.search('.+(?=_\d+\.pickle)', path).group()
            if movie_name in X_train:
                train_idx.append(i)
            elif movie_name in X_val:
                val_idx.append(i)
            else:
                raise NameError("movie not in X_train or in X_val")
        fold_indexes.append((train_idx, val_idx))

    return fold_indexes


def online_mean_and_std(loader):
    """Compute the mean and std and print them out

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
    """
    receives the predictions for a full epoch of videos. Assuming each video has at least 5 mini-clips,
    runs majority vote per video and finally calculates the accuracy.
    tie counts as mistake.
    """
    # prediction_dict is a dictionary that aggregates videos and their predictions.
    predictions_dict = {}
    # preprocessing
    for (pred, true, path) in prediction_list:
        # get index of max prediction
        pred = max(pred, 0)[1].item()
        true = true.item()
        file_name = re.search('.+(?=_\d+\.pickle)', path).group()

        # if the movie is not in prediction_dict add it if it is then add the prediction to it.

        if file_name not in predictions_dict.keys():
            predictions_dict[file_name] = {'pred': [pred], 'true': true}
        else:
            predictions_dict[file_name]['pred'].append(pred)

    num_correct = 0
    num_mistakes = 0
    for video, preds in predictions_dict.items():
        majority_vote = find_majority(preds['pred']) # note that in case of ties, it counts as a mistake
        if majority_vote == preds['true']:
            num_correct += 1
        else:
            num_mistakes += 1

    # print(num_correct, video_count, misclassified_videos)
    video_count = len(predictions_dict.keys())
    return np.round(num_correct/video_count*100, 4)

def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return -1
    return top_two[0][0]

def save_plot_clip_frames(clip, label, path, target_folder ="clip_plots", added_info_to_path = ""):
    movie_name = re.search('.+(?=_\d+\.pickle)', path).group()
    try: # this is in case label is not a tensor and doesn't have ".item()"
        view_class = str(label.item())
    except(AttributeError):
        view_class = str(label)
    n_plots = clip.shape[0]
    cols = 5
    rows = int(math.ceil(n_plots / cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (20, 10))
    fig.suptitle(movie_name + ", label:" + view_class, size=20)
    for n in range(n_plots):
        ax = fig.add_subplot(gs[n])
        ax.set_title("Clip im " + str(n+1))
        ax.imshow(clip[n], cmap="hot")
        #     # imgplot.set_clim(0.0, 0.7)
        #     # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

    mkdir_if_missing(target_folder)
    mkdir_if_missing(os.path.join(target_folder, view_class))
    gs.tight_layout(fig)
    full_path = Path(os.path.join(".", target_folder, view_class, movie_name + path[-9: -7] + added_info_to_path + ".jpg"))
    plt.savefig(full_path, dpi = 300)
    print(full_path)
    if not full_path.exists():
        print("Error - figure not saved!")

def get_files_list(directory):
    """
    returns a list of file names in given directory
    """
    file_list = list()
    with os.scandir(directory) as files:
        for file in files:
            file_list.append(file.name)
    return file_list


def load_and_process_images(folder, frame_list, crop_size = 300, resize_dim = 100, to_numpy = True):
    images = []
    for frame in frame_list:
        img = Image.open(os.path.join(folder,frame))
        # we want to crop a 300x300 square from the center of the image
        width, height = img.size
        left = (width - crop_size)/2
        right = left + crop_size
        top = (height -crop_size)/2
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))
        img = img.resize((resize_dim, resize_dim), 1)
        if to_numpy:
            img = np.array(img)
        images.append(img)
    return np.array(images)

def save_image_sequence(frame_list, source_directory, view, target_directory, video, clip_number):
    # sort frames
    sorted_frame_list = sorted(frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+(?=\.jpg)', x).group()))
    # process images and convert to numpy array
    image_array = load_and_process_images(source_directory[view], sorted_frame_list, to_numpy=True)
    # save image set as pickle
    with open(os.path.join(target_directory[view], video + '_' + str(clip_number) + '.pickle'), 'wb') as file:
        pickle.dump(image_array, file, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_object(obj, path = "pickle_object"):
    with open(os.path.join(path + '.pickle'), 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Object pickled succesfuly.")

def open_pickled_object(path):
    with open(path, 'rb') as handle:
        file = pickle.load(handle)
    return file