import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import pickle
import re
from torchutils import Normalize, ToTensor, UnNormalize
from torchvision import transforms
import torch
from utils import get_class_number


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    np_arr = np_arr[0][0]
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255)
    return np_arr.astype(np.uint8)


def save_image(im, path):
    """
        Saves a numpy matrix as an image
    Args:
        im (Numpy array): Matrix of shape -
        path (str): Path to the image
    """

    # volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(im[0]), vmin=0, vmax=0.8)
    if isinstance(im, (np.ndarray, np.generic)):
        # im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)
    # mlab.draw()
    # mlab.savefig(path)


def preprocess_clip(input_clip):
    """
        Processes clip for CNNs
    Args:
        clip (numpy array): Clip to process
    returns:
        clip (torch variable): Variable that contains processed float tensor
        the returned clip is in the range -1 to 1, after normalization
    """
    transformer = transforms.Compose([
        ToTensor()
        ,Normalize(0.213303, 0.21379)
    ])
    output_clip = transformer(input_clip)
    # Add one more channel to the beginning. This is like number of channels.
    output_clip.unsqueeze_(0)
    # Add one more channel to the beginning. This is like batch size.
    output_clip.unsqueeze_(0)
    # Move to the GPU if available
    if torch.cuda.is_available():
        output_clip = output_clip.cuda(0)

    return output_clip


def get_clip_to_run(clip_path):
    """
        Gets clip for visualizations
    Args:
        path: full path to clip
    returns:
        original_clip (numpy arr): Original clip read from the file
        prep_clip (numpy_arr): Processed clip
        target_class (str): Target class for the clip
        pretrained_model(Pytorch model): Model to use for the operations
    """
    with open(clip_path, 'rb') as handle:
        original_clip = pickle.load(handle)/255

    movie_name = re.search('.+(?=_\d+\.pickle)', clip_path).group()
    target_class_name = re.search(r'(\/)(2CH|3CH|4CH|apex|mitral|papillary)(\/)', clip_path).group()[1:-1]
    # Process clip
    prep_clip = preprocess_clip(original_clip)
    target_class_number = get_class_number(target_class_name)

    return (original_clip,
            prep_clip,
            movie_name,
            target_class_name,
            target_class_number)
