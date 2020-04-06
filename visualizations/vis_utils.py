import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import pickle
import re
from torchutils import Normalize, ToTensor
from torchvision import transforms
from mayavi import mlab
import torch
from torch.autograd import Variable
from torchvision import models


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


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
        im = format_np_output(im)
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
    """
    transformer = transforms.Compose([
        ToTensor()
        ,Normalize(0.213303, 0.21379)
        # ,RandomHorizontalFlip(self.hyper_params["flip_prob"])
    ])
    output_clip = transformer(input_clip)
    # Add one more channel to the beginning. This is like batch size.
    output_clip.unsqueeze_(0)

    return output_clip


def recreate_image(clip):
    """
        Recreates clip from a torch variable, sort of reverse preprocessing
    Args:
        clip (torch variable): clip to recreate
    returns:
        recreated_clip (numpy arr): Recreated clip in array
    """
    reverse_mean = -1*np.array([0.21308169, 0.2133352,  0.21363254, 0.21361411, 0.21371935, 0.21371628, 0.21360689, 0.21336799, 0.21276978, 0.2121863])
    reverse_std = 1/np.array([0.21369012, 0.21371147, 0.21385933, 0.2138414,  0.21385933, 0.21388142, 0.21384666, 0.21386346, 0.2137403,  0.21360974])
    recreated_clip = copy.copy(clip.data.numpy()[0])
    for image, _ in enumerate(clip):
        recreated_clip[image] /= reverse_std[image]
        recreated_clip[image] -= reverse_mean[image]
    recreated_clip[recreated_clip > 1] = 1
    recreated_clip[recreated_clip < 0] = 0
    recreated_clip = np.round(recreated_clip * 255)

    # recreated_clip = np.uint8(recreated_clip).transpose(1, 2, 0)
    return recreated_clip


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


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
        original_clip = pickle.load(handle)

    movie_name = re.search('.+(?=_\d+\.pickle)', clip_path).group()
    target_class = re.search(r'(\/)(2CH|3CH|4CH|apex|mitral)(\/)', clip_path).group()[1:-1]
    # Process clip
    prep_clip = preprocess_clip(original_clip)
    return (original_clip,
            prep_clip,
            movie_name,
            target_class)
