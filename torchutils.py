from torchvision import datasets
import pickle
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


def save_checkpoint(state, is_best, filename='checkpoint.pt.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt.tar')