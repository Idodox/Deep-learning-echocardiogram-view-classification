import numpy as np
import os
from PIL import Image
import pickle
from utils import get_frame_sequences
from tqdm import tqdm
import re

############### CONFIG ########################

# view_list = ['apex', 'mitral', 'papillary']
view_list = ['papillary']

source_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/papillary'}

target_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/5frame_steps/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/5frame_steps/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/5frame_steps/papillary'}

#################################################

video_list = {}
file_names = get_frame_sequences('/Users/idofarhi/Documents/Thesis/Data/frames')
for view in view_list:
    video_list[view] = list(file_names[view].keys())


def get_files_list(directory):
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


for view in view_list:
    file_list = get_files_list(source_directory[view])
    print('Running view:', view)
    for video in tqdm(video_list[view]):
        # get frame list
        for i in range(5): # to get frames 0,5,10 etc then 1, 6, 11 etc ...
            frame_list = []
            for file in file_list:
                file_name = re.match(r".+?(?=_)", file).group()
                if file_name == video: # TODO: need to fix to get file name regardless of size
                    if int(re.search(r'(?<=_)[\d]+', file).group() ) % 5 == i:
                        frame_list.append(file)
            # skip any less than 10 frames
            if len(frame_list) < 10:
                continue

            frame_list = sorted(frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+', file).group() ))
            # load and process images from list
            image_array = load_and_process_images(source_directory[view], frame_list, to_numpy = True)
            # output is a numpy array of frames


            # save image set as pickle
            with open(os.path.join(target_directory[view], video + '_' + str(i) + '.pickle'), 'wb') as file:
                pickle.dump(image_array, file, protocol=pickle.HIGHEST_PROTOCOL)
