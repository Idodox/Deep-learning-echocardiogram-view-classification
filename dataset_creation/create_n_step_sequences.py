import os
import pickle
import re

from tqdm import tqdm

from utils import get_frame_sequences, mkdir_if_missing, get_files_list, load_and_process_images

############### CONFIG ########################

view_list = ['apex', 'mitral', 'papillary']
# view_list = ['papillary']
step_size = 5

source_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/papillary'}

target_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(
                        step_size) + 'frame_steps/papillary'}

#################################################

for folder in target_directory.values():
    mkdir_if_missing(folder)

video_list = {}
file_names = get_frame_sequences('/Users/idofarhi/Documents/Thesis/Data/frames')
for view in view_list:
    video_list[view] = list(file_names[view].keys())

for view in view_list:
    file_list = get_files_list(source_directory[view])
    print('Running view:', view)
    for video in tqdm(video_list[view]):
        # get frame list
        for i in range(step_size):  # if this is 5: to get frames 0,5,10 etc then 1, 6, 11 etc ...
            frame_list = []
            for file in file_list:
                file_name = re.match(r".+?(?=_)", file).group()
                if file_name == video:  # TODO: need to fix to get file name regardless of size
                    if int(re.search(r'(?<=_)[\d]+', file).group()) % step_size == i:
                        frame_list.append(file)
            # skip any less than 10 frames
            if len(frame_list) < 10:
                continue

            frame_list = sorted(frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+', file).group()))
            # load and process images from list
            image_array = load_and_process_images(source_directory[view], frame_list, to_numpy=True)
            # output is a numpy array of frames

        # save image set as pickle
        with open(os.path.join(target_directory[view], video + '_' + str(i) + '.pickle'), 'wb') as file:
            pickle.dump(image_array, file, protocol=pickle.HIGHEST_PROTOCOL)
