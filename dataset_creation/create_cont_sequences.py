import os
import pickle
import re

from tqdm import tqdm

from utils import get_frame_sequences, mkdir_if_missing, get_files_list, load_and_process_images, save_image_sequence

############### CONFIG ########################

view_list = ['apex', 'mitral', 'papillary']
# view_list = ['papillary']
step_size = 5
num_frames = 10

source_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/papillary'}

target_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/cont' + str(step_size) + 'frame_steps_'+ str(num_frames) + '/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/cont' + str(step_size) + 'frame_steps_'+ str(num_frames) + '/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/cont' + str(step_size) + 'frame_steps_'+ str(num_frames) + '/papillary'}

#################################################

for folder in target_directory.values():
    mkdir_if_missing(folder)

# get list of movie files for each view
video_list = {}
file_names = get_frame_sequences(source_directory['apex'][:-5])
for view in view_list:
    video_list[view] = list(file_names[view].keys())

"""
(1) run through each view...
(2) get a list of all files in the relevant view directory
(3) run through each video...
(4) create a temporary list with only the relevant video frame names
(5) get video number of frames (max)
(6) make sure we can create at least 5 mini-clips from the video or skip it
    e.g. if we want mini clips of 10 frames every 3 frames then minimum is 10*3+5 = 35 frames
(7) calculate the number of mini-clips we can get from it 
    e.g. if there are 50 frames and we want mini clips of 10 frames in spans of 3 frames:
    50 - (10-1)*3 = 23
(8) run through each mini clip number up to the max
(9) append relevant frames to frame_list (%step_size) up to desired number of frames (num_frames)
"""

# (1) for each view...
for view in view_list:
    # (2) get a list of all files in the relevant view directory
    file_list = get_files_list(source_directory[view])
    print('Running view:', view)

    # (3) run through each video...
    for video in tqdm(video_list[view]):
        # (4) create a temporary list with only the relevant video frame names
        video_frame_list = []
        for file in file_list:
            file_name = re.match(r".+?(?=_)", file).group()
            if file_name == video:
                video_frame_list.append(file)
        video_frame_list = sorted(video_frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+', x).group()))

        # (5) get video number of frames (max)
        max_frame_num = file_names[view][video] + 1
        # (6) make sure we can create at least 5 mini-clips from the video or skip it
        if 5 > max_frame_num - step_size * num_frames:
            continue
        # (7) calculate the number of mini-clips we can get from it
        n_mini_clips = max_frame_num - (step_size - 1) * num_frames
        # (8) run through each mini clip number up to the max
        for clip_num in range(n_mini_clips):
            # (9) append relevant frames to frame_list (%step_size) up to desired number of frames (num_frames)
            frame_list = []
            for file in video_frame_list[clip_num:]:
                if (int(re.search(r'(?<=_)[\d]+', file).group()) - clip_num) % step_size == 0:
                    frame_list.append(file)
                if len(frame_list) == num_frames:
                    save_image_sequence(frame_list, source_directory, view, target_directory, video, clip_num)