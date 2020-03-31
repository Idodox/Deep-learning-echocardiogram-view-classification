import os
import pickle
import re

from tqdm import tqdm

from utils import get_frame_sequences, mkdir_if_missing, get_files_list, load_and_process_images

############### CONFIG ########################

# view_list = ['apex', 'mitral', 'papillary']
# view_list = ['papillary']
view_list = ['2CH', '3CH', '4CH']
step_size = 5
min_frames = 10
final_dim = 100

# source_directory = {}

source_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/color/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/color/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/color/papillary',
                    '2CH': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/2CH',
                    '3CH': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/3CH',
                    '4CH': '/Users/idofarhi/Documents/Thesis/Data/frames/raw/4CH'}

target_directory = {'apex': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps' + str(min_frames) + '/apex',
                    'mitral': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps' + str(min_frames) + '/mitral',
                    'papillary': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps' + str(min_frames) + '/papillary',
                    '2CH': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps' + str(min_frames) + '/2CH',
                    '3CH': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps' + str(min_frames) + '/3CH',
                    '4CH': '/Users/idofarhi/Documents/Thesis/Data/frames/' + str(step_size) + 'frame_steps' + str(min_frames) + '/4CH'
                    }

#################################################

for folder in target_directory.values():
    mkdir_if_missing(folder)

video_list = {}
file_names = get_frame_sequences('/Users/idofarhi/Documents/Thesis/Data/frames/raw', class_folders=['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH'])
for view in view_list:
    video_list[view] = list(file_names[view].keys())

# for each view...
for view in view_list:
    # get a list of all files in the relevant view directory (file list)
    file_list = get_files_list(source_directory[view])
    print('Running view:', view)

    # run through each video...
    for video in tqdm(video_list[view]):

        # create a temporary list with only the relevant video frame names
        video_frame_list = []
        for file in file_list:
            if file == '.DS_Store': continue
            file_name = re.match(r".+(?=_\d+\.jpg)", file).group()
            if file_name == video:
                video_frame_list.append(file)
        video_frame_list = sorted(video_frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+', x).group()))

        # get video number of frames and make sure we can make
        # the needed number of sequences with at least min_frames each, else skip
        if len(video_frame_list) < min_frames * step_size:
            continue

        for i in range(step_size):  # if this is 5: to get frames 0,5,10 etc then 1, 6, 11 etc ...
            frame_list = []
            for file in video_frame_list:
                file_name = re.match(r".+(?=_\d+\.jpg)", file).group()
                if file_name == video:
                    if int(re.search(r'(?<=_)[\d]+(?=\.jpg)', file).group()) % step_size == i:
                        frame_list.append(file)
            assert(len(frame_list) >= min_frames) # just to make sure we don't have a bug.

            frame_list = sorted(frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+(?=\.jpg)', file).group()))
            # load and process images from list
            image_array = load_and_process_images(source_directory[view], frame_list, to_numpy=True, resize_dim = final_dim)
            # output is a numpy array of frames

            # save image set as pickle
            with open(os.path.join(target_directory[view], video + '_' + str(i) + '.pickle'), 'wb') as file:
                pickle.dump(image_array, file, protocol=pickle.HIGHEST_PROTOCOL)
