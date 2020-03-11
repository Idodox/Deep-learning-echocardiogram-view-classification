# Verify that the flip transform works as expected

import torch
from utils import open_pickled_object, save_plot_clip_frames

path = str('/Users/idofarhi/Documents/Thesis/Data/frames/5frame_steps/apex/E1UB4SG2_0.pickle')

clip = torch.from_numpy(open_pickled_object(path))[:10]
print(clip.shape)
save_plot_clip_frames(clip, 0, path, target_folder ="/Users/idofarhi/Documents/Thesis/code/clip_plots")

for i, image in enumerate(clip):
    clip[i] = torch.flip(image, [1])
print(clip.shape)
save_plot_clip_frames(clip, 0, path, target_folder ="/Users/idofarhi/Documents/Thesis/code/clip_plots", added_info_to_path="_flipped")