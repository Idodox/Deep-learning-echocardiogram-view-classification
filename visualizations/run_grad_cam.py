import os
import cv2
import torch
import numpy as np
import shutil

from vis_utils import get_clip_to_run, save_image, recreate_clip
from torchutils import create_model, load_checkpoint
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.jet()

from grad_cam_video import GradCamVideo

hyper_params = {"max_frames": 10
        , "random_seed": 999
        , "classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
        , "model_type": "3dCNN"
        , "resolution": 100
        , "adaptive_pool": (7, 5, 5)
        , "features": [16, 16, "M", 16, 16, "M", 32, 32, "M"]
        ,"classifier": [0.5, 200, 0.5, 150, 0.4, 100]
     }

# clip_path = '/Users/idofarhi/Documents/Thesis/Data/frames/5frame_steps10/2CH/AA-055KAP_2CH_0.pickle'
clip_path = '/Users/idofarhi/Documents/Thesis/Data/frames/5frame_steps10/papillary/F4BHJOAE_3.pickle'
# clip_path = '/home/ido/data/5frame_steps10/2CH/AA-055KAP_2CH_0.pickle'
checkpoint_path = '/Users/idofarhi/Documents/Thesis/Code/model_best.pt.tar'
# checkpoint_path = '/home/ido/PycharmProjects/us-view-classification/model_best.pt.tar'


model = create_model(hyper_params)
model.eval()

pretrained_model = load_checkpoint(model, checkpoint_path)

original_clip, prep_clip, movie_name, target_class_name, target_class_number = get_clip_to_run(clip_path)
print("loaded sample '{}'".format(clip_path))
print("label:", target_class_name)
print("label number:", target_class_number)

# Can choose the class for which we get CAM by changing "target_index"
# If None, use class predicted by model
target_index = None

grad_cam = GradCamVideo(model=model, target_layer_names=["20"], use_cuda=False, input_spatial_size=hyper_params['resolution'])

mask, output = grad_cam(prep_clip, target_index)

preds = model(prep_clip).squeeze(0) # squeeze to remove batch dimension
softmax_preds = torch.nn.functional.softmax(preds, dim=0)

# compute top5 predictions
pred_prob, pred_top5 = softmax_preds.data.topk(5)
pred_prob = pred_prob.numpy()
pred_top5 = pred_top5.numpy()


# WRITING IMAGES TO DISK

output_images_folder_cam_combined = os.path.join("cam_saved_images", "combined")
output_images_folder_original = os.path.join("cam_saved_images", "original")
output_images_folder_cam = os.path.join("cam_saved_images", "cam")

try:
    shutil.rmtree(output_images_folder_cam_combined)
except FileNotFoundError:
    pass
os.makedirs(output_images_folder_cam_combined, exist_ok=True)
os.makedirs(output_images_folder_cam, exist_ok=True)
os.makedirs(output_images_folder_original, exist_ok=True)

RESIZE_SIZE = 200
RESIZE_FLAG = 0
SAVE_INDIVIDUALS = 0

for i in range(mask.shape[0]):
    input_data_img = original_clip[i, :, :]
    input_data_img = np.expand_dims(input_data_img, axis = 2) # add grayscale color channel
    input_data_img = input_data_img.astype('float32') # cv2 only works with 32 bit floating point
    input_data_img = cv2.cvtColor(input_data_img,cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask[i]), cv2.COLORMAP_JET)
    if RESIZE_FLAG:
        input_data_img = cv2.resize(input_data_img, (RESIZE_SIZE, RESIZE_SIZE))
        heatmap = cv2.resize(heatmap, (RESIZE_SIZE, RESIZE_SIZE))
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(input_data_img)
    cam = cam / np.max(cam)
    combined_img = np.concatenate((np.uint8(255 * input_data_img), np.uint8(255 * cam)), axis=1)

    # these always need to be saved so we can make the gif
    save_image(combined_img, os.path.join(output_images_folder_cam_combined, "img%02d.jpg" % (i + 1)))

    if SAVE_INDIVIDUALS:
        save_image(np.uint8(255 * cam), os.path.join(output_images_folder_cam, "img%02d.jpg" % (i + 1)))
        save_image(np.uint8(255 * input_data_img), os.path.join(output_images_folder_original, "img%02d.jpg" % (i + 1)))

path_to_combined_gif = os.path.join(output_images_folder_cam_combined, "mygif.gif")
os.system("convert -delay 20 -loop 0 {}.jpg {}".format(os.path.join(output_images_folder_cam_combined, "*"), path_to_combined_gif))
