import pickle
import re
import shutil

import torch

from torchutils import create_model, load_model
from utils import get_class_name
from utils import mkdir_if_missing, load_and_process_images, load_file, write_frames
from vis_utils import prep_pickle_for_inference


def inference(dicom_file_path, hyper_params, model_weights_path):

    ############### CONFIG ########################

    pickle_target_path = dicom_file_path.parent.parent.joinpath('pickle/')
    frames_dir = dicom_file_path.parent.parent.joinpath('temp')

    # DO NOT TOUCH
    step_size = 5
    min_frames = 10
    final_dim = 100

    #################################################


    assert(dicom_file_path.is_file() == True)
    assert(str(dicom_file_path)[-4:] == '.dcm')

    # extract frames to temp directory
    mkdir_if_missing(frames_dir)
    mkdir_if_missing(pickle_target_path)
    for child in dicom_file_path.parent.iterdir():
        if str(child)[-9:] == '.DS_Store': continue
        video_name = str(child.name)
    video = load_file(dicom_file_path)
    write_frames(video, video_name, frames_dir, convert_to_grayscale=True)


    # create a temporary list with only the relevant video frame names
    video_frame_list = []
    for file in frames_dir.iterdir():
        file = str(file.name)
        if file[-9:] == '.DS_Store': continue
        file_name = re.match(r".+(?=_\d+\.jpg)", file).group()
        if file_name == video_name[:-4]:
            video_frame_list.append(file)
    video_frame_list = sorted(video_frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+', x).group()))
    # get video number of frames and make sure we can make
    # the needed number of sequences with at least min_frames each
    if len(video_frame_list) < min_frames * step_size:
        print("File {} doesn't have minimum number of frames required.".format(file_name))
        raise ValueError

    for i in range(step_size):  # if this is 5: to get frames 0,5,10 etc then 1, 6, 11 etc ...
        frame_list = []
        for file in video_frame_list:
            file_name = re.match(r".+(?=_\d+\.jpg)", file).group()
            if file_name == video_name[:-4]:
                if int(re.search(r'(?<=_)[\d]+(?=\.jpg)', file).group()) % step_size == i:
                    frame_list.append(file)
        assert (len(frame_list) >= min_frames)  # just to make sure we don't have a bug.

        frame_list = sorted(frame_list, key=lambda x: int(re.search(r'(?<=_)[\d]+(?=\.jpg)', file).group()))
        # load and process images from list
        image_array = load_and_process_images(frames_dir, frame_list, to_numpy=True, resize_dim=final_dim)
        # output is a numpy array of frames

        # save image set as pickle
        with open(str(pickle_target_path.joinpath(video_name[:-4])) + '_' + str(i) + '.pickle', 'wb') as file:
            pickle.dump(image_array, file, protocol=pickle.HIGHEST_PROTOCOL)
            # print("Saved:", str(pickle_target_path.joinpath(video_name[:-4])) + '_' + str(i) + '.pickle')



    model = create_model(hyper_params)

    load_model(model, model_weights_path)
    # model.eval() - implemented in load model

    sum_predictions = None

    for pickle_clip in pickle_target_path.iterdir():
            if str(pickle_clip)[-9:] == '.DS_Store': continue
            prep_clip = prep_pickle_for_inference(pickle_clip)

            preds = model(prep_clip).squeeze(0) # squeeze to remove batch dimension
            print(preds)
            softmax_preds = torch.nn.functional.softmax(preds, dim=0)
            print(softmax_preds)

            if sum_predictions is None:
                    sum_predictions = softmax_preds
            else:
                    sum_predictions += softmax_preds

    print('The final prediction vector is:', sum_predictions.tolist())
    predicted_class = get_class_name(sum_predictions.max(0)[1].item())
    print('The final predicted class is:', predicted_class)


    # Clean up
    shutil.rmtree(frames_dir)
    shutil.rmtree(pickle_target_path)
