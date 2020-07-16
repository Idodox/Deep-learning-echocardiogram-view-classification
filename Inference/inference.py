from inference_utils import inference
from pathlib import Path


dicom_file_path = Path('/Users/idofarhi/Documents/Thesis/to_label/AA124128/dicom/H23CD6SQ.dcm')

hyper_params = {"max_frames": 10
    , "random_seed": 999
    , "classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
    , "model_type": "3dCNN"
    , "resolution": 100
    , "adaptive_pool": (7, 5, 5)
    , "features": [16, 16, 16, "M", 32, 32, 32, "M", 32, 32, 32, "M", 64, 64, 64, "M"]
    , "classifier": [200, 0.5, 150, 0.5]
                }

model_weights_path = '/Users/idofarhi/Documents/Thesis/model.pt'


inference(dicom_file_path, hyper_params, model_weights_path)
