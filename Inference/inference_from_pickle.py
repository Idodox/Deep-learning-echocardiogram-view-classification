import torch
from torchutils import create_model, load_checkpoint, load_model
from vis_utils import prep_pickle_for_inference
from pathlib import Path
from utils import get_class_name

############### CONFIG ########################

pickle_clips_path = Path('/Users/idofarhi/Documents/Thesis/to_label/temp/pickle/')


hyper_params = {"max_frames": 10
        , "random_seed": 999
        , "classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
        , "model_type": "3dCNN"
        , "resolution": 100
        , "adaptive_pool": (7, 5, 5)
        , "features": [16,16,16,"M",32,32,32,"M",32,32,32,"M",64,64,64,"M"]
        ,"classifier": [200,0.5,150,0.5]
     }

#################################################

weights_path = '/Users/idofarhi/Documents/Thesis/fold6model.pt'

model = create_model(hyper_params)

load_model(model, weights_path)
model.eval()

sum_predictions = None

for pickle_clip in pickle_clips_path.iterdir():
        if str(pickle_clip)[-9:] == '.DS_Store': continue
        prep_clip = prep_pickle_for_inference(pickle_clip)

        preds = model(prep_clip).squeeze(0) # squeeze to remove batch dimension
        softmax_preds = torch.nn.functional.softmax(preds, dim=0)
        # print(softmax_preds)

        if sum_predictions is None:
                sum_predictions = softmax_preds
        else:
                sum_predictions += softmax_preds

print('The final prediction vector is:', sum_predictions.tolist())
predicted_class = get_class_name(sum_predictions.max(0)[1].item())
print('The final predicted class is:', predicted_class)
