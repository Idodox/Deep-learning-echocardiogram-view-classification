from torchutils import *
import os
import json
from torch.nn import Softmax
from tqdm import tqdm
from utils import calibration
import numpy as np

#********* part 1: produce inference on val set
#
# hyper_params = {"batch_size": 64
#                ,"num_workers": 5
#                ,"max_frames": 10
#                ,"dataset": "10frame_5steps_100px"
#                ,"classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
#                ,"model_type": "3dCNN"
#                ,"resolution": 100
#                ,"adaptive_pool": [7,5,5]
#                ,"features": [16,16,"M",32,32,"M",32,32,"M",64,64,64,"M"]
#                ,"classifier": [0.5,200,0.5,150,0.4,100]
#                         }
#
# checkpoint_path = os.getcwd() + '/model_best.pt.tar'
# run_indexes_path = os.getcwd() + '/run_indexes.pickle'
#
#
# run = load_run_for_inference(hyper_params, checkpoint_path, 'pc')
#
# with open(run_indexes_path, 'rb') as handle:
#     (run.train_idx, run.val_idx) = pickle.load(handle)
#
# run.master_dataset = run.set_up_master_dataset()
# run.train_loader, run.val_loader = run.set_up_train_val_loaders()
#
#
# incorrect_classifications_val = []
# val_preds = []
# val_labels = []
# with torch.no_grad():
#     for batch_number, (images, labels, paths) in tqdm(enumerate(run.val_loader)):
#         if run.grayscale:
#             images = torch.unsqueeze(images, 1).double()  # added channel dimensions (grayscale)
#         else:
#             images = images.float().permute(0, 4, 1, 2, 3).float()
#         labels = labels.long()
#
#         if torch.cuda.is_available():
#             images, labels = images.cuda(), labels.cuda()
#
#         preds = run.model(images)  # Pass Batch
#         sm = Softmax()
#         preds_probability = sm(preds)
#
#         incorrect_classifications_val.append(get_mistakes(preds, labels, paths))
#
#         for prediction in zip(preds_probability, labels):
#             val_preds.append(prediction[0].tolist())
#             val_labels.append(prediction[1].item())
#
# print(incorrect_classifications_val)
#
# with open("temp_val_preds.json", 'w') as f:
#     json.dump(val_preds, f, indent=2)
# with open("temp_val_labels.json", 'w') as f:
#     json.dump(val_labels, f, indent=2)


#********* part 2: plot results

with open("temp_val_preds.json", 'r') as f:
    val_preds = json.load(f)

with open("temp_val_labels.json", 'r') as f:
    val_labels = json.load(f)

n_bins = 50

cal = calibration(val_labels, val_preds, num_bins=n_bins)

print(cal)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf_name in ['default']:

    mean_predicted_value = cal['reliability_diag'][0]
    fraction_of_positives = cal['reliability_diag'][1]
    samples_per_bucket = cal['samples_per_bucket']

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (clf_name, ))

    ax2.hist(np.max(val_preds, axis=1), range=(0, 1), bins=n_bins, label=clf_name, histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()

