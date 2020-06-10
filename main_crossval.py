from comet_ml import Experiment #not in use here but must be imported first
from tqdm import tqdm
from run_state import Run
from torchutils import *
from utils import extract_video_names

print('CUDA available:', torch.cuda.is_available())
print('CUDA enabled:', torch.backends.cudnn.enabled)
# torch.cuda.empty_cache()
torch.cuda.get_device_capability(0)


HP1 = {"learning_rate": 0.001
      ,"lr_scheduler": {'step_size': 5, 'gamma': 0.8}
      ,"n_epochs": 40
      ,"k_folds": 10
      ,"batch_size": 128
      ,"num_workers": 4
      ,"normalized_data": True
      ,"stratified": False
      ,"horizontal_flip": False
      ,"max_frames": 10
      ,"random_seed": 43
      ,"flip_prob": 0.5
      ,"dataset": "10frame_5steps_100px"
      ,"classes": ['apex', 'papillary', 'mitral', '2CH', '3CH', '4CH']
      ,"model_type": "3dCNN"
      ,"resolution": 100
      ,"adaptive_pool": [7,5,5]
      ,"features": [16,16,"M",32,32,"M",32,32,"M",64,64,64,"M"]
      ,"classifier": [200,0.6,150,0.6,100]
                }

save_experiment = True

run = Run(disable_experiment = not save_experiment,
          cross_val = True,
          machine = 'server',
          hyper_params= HP1)


acc_list_train = []
acc_list_val = []

for fold_number, (train_idx, val_idx) in enumerate(run.cross_val_idx):
    print("\n*********************   starting fold number {}   *********************\n".format(fold_number))

    run.reinitialize_run(train_idx, val_idx)

    for epoch in tqdm(range(HP1["n_epochs"])):

        train(epoch, run, mod_name = "fold {}: ".format(fold_number))
        evaluate(epoch, run, mod_name = "fold {}: ".format(fold_number))

    run.best_folds_val_acc.append(run.best_val_acc)
    run.best_folds_model_preds.append(run.best_model_preds)
    run.log_confusion_matrices(mod_name = "fold_{}_".format(fold_number))
    run.save_model(os.getcwd(), "/fold{}model.pt".format(fold_number))
    print(extract_video_names(run.best_model_mistakes))

    run.experiment.log_other('Best accuracy fold {}'.format(fold_number), run.best_val_acc)

run.log_cv_confusion_matrices(mod_name = 'Cumulative_CV_')
run.save_run_indexes(os.getcwd(), "/cv_run_indexes.pickle")
run.experiment.log_other('Run finished', True)
