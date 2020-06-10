from torchutils import *
from utils import *
from torchvision import transforms
from functools import partial
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from tqdm import tqdm
import numpy as np
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from utils import calc_accuracy, get_cross_val_idx


"""
This class holds the state for the run.
"""


class Run:

    def __init__(self, hyper_params, machine, disable_experiment = True, inference = False, cross_val = False,
                 checkpoint_path = ""):
        torch.backends.cudnn.benchmark=True

        self.hyper_params = hyper_params
        self.machine = machine
        self.disable_experiment = disable_experiment
        self.cross_val = cross_val
        self.log_number_train = 0
        self.log_number_val = 0
        self.best_val_acc = 0
        self.best_val_loss = np.inf
        self.best_model_mistakes = None
        self.best_model_preds = None
        self.grayscale = True
        self.color_channels = 1
        self.root_path = str()
        self.create_model()
        self.log_n_parameters()
        self.set_up_machine()
        self.set_up_color()
        self.set_up_transforms()
        self.set_up_criterion()
        if inference:
            self.model.eval()
            self.pretrained_model = load_checkpoint(self.model, checkpoint_path)
        else:
            self.set_up_master_dataset()
            self.set_up_optimizer()
            self.set_up_lr_scheduler()
            self.set_up_experiment()
            if cross_val:
                self.acc_list_val = []
                self.set_cross_val_idx()
                self.set_up_cross_val_vars()
            else:
                self.prepare_train_val_idx()
                self.initialize_train_val_loaders()


    def create_model(self):
        if self.hyper_params['model_type'] == "3dCNN":
            self.model = get_modular_3dCNN(self.hyper_params)
        elif self.hyper_params['model_type'] == 'resnext':
            self.model = get_resnext(self.hyper_params)
        else:
            raise NameError("Unknown model type")

    def reinitialize_run(self, train_idx, val_idx):
        self.best_val_acc = 0
        self.best_model_preds = None
        self.set_train_val_indexes(train_idx, val_idx)
        self.initialize_train_val_loaders()
        self.create_model()
        self.set_up_criterion()
        self.set_up_optimizer()
        self.set_up_lr_scheduler()


    def log_n_parameters(self):
        self.hyper_params['trainable_params'] = sum(p.numel() for p in self.model.parameters())
        print('N_trainable_params:', self.hyper_params['trainable_params'])

    def set_up_machine(self):
        if self.machine == 'server':
            self.root_path = str("/home/ido/data/" + self.hyper_params['dataset'])
        elif self.machine == 'pc':
            self.root_path = str('/Users/idofarhi/Documents/Thesis/Data/frames/' + self.hyper_params['dataset'])
        else:
            raise NameError("Unknown machine")

    def set_up_transforms(self):
        self.data_transforms = transforms.Compose([
            ToTensor()
            ,Normalize(0.213303, 0.21379)
            # ,RandomHorizontalFlip(self.hyper_params["flip_prob"])
        ])

    def set_up_master_dataset(self):
        self.master_dataset = DatasetFolderWithPaths(self.root_path
                                        ,transform = self.data_transforms
                                        ,loader = partial(pickle_loader, min_frames = self.hyper_params['max_frames'], shuffle_frames = False)
                                        ,extensions = '.pickle'
                                        )
    def prepare_train_val_idx(self):
        self.set_train_val_indexes(extract_train_val_idx(self.master_dataset, random_state = self.hyper_params['random_seed'], test_size = 0.2))

    def set_train_val_indexes(self, train_idx, val_idx):
        self.train_idx, self.val_idx = train_idx, val_idx

    def set_cross_val_idx(self):
        self.cross_val_idx = get_cross_val_idx(self.master_dataset, random_state = self.hyper_params['random_seed'], n_splits = self.hyper_params['k_folds'])

    def set_up_cross_val_vars(self):
        self.best_folds_val_acc = []
        self.best_folds_model_preds = []


    def initialize_train_val_loaders(self):

        train_set = torch.utils.data.Subset(self.master_dataset, self.train_idx)
        val_set = torch.utils.data.Subset(self.master_dataset, self.val_idx)

        self.train_loader = torch.utils.data.DataLoader(train_set
                                             , batch_size=self.hyper_params['batch_size']
                                             , shuffle=True
                                             # ,batch_sampler =  # TODO: add stratified sampling
                                             , num_workers=self.hyper_params['num_workers']
                                             , drop_last=False
                                             )

        # online_mean_and_std(train_loader)

        self.val_loader = torch.utils.data.DataLoader(val_set
                                             , batch_size=self.hyper_params['batch_size']
                                             , shuffle=True
                                             # ,batch_sampler =  # TODO: add stratified sampling
                                             , num_workers=self.hyper_params['num_workers']
                                             , drop_last=False
                                             )


    def set_up_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper_params['learning_rate'])

    def set_up_lr_scheduler(self):
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hyper_params["lr_scheduler"]["step_size"], gamma=self.hyper_params["lr_scheduler"]["gamma"])

    def set_up_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def set_up_experiment(self):
        self.experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1" ,project_name="thesis" ,workspace="idodox", disabled=self.disable_experiment)
        self.experiment.log_parameters(self.hyper_params)

    def set_up_color(self):
        if "color" in self.hyper_params['dataset']:
            self.grayscale = False

    def print_summary(self):
        summary(self.model, (self.color_channels, self.hyper_params["max_frames"], self.hyper_params["resolution"], self.hyper_params["resolution"]))

    def log_confusion_matrices(self, print_matrix = True, mod_name = ''):
        cm = generate_cm(self.best_model_preds)
        if print_matrix:
            print(cm)
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm.tolist(), title = mod_name + "Confusion matrix, individual clips", file_name= mod_name + "individual_clips.json")
        cm = confusion_matrix(*calc_accuracy(self.best_model_preds, method = 'sum_predictions', export_for_cm=True))
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm, title = mod_name + "Confusion matrix, sum predictions", file_name= mod_name + "sum_predictions.json")
        cm = confusion_matrix(*calc_accuracy(self.best_model_preds, method = 'majority_vote', export_for_cm=True))
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm, title = mod_name + "Confusion matrix, majority vote", file_name= mod_name + "majority_vote.json")

    def log_cv_confusion_matrices(self, mod_name = ''):
        cm_all_videos = np.empty((10,6,6))
        cm_sum_predictions = np.empty((10,6,6))
        cm_majority_vote = np.empty((10,6,6))
        for i, preds in enumerate(self.best_folds_model_preds):
            cm_all_videos[i] = generate_cm(preds)
            cm_sum_predictions[i] = confusion_matrix(*calc_accuracy(preds, method = 'sum_predictions', export_for_cm=True))
            cm_majority_vote[i] = confusion_matrix(*calc_accuracy(preds, method = 'majority_vote', export_for_cm=True))

        cm = cm_all_videos.sum(axis=0)
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm, title = mod_name + "Confusion matrix, individual clips", file_name= mod_name + "individual_clips.json")
        cm = cm_sum_predictions.sum(axis=0)
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm, title = mod_name + "Confusion matrix, sum predictions", file_name= mod_name + "sum_predictions.json")
        cm = cm_majority_vote.sum(axis=0)
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm, title = mod_name + "Confusion matrix, majority vote", file_name= mod_name + "majority_vote.json")

    def save_model(self, path, filename):
        torch.save(self.model.state_dict(), path + filename)
        print("model saved at:", path + filename)

    def save_run_indexes(self, path, filename):
        # save run indexes tuple as pickle
        with open(path + filename, 'wb') as file:
            pickle.dump((self.train_idx, self.val_idx), file, protocol=pickle.HIGHEST_PROTOCOL)
