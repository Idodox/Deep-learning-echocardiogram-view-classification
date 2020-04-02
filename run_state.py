from torchutils import *
from utils import *
from torchvision import transforms
from functools import partial
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from sklearn.metrics import confusion_matrix


"""
This class holds the state for the run.
"""


class Run:

    def __init__(self, hyper_params, machine, disable_experiment = True):
        torch.backends.cudnn.benchmark=True

        self.hyper_params = hyper_params
        self.machine = machine
        self.disable_experiment = disable_experiment
        self.log_number_train = 0
        self.log_number_val = 0
        self.best_val_acc = 0
        self.best_model_preds = None
        self.grayscale = True
        self.color_channels = 1
        self.root_path = str()
        self.model = self.create_model()
        self.log_n_parameters()
        self.set_up_machine()
        self.data_transforms = self.set_up_transforms()
        self.master_dataset = self.set_up_master_dataset()
        self.train_loader, self.val_loader = self.set_up_train_val_loaders()
        self.optimizer = self.set_up_optimizer()
        self.criterion = self.set_up_criterion()
        self.experiment = self.set_up_experiment()
        self.set_up_color()

    def create_model(self):
        if self.hyper_params['model_type'] == "3dCNN":
            return get_modular_3dCNN(self.hyper_params)
        elif self.hyper_params['model_type'] == 'resnext':
            return get_resnext(self.hyper_params)
        else:
            raise NameError("Unknown model type")

    def log_n_parameters(self):
        self.hyper_params['trainable_params'] = sum(p.numel() for p in self.model.parameters())
        print('N_trainable_params:', self.hyper_params['trainable_params'])

    def set_up_machine(self):
        if self.machine == 'server':
            self.root_path = str("/home/ido/data/" + self.hyper_params['dataset'])
        elif self.machine == 'local':
            self.root_path = str('/Users/idofarhi/Documents/Thesis/Data/frames/' + self.hyper_params['dataset'])
        else:
            raise NameError("Unknown machine")

    def set_up_transforms(self):
        return transforms.Compose([
            ToTensor()
            ,Normalize(0.213303, 0.21379)
            # ,RandomHorizontalFlip(self.hyper_params["flip_prob"])
        ])

    def set_up_master_dataset(self):
        return DatasetFolderWithPaths(self.root_path
                                        ,transform = self.data_transforms
                                        ,loader = partial(pickle_loader, min_frames = self.hyper_params['max_frames'], shuffle_frames = False)
                                        ,extensions = '.pickle'
                                        )

    def set_up_train_val_loaders(self):

        train_idx, val_idx = get_train_val_idx(self.master_dataset, random_state = self.hyper_params['random_seed'], test_size = 0.2)

        train_set = torch.utils.data.Subset(self.master_dataset, train_idx)
        val_set = torch.utils.data.Subset(self.master_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_set
                                             , batch_size=self.hyper_params['batch_size']
                                             , shuffle=True
                                             # ,batch_sampler =  # TODO: add stratified sampling
                                             , num_workers=self.hyper_params['num_workers']
                                             , drop_last=False
                                             )

        # online_mean_and_std(train_loader)

        val_loader = torch.utils.data.DataLoader(val_set
                                             , batch_size=self.hyper_params['batch_size']
                                             , shuffle=True
                                             # ,batch_sampler =  # TODO: add stratified sampling
                                             , num_workers=self.hyper_params['num_workers']
                                             , drop_last=False
                                             )

        return train_loader, val_loader

    def set_up_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.hyper_params['learning_rate'])

    def set_up_criterion(self):
        return nn.CrossEntropyLoss()

    def set_up_experiment(self):
        experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1" ,project_name="thesis" ,workspace="idodox", disabled=self.disable_experiment)
        experiment.log_parameters(self.hyper_params)
        return experiment

    def set_up_color(self):
        if "color" in self.hyper_params['dataset']:
            self.grayscale = False

    def print_summary(self):
        summary(self.model, (self.color_channels, self.hyper_params["max_frames"], self.hyper_params["resolution"], self.hyper_params["resolution"]))

    def log_confusion_matrix(self):
        y_pred, y_true = list(), list()
        for (pred, true, path) in self.best_model_preds:
            y_pred.append(max(pred, 0)[1].item())
            y_true.append(true.item())
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        self.experiment.log_confusion_matrix(labels=self.train_loader.dataset.dataset.classes, matrix=cm)

