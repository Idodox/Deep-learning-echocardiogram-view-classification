import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision.transforms import ToTensor
import numpy as np



class CNNCell(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNCell, self).__init__()
        self.conv = nn.Conv3d(in_channels=input_channels,
                              kernel_size=3,
                              output_channels=output_channels)
        self.bn = nn.BatchNorm3d(num_features=output_channels)
        self.relu = nn.Relu()

        def forward(self, batch_data):
            t = self.conv(batch_data)
            t = self.bn(t)
            t = self.relu(t)

            return t


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()

        # self.conv1_ch = params["conv1_ch"]
        # self.conv1_kernel = params["conv1_kernel"]
        # self.conv2_ch = params["conv2_ch"]
        # self.conv2_kernel = params["conv2_kernel"]
        # self.conv3_ch = params["conv3_ch"]
        # self.conv3_kernel = params["conv3_kernel"]
        # self.conv4_ch = params["conv4_ch"]
        # self.conv4_kernel = params["conv4_kernel"]
        # self.maxpool1_kernel = params["maxpool1_kernel"]
        # self.fc1_size = params["fc1_size"]
        # self.dropout1_ratio = params["dropout1_ratio"]
        # self.fc2_size = params["fc2_size"]
        # self.dropout2_ratio = params["dropout2_ratio"]
        # self.fc3_size = params["fc3_size"]
        # self.dropout3_ratio = params["dropout3_ratio"]

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.features = nn.Sequential(CNNCell(input_channels=1, output_channels=32)
                                     ,CNNCell(input_channels=32, output_channels=32)
                                     ,CNNCell(input_channels=32, output_channels=32)
                                     ,nn.MaxPool3d(kernel_size=2)
                                     ,CNNCell(input_channels=32, output_channels=64)
                                     ,CNNCell(input_channels=64, output_channels=64)
                                     ,CNNCell(input_channels=64, output_channels=64)
                                     ,nn.MaxPool3d(kernel_size=2)
                                      )

        self.classifier = nn.Sequential(nn.Dropout(0.5)
                                       ,nn.Linear(128)
                                       ,nn.ReLU() # arg inplace = True?
                                       ,nn.Dropout(0.5)
                                       ,nn.Linear(64)
                                       ,nn.ReLU(inplace = True)
                                        )

        self.to(self.device)

    def forward(self, batch_data):
        batch_data = T.tensor(batch_data).to(self.device)

        t = self.features(batch_data)
        t = torch.flatten(t)
        t = self.classifier(t)

        return t
