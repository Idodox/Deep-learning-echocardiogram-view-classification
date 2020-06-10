import torch
import torch.nn as nn
from numpy import prod



class ModularCNN(nn.Module):

    def __init__(self, features, classifier, adaptive_pool=(6, 6, 6), num_classes=3, init_weights=True):
        super(ModularCNN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool3d(adaptive_pool)
        self.classifier = nn.Sequential(
            nn.Linear(features[-4].out_channels * prod(adaptive_pool), classifier[0]),
            nn.ReLU(True),
            nn.Dropout(classifier[1]),
            nn.Linear(classifier[0], classifier[2]),
            nn.ReLU(True),
            nn.Dropout(classifier[3]),
            nn.Linear(classifier[2], num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=(1, 2, 2))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

