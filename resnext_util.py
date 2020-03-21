import torch
from torch import nn
import resnext


def generate_resnext_model(mode, model_depth = 101, n_classes = 3, sample_size = 112, sample_duration = 16, resnet_shortcut = 'A', resnext_cardinality = 32):
    assert mode in ['score', 'feature']
    if mode == 'score':
        last_fc = True
    elif mode == 'feature':
        last_fc = False

    assert model_depth in [50, 101, 152]

    if model_depth == 50:
        model = resnext.resnet50(num_classes=n_classes, shortcut_type=resnet_shortcut, cardinality=resnext_cardinality,
                                 sample_size=sample_size, sample_duration=sample_duration,
                                 last_fc=last_fc)
    elif model_depth == 101:
        model = resnext.resnet101(num_classes=n_classes, shortcut_type=resnet_shortcut, cardinality=resnext_cardinality,
                                  sample_size=sample_size, sample_duration=sample_duration,
                                  last_fc=last_fc)
    elif model_depth == 152:
        model = resnext.resnet152(num_classes=n_classes, shortcut_type=resnet_shortcut, cardinality=resnext_cardinality,
                                  sample_size=sample_size, sample_duration=sample_duration,
                                  last_fc=last_fc)

    return model