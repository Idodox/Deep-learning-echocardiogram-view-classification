import torch
from torch import nn
import resnext


def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name == 'resnext'

    assert opt.model_depth in [50, 101, 152]

    if opt.model_depth == 50:
        model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                 sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                 last_fc=last_fc)
    elif opt.model_depth == 101:
        model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                  sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                  last_fc=last_fc)
    elif opt.model_depth == 152:
        model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                  sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                  last_fc=last_fc)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model