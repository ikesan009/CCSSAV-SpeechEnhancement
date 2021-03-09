#encoding: utf-8
import math
import torch
import torch.utils.data
import numpy as np
from _utils.dataset import *
from torch.optim.optimizer import Optimizer


def data_loader(args, assets):
    print('-' * 95)
    dsets = {x: MyDataset(x, args, assets) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('Statistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'],
                                                              dset_sizes['test']))
    video_shape, audio_spectrogram_shape = dsets['train'].get_shapes()
    return dset_loaders, dset_sizes, video_shape, audio_spectrogram_shape


def reload_model(assets, name_model, model, epoch, logger=None):
    if logger:
        print('reload model '+name_model+'...')
    if epoch==0:
        if logger:
            logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(assets.get_model_cache_path(name_model, epoch-1))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if logger:
            logger.info('*** model ['+str(epoch)+'] has been successfully loaded! ***')
        return model


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


class AdjustLR(object):
    def __init__(self, optimizer, init_lr, sleep_epochs=5, half=5, verbose=0):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.sleep_epochs = sleep_epochs
        self.half = half
        self.init_lr = init_lr
        self.verbose = verbose

    def step(self, epoch):
        if epoch >= self.sleep_epochs:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.init_lr[idx] * math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
                param_group['lr'] = new_lr
            if self.verbose:
                print('>>> reduce learning rate <<<')