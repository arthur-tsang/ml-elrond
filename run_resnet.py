import numpy as np
import os, sys
import datetime

from copy import deepcopy

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

# from resnet import ResNetEstimator
from torchvision.models.resnet import resnet18, resnet34
from torchvision.transforms.functional import pad

import pickle
import csv
from pathlib import Path

from train import train

import argparse

class ResNetDataset(Dataset):

    def __init__(self, data_dir, x_filenames, y_filenames, imgs_per_file, maxlen=np.inf, normalization='log10'):
        assert isinstance(imgs_per_file, int)
        assert maxlen == np.inf or isinstance(maxlen, int)
        assert normalization in ['none', 'log10']
        assert len(x_filenames) == len(y_filenames)
        
        self.data_dir = data_dir
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.imgs_per_file = imgs_per_file
        self.maxlen = maxlen
        self.normalization = normalization
        
    def __getitem__(self, index):
        # The current structure is that we have files of about 1000 images each,
        # and we open the relevant file and pick the relevant image from it.
        # This might be inefficient if we really need to do open the same file
        # every time though... We also need to worry about the padding, since
        # each image is a different size, but the standard resnet requires a
        # fixed input size for the fully connected layer at the end.
        file_number = index // self.imgs_per_file
        idx_in_file = index % self.imgs_per_file

        with open(self.data_dir / Path(self.x_filenames[file_number]), 'rb') as f:
            image_list = pickle.load(f)
        x_numpy = image_list[idx_in_file]
        x_unpadded = torch.from_numpy(x_numpy).float()
        old_size = x_unpadded.shape[0]
        new_size = 256 # round number near maximum size in all our data
        size_diff = new_size - old_size
        padding = [size_diff//2, size_diff//2, size_diff//2 + size_diff%2, size_diff//2 + size_diff%2]
        if self.normalization == 'none':
            x_tensor = pad(x_unpadded, padding)
        elif self.normalization == 'log10':
            # The clamping is to make sure all the log values are well-defined,
            # and not too extremely negative.
            x_tensor = np.log10(torch.clamp(pad(x_unpadded, padding), 1e-6, None))
        else:
            raise ValueError('Unrecognized normalization')

        # Then expand for 3x channels
        x_tensor = x_tensor.expand(3, -1, -1)


        with open(self.data_dir / Path(self.y_filenames[file_number])) as f:
            input_kwargs = csv.reader(f, delimiter=',')

            kwargs_labels = next(input_kwargs)
            kwarg_vals = [x for x in input_kwargs]

        kwarg_val = kwarg_vals[idx_in_file]
        gamma1_los = float(kwarg_val[1]) + float(kwarg_val[5]) - float(kwarg_val[9])
        gamma2_los = float(kwarg_val[2]) + float(kwarg_val[6]) - float(kwarg_val[10])
        y_tensor = torch.tensor([gamma1_los, gamma2_los])

        return x_tensor, y_tensor

    def __len__(self):
        return min(self.maxlen, self.imgs_per_file * len(self.x_filenames))
    
## We will "define" our resnet in terms of the default torch `resnet18`
## implementation. But we want to be able to apply dropout to control
## overfitting, so we use the following function.

def append_dropout(model, pdrop):
    """Helper function to modify `model` to replace every ReLU with a sequence of
    ReLU and dropout.

    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # Need to recurse
            append_dropout(module, pdrop)
        if isinstance(module, nn.ReLU):
            # replace relu with relu and dropout
            new = nn.Sequential(module, nn.Dropout2d(p=pdrop, inplace=False))
            setattr(model, name, new)

    ## No return, since we did everything "in place"

def custom_resnet18(n_out=2, pdrop=0):
    """Function to produce a resnet based on the pretrained default but with a
    custom number of outputs and dropout

    """
    return custom_resnet(resnet18, n_out, pdrop)

def custom_resnet(resnet_type, n_out, pdrop):
    if resnet_type == 'resnet18':
        model = resnet18(pretrained=True)
    elif resnet_type == 'resnet34':
        model = resnet34(pretrained=True)
    elif isinstance(resnet_type, str):
        raise ValueError('ResNet type {} not recognized'.format(resnet_type))
    else:
        model = resnet_type(pretrained=True)

    ## We need to replace the last layer with a new linear layer that has the
    ## right number of output classes. Also, we're adding dropout to this last
    ## layer.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_out)

    ## Then we add dropout to every ReLU within the resnet itself
    append_dropout(model, pdrop)

    return model
    

if __name__ == '__main__':
    # Train a ResNet model
    
    parser = argparse.ArgumentParser()
    parser.add_argument('num_workers', type=int)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--pdrop', type=float, default=0)
    args = parser.parse_args()

    # First will try as a regression problem. This means one output for each LOS component.
    # model = custom_resnet18(n_out=2, pdrop=args.pdrop)
    model = custom_resnet(resnet34, 2, args.pdrop)
    
    data_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/elronddata/datasets'
    xtrain_filenames = ['mlfodder_{}_image_list.pickle'.format(i) for i in range(100)]
    ytrain_filenames = ['mlfodder_{}_input_kwargs.csv'.format(i) for i in range(100)]
    xval_filenames = ['mlfodder_{}_image_list.pickle'.format(i) for i in range(100,101)]
    yval_filenames = ['mlfodder_{}_input_kwargs.csv'.format(i) for i in range(100,101)]
    
    train_dataset = ResNetDataset(data_dir, xtrain_filenames, ytrain_filenames, 1000, normalization='log10')
    val_dataset = ResNetDataset(data_dir, xval_filenames, yval_filenames, 1000, normalization='log10')

    # regress1: 9 train files, 1 val
    # regress2: 100 train files, 1 val
    # regress3: Switching from resnet18 to resnet34
    mname = 'regress3'
    batch_size = args.batch
    num_workers = args.num_workers
    to_normalize = False

    criterion = torch.nn.MSELoss()
    
    train(model, (train_dataset, val_dataset), mname, batch_size, num_workers, to_normalize=to_normalize, criterion=criterion)
