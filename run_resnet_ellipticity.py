import numpy as np
import os, sys
import torch
import pickle, csv
import argparse
from run_resnet import custom_resnet
from run_resnet import ResNetDataset
from train import train
from pathlib import Path

from torchvision.transforms.functional import pad

class EllipticityDataset(ResNetDataset):
    def __getitem__(self, index):
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
        # gamma1_los = float(kwarg_val[1]) + float(kwarg_val[5]) - float(kwarg_val[9])
        # gamma2_los = float(kwarg_val[2]) + float(kwarg_val[6]) - float(kwarg_val[10])
        # y_tensor = torch.tensor([gamma1_los, gamma2_los])

        e1_bar = float(kwarg_val[19])
        e2_bar = float(kwarg_val[20])
        e1_nfw = float(kwarg_val[28])
        e2_nfw = float(kwarg_val[29])
        e1_sl = float(kwarg_val[36])
        e2_sl = float(kwarg_val[37])
        e1_ll = float(kwarg_val[40])
        e2_ll = float(kwarg_val[41])
        
        y_tensor = torch.tensor([e1_bar, e2_bar, e1_nfw, e2_nfw, e1_sl, e2_sl, e1_ll, e2_ll])
        
        return x_tensor, y_tensor
    
if __name__ == '__main__':

    # Train a ResNet model, but unlike in `run_resnet.py`, try to predict the
    # gamma_LOS components in bins
    
    parser = argparse.ArgumentParser()
    parser.add_argument('num_workers', type=int)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--pdrop', type=float, default=0)
    args = parser.parse_args()

    # And we want to make a bin prediction for each axis, so n_out=2*nbins.
    model = custom_resnet('resnet18', n_out=8, pdrop=args.pdrop)

    data_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/elronddata/datasets'
    xtrain_filenames = ['mlfodder_{}_image_list.pickle'.format(i) for i in range(100)]
    ytrain_filenames = ['mlfodder_{}_input_kwargs.csv'.format(i) for i in range(100)]
    xval_filenames = ['mlfodder_{}_image_list.pickle'.format(i) for i in range(100,101)]
    yval_filenames = ['mlfodder_{}_input_kwargs.csv'.format(i) for i in range(100,101)]

    train_dataset = EllipticityDataset(data_dir, xtrain_filenames, ytrain_filenames, 1000, normalization='log10')
    val_dataset = EllipticityDataset(data_dir, xval_filenames, yval_filenames, 1000, normalization='log10')

    # ellip1: 100 train files, 1 val, try to predict 8 ellipticity params (bar, nfw, sl, ll)
    mname = 'ellip1'
    batch_size = args.batch
    num_workers = args.num_workers
    to_normalize = False

    # criterion = make_bin_criterion(bin_edges)
    criterion = torch.nn.MSELoss()

    train(model, (train_dataset, val_dataset), mname, batch_size, num_workers,
          to_normalize=to_normalize, criterion=criterion)
