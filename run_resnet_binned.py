import os, sys
import numpy as np
import torch
import argparse
from run_resnet import ResNetDataset
from run_resnet import custom_resnet18
from train import train

def make_bin_criterion(bin_edges):
    nbins = len(bin_edges) + 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bin_edges = torch.tensor(bin_edges, device=device).contiguous()
    
    def bin_criterion(outputs, labels):
        """This is our loss function

        (Remember the first dimension is the batch, so we index over the second dimension)"""

        # outputs.shape is (8, 16)
        # labels.shape is (8, 2)
        
        x_idx = torch.bucketize(labels[:,0], bin_edges)
        y_idx = torch.bucketize(labels[:,1], bin_edges)
        
        x_loss = torch.nn.functional.cross_entropy(outputs[:,:nbins], x_idx)
        y_loss = torch.nn.functional.cross_entropy(outputs[:,nbins:], y_idx)

        return x_loss + y_loss

    return bin_criterion
    
if __name__ == '__main__':

    # Train a ResNet model, but unlike in `run_resnet.py`, try to predict the
    # gamma_LOS components in bins
    
    parser = argparse.ArgumentParser()
    parser.add_argument('num_workers', type=int)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--pdrop', type=float, default=0)
    args = parser.parse_args()

    # Now we will try this as a binned classification. Judging by the histogram
    # of values along each component, it seems reasonble to define the bins as
    # follows:
    bin_edges = [-.03, -.02, -.01, 0, .01, .02, .03]
    nbins = len(bin_edges) + 1 # i.e. 8
    # And we want to make a bin prediction for each axis, so n_out=2*nbins.
    model = custom_resnet18(n_out=2*nbins, pdrop=args.pdrop)

    data_dir = '/n/holyscratch01/dvorkin_lab/Users/atsang/elronddata/datasets'
    xtrain_filenames = ['mlfodder_{}_image_list.pickle'.format(i) for i in range(100)]
    ytrain_filenames = ['mlfodder_{}_input_kwargs.csv'.format(i) for i in range(100)]
    xval_filenames = ['mlfodder_{}_image_list.pickle'.format(i) for i in range(100,101)]
    yval_filenames = ['mlfodder_{}_input_kwargs.csv'.format(i) for i in range(100,101)]

    train_dataset = ResNetDataset(data_dir, xtrain_filenames, ytrain_filenames, 1000, normalization='log10')
    val_dataset = ResNetDataset(data_dir, xval_filenames, yval_filenames, 1000, normalization='log10')

    # bin1: 100 train files, 1 val
    mname = 'bin1'
    batch_size = args.batch
    num_workers = args.num_workers
    to_normalize = False

    criterion = make_bin_criterion(bin_edges)

    train(model, (train_dataset, val_dataset), mname, batch_size, num_workers, to_normalize=to_normalize, criterion=criterion)
