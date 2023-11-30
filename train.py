## Run the training loop, with checkpoints

import numpy as np
import os, sys
import datetime

from copy import deepcopy

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset


def train(model, dataset_train_val, mname, batch_size, num_workers,
          opt='adam', lr=1e-3, to_normalize=True,
          criterion=torch.nn.CrossEntropyLoss()):
    """Trains and saves a NN model
    """
    
    print('running mname', mname)

    assert len(dataset_train_val) == 2


    pin_memory = True # This might help if we're close to using all the GPU memory
    train_dataset = dataset_train_val[0]
    val_dataset = dataset_train_val[1]
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    ################################################################################
    ## Create model (main_fast)
    ################################################################################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, flush=True)

    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)

    ## Optimizer ##
    
    ## In case we want to try different optimization algorithms.
    optim_params = model.parameters()
    if opt == 'adam':
        optimizer = optim.Adam(optim_params, lr=lr)    
    elif opt == 'sgd':
        optimizer = optim.SGD(optim_params, lr=lr, momentum=0.9)
    else:
        raise ValueError('`opt` must be either "adam" or "sgd"')

    ## Loading from checkpoints ##
    save_path = 'Models/{}.tar'.format(mname)
    chk_path = 'Models/chk_{}.tar'.format(mname)
    init_path = 'Models/init_{}.tar'.format(mname) # just a quick slightly hacky way of initializing on a nicer set of weights
    
    ## This next section of the code is a little messy, but the point is that we
    ## save regular checkpoints with the model weights and will automatically
    ## pick up training from these files if they are found. We save not only the
    ## weights but also the state of the learning rate optimizer and a record of
    ## the losses by the end of every epoch.

    if os.path.islink(init_path):
        # Redefine `init_path` as an actual file path rather than a symlink address
        init_path = os.readlink(init_path)
    
    if os.path.isfile(save_path):
        # if a regular save file exists (these are saved at the end of every epoch)
        print('continuing partially-trained model')
        loaded = torch.load(save_path)

        model_dict_updt = {'model_state_dict': loaded['model_state_dict'],
                           'TrainingLoss': loaded['TrainingLoss'],
                           'ValidationLoss': loaded['ValidationLoss']}

        optimizer.load_state_dict(loaded['optimizer_state_dict'])
        if optimizer.state_dict()['param_groups'][0]['lr'] != lr:
            print(f'Warning! Using the loaded learning rate of {optimizer.state_dict()["param_groups"][0]["lr"]}'
                  f' which may be different from the learning rate {lr} specified to the function `train`.')
        
        start_epoch = loaded['epoch']

        ## Great, now we loaded and processed all the information except the
        ## model weights, so let's see if there's a checkpoint from midway
        ## through the epoch which we can start training from.
        loaded_chk = torch.load(chk_path) if os.path.isfile(chk_path) else None
        if loaded_chk and loaded_chk['epoch'] >= start_epoch:
            # if found checkpoint file (for saving in the middle of an epoch)
            # and the checkpoint is new enough to be useful

            assert loaded_chk['epoch'] == start_epoch # make sure the checkpoint is not too far in advance to make sense
            print('Using checkpoint file')

            model.load_state_dict(loaded_chk['tmp_checkpoint'])
            checkpoint_batch_nr = loaded_chk['batch_nr']
            use_chk_file = True
        else:
            # if not found checkpoint file, then we just use the regular save file
            print('Not using checkpoint file (may or may not exist)')

            model.load_state_dict(loaded['model_state_dict'][-1])
            use_chk_file = False
    elif os.path.isfile(chk_path):
        # if there is a checkpoint file, but we haven't completed a full epoch
        # yet (i.e. we're picking up from being stopped partway through Epoch 0)
        loaded =  torch.load(chk_path)

        model_dict_updt = {'model_state_dict': [],
                           'TrainingLoss': [],
                           'ValidationLoss': []}

        start_epoch = loaded['epoch']
        assert start_epoch == 0
        
        print('Using checkpoint file (no full-epoch save file found)')
        
        model.load_state_dict(loaded['tmp_checkpoint'])
        checkpoint_batch_nr = loaded['batch_nr']
        use_chk_file = True
    else:
        # if no save file exists whatsoever
        model_dict_updt={'model_state_dict': [],
                         'TrainingLoss': [],
                         'ValidationLoss': []}
        start_epoch = 0
        use_chk_file = False

        if os.path.isfile(init_path):
            print('No save file found, but using initalization file {}'.format(init_path))
            
            loaded = torch.load(init_path)
            init_state_dict = loaded['model_state_dict'][-1] if 'model_state_dict' in loaded else loaded['tmp_checkpoint']
            model.load_state_dict(init_state_dict)
            del loaded
            
    ############################################################################
    # In case we are loading from a checkpoint, we'll need a special DataLoader
    # that starts where the checkpoint starts.
    if use_chk_file:
        chk_sampler_start = (checkpoint_batch_nr + 1) * batch_size # +1 to not redo checkpoint batch again after saving
        chk_dataset = Subset(train_dataset, range(chk_sampler_start, len(train_dataset)))
        chk_loader = DataLoader(chk_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        
        ## chk_loader is just like train_loader, except it starts right after the checkpoint

        
    ############################################################################

    ## Learning rate schedule (reduce learning rate by a factor of 10 every time we plateau for 5 epochs)
    Scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=5,
                                                     verbose=True,
                                                     min_lr=1e-6)
    
    ################################################################################
    ## Training Loop (finally, this is the part where we start training)
    ################################################################################

    model_state_dicts = model_dict_updt['model_state_dict']
    epoch_dict = {'TrainingLoss': model_dict_updt['TrainingLoss'],
                  'ValidationLoss': model_dict_updt['ValidationLoss'],
                  'Save': mname}
    min_loss = np.inf
    stopping_patience_START = 15
    stopping_patience = stopping_patience_START

    # criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1000):
        if epoch < start_epoch:
            continue

        print('time stamp:', datetime.datetime.now(), flush=True)

        model.train()
        train_loss = 0
        train_size = 0
        if use_chk_file and epoch == start_epoch:
            relevant_train_loader = chk_loader
            i_batch_offset = checkpoint_batch_nr + 1
        else:
            relevant_train_loader = train_loader
            i_batch_offset = 0

        for i_batch, (x, labels) in enumerate(relevant_train_loader):
            ## Check for nans
            if torch.sum(torch.isnan(x)) > 0:
                print('i_batch', i_batch, 'x contains nan')
                torch.save(x, 'tmp/x_with_nan.npy')
                raise ValueError

            # (add offset because enumerate will start from 0 even if the loader
            # starts from the middle of the dataset -- only relevant if using chk_loader)
            i_batch = i_batch + i_batch_offset 

            if i_batch % 5000 == 0 and i_batch != 0:
                torch.save({'tmp_checkpoint':model.state_dict(),
                            'epoch':epoch,
                            'batch_nr':i_batch},
                           'Models/chk_{}.tar'.format(epoch_dict['Save']))
                print('saved checkpoint at epoch {} batch {}'.format(epoch, i_batch), flush=True)
                print('time stamp:', datetime.datetime.now(), flush=True)

            ## It is possible depending on the dataset class that we will not
            ## want to use this default normalization, but what we've been doing
            ## as a default is to normalize based on the brightest single pixel
            if to_normalize:
                for i in range(len(x)):
                    x[i] = x[i] / torch.max(x[i]) # normalize inputs

            x = x.to(device) # .double()
            optimizer.zero_grad()
            torch.cuda.empty_cache() # this may possibly help with memory issues
            outputs = model(x)

            loss = criterion(outputs, labels.to(device))

            loss.backward()

            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            train_size += x.shape[0]
            if i_batch % 50 == 0: # used to be 100, but we're using such a small dataset this time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i_batch * len(x), len(train_loader.dataset),
                    100. * i_batch / len(train_loader),
                    loss.item() ))

            if loss.item() != loss.item():
                print('Loss is nan')
                raise ValueError

            del x, loss # trying to save memory with this

        # end batch
        torch.cuda.empty_cache() # desperate attempt to save memory

        epoch_dict['TrainingLoss'].append(train_loss / train_size)

        val_loss = 0
        val_samples = 0
        model.eval()

        with torch.no_grad(): # disable gradient calculation while validating (otherwise it's too slow)
            for i_batch, (x, labels) in enumerate(val_loader):
                if i_batch % 10 == 0:
                    print('    Validation Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, i_batch * len(x), len(val_loader.dataset),
                        100. * i_batch / len(val_loader)
                        ))

                if to_normalize:
                    for i in range(len(x)):
                        x[i] = x[i] / torch.max(x[i])
                
                x = x.to(device) # .double()
                torch.cuda.empty_cache() # hopefully this will solve the memory issues
                outputs = model(x) # seems the output gets all wrong when we get to eval mode

                loss = criterion(outputs, labels.to(device))
                
                val_loss += loss.item() * x.shape[0]
                val_samples += x.shape[0]

        val_loss_per_sample = val_loss / val_samples
        epoch_dict['ValidationLoss'].append(val_loss_per_sample)
        print('Validation loss = {0:.4f}'.format(val_loss_per_sample))

        model_state_dicts.append(deepcopy(model.state_dict()))
        if len(model_state_dicts) > 1:
            # delete non-optimal weights to save space!!
            # (note that we never delete the most recent weights)
            optimal_epoch = np.argmin(epoch_dict['ValidationLoss'])

            for i in range(len(model_state_dicts) - 1):
                if i != optimal_epoch:
                    model_state_dicts[i] = None


        if val_loss_per_sample < min_loss:
            min_loss = val_loss_per_sample
            stopping_patience = stopping_patience_START
        else:
            stopping_patience -= 1

        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model_state_dicts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'TrainingLoss': epoch_dict['TrainingLoss'],
                    'ValidationLoss': epoch_dict['ValidationLoss']},
                   os.path.join('Models', epoch_dict['Save'] + '.tar'))

        Scheduler.step(val_loss_per_sample)

        if stopping_patience == 0:
            print('Model has not improved in {0} epochs...stopping early'.format(stopping_patience_START))
            break

    # end epoch
