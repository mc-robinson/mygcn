'''
Generic code for training.
Much of this code is adapted from the excellent tutorial by Jeremy Howard: 
https://pytorch.org/tutorials/beginner/nn_tutorial.html
'''

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import dgl

import numpy as np
import pandas as pd

# imports for typing
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, \
                      Iterable, List, Mapping, NewType, Optional, Sequence, \
                      Tuple, TypeVar, Union                   
from types import SimpleNamespace

def train_epoch(model:nn.Module, train_dl:DataLoader,
                opt:optim.Optimizer, loss_func:Callable) -> float:
    "Trains given `model` for one epoch given the `opt` and `loss_func`"
    model.train()
    train_loss=0
    for iteration, (bg, label) in enumerate(train_dl):

        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.set_n_initializer(dgl.init.zero_initializer)

        # need to handle special case for regression with nn.MSELoss()
        if model(bg).shape[1] == 1:
            label = label.reshape(-1,1)
            # print(label)

        loss = loss_func(model(bg), label)
        train_loss += loss.detach().item()
        
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # train_loss /= (iteration + 1) # per batch loss
    train_loss /= len(train_dl.dataset) # per sample loss
    return train_loss

def validate(model:nn.Module, valid_dl:DataLoader, loss_func:Callable) -> float:
    "Gives valid loss using the given `model` and `loss_func`"
    model.eval()
    # don't track gradients
    with torch.no_grad():
        valid_loss = 0
        for (bg, label) in valid_dl:

            ### FIX THESE SHAPES TOO ###

            bg.set_e_initializer(dgl.init.zero_initializer)
            bg.set_n_initializer(dgl.init.zero_initializer)

            # need to handle special case for regression with nn.MSELoss()
            if model(bg).shape[1] == 1:
                label = label.reshape(-1,1)

            loss = loss_func(model(bg), label)
            valid_loss += loss.detach().item()

        # valid_loss = sum(loss_func(model(bg), label) for bg, label in valid_dl)

    # return valid_loss/len(valid_dl) # this gives per batch loss
    return valid_loss/len(valid_dl.dataset) # this gives per sample loss

def fit(model:nn.Module, train_dl:DataLoader, valid_dl:DataLoader,
        loss_func:Callable, opt:optim.Optimizer, n_epochs:int,
        report_valid_loss:bool=False) -> None:
    "Fit the given `model` using the given `opt` and `loss_func` for `n_epochs`"
    
    epoch_train_losses = []
    epoch_valid_losses = []

    for epoch in range(n_epochs):

        # train
        train_loss = train_epoch(model, train_dl, opt, loss_func)
        epoch_train_losses.append(train_loss)

        # evaluate on validation set
        if report_valid_loss:
            valid_loss = validate(model, valid_dl, loss_func)
            epoch_valid_losses.append(valid_loss)
        else:
            valid_loss = 'N/A'

        if n_epochs > 10:
            if epoch % (n_epochs//10) == 0:
                print(
                    f'Epoch {epoch}, train loss {train_loss:.4f}',
                    f'valid loss {valid_loss}'
                )  
        else:
            print(
                    f'Epoch {epoch}, train loss {train_loss:.4f}',
                    f'valid loss {valid_loss}'
            )  



