import time
import os
import copy
import random
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter


def train_step(num_epochs, model, optimizer, scheduler, criterion, data, device):
    
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_acc = 0.0
    # Iterate over data.
    for i, batch in enumerate(data):
        samples, targets = batch ## samples Shape: batch_size*C*(T+N)*H*W, targets Shape: batch_size*2*(1+N)
        samples = samples.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            preds = model(samples).squeeze() ## Shape: batch_size*2*(1+N)
            
            loss = criterion(preds, targets.float())
            loss.backward()
            optimizer.step()
            correct = (targets.argmax(dim = 1).eq(preds.argmax(dim = 1))).sum()
        
        print(f'batch: {i}, loss: {loss}, correct : {correct}')
        running_loss += loss.item()
        running_acc += correct.item()

    print(f'Total loss : {running_loss}')
    print(f'Total Acc : {running_acc*100/(len(data)*data[1][1].numel())}')
    
    
def val_step(num_epochs, model, optimizer, scheduler, criterion, data, device):
    
    model.eval()  # Set model to training mode
    running_loss = 0.0
    running_acc = 0.0

    # Iterate over data.
    for i, batch in enumerate(data):
        samples, targets = batch
        samples = samples.to(device)

        with torch.set_grad_enabled(True):
            preds = model(samples).squeeze() ## Shape: batch_size*2*(1+N)
            loss = criterion(preds, targets.float())
            correct = (targets.argmax(dim = 1).eq(preds.argmax(dim = 1))).sum()
        
        print(f'batch: {i}, loss: {loss}, correct : {correct}')
        running_loss += loss.item()
        running_acc += correct.item()

    print(f'Total loss : {running_loss}')
    print(f'Total Acc : {running_acc*100/(len(data)*data[1][1].numel())}')
    
    
def train(num_epochs, model, optimizer, scheduler, criterion, data, device):
    
    epoch = 0
    
    while epoch < num_epochs:
        print(f'Running epoch : {epoch}')
        # Each epoch has a training and validation phase
        print('Train Step')
        train_step(num_epochs, model, optimizer, scheduler, criterion, data, device)
        print('Val Step')
        val_step(num_epochs, model, optimizer, scheduler, criterion, data, device)
        
        epoch += 1