import wandb
import logging
import json
import os
from credentials import wandb_setup
import pandas as pd

import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import glob
import random
import string

from bitbybit.viz_utils import *
from torch import nn
from torch.optim import lr_scheduler

from fvcore.nn import FlopCountAnalysis
import pytorch_warmup as warmup
import sys
import inspect 
from typing import Dict, Any, Callable


def subsample_dataset(dataset, frac):
    if frac < 1.0:
        num_samples = len(dataset)
        num_subset = int(num_samples * frac)
        indices = np.random.choice(num_samples, num_subset, replace=False)
        sampled = torch.utils.data.Subset(dataset, indices)
        return sampled
    return dataset

def loss_fn(output, target, name="MSE"):
    default_losses = {
        "CE": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss()
    }
    # If using MSELoss, convert y to one-hot encoding to match output shape
    if name == "MSE" and output.shape  != target.shape :
        num_classes = output.shape[1]
        target = torch.nn.functional.one_hot(target.to(torch.long), num_classes=num_classes).float()
    
    return default_losses[name](output, target)



def setup_optimizer(config, network):
    """ Initializing optimizer"""
    default_optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop
    }
    opt_name = config['name']
    optimizer_cls = default_optimizers[opt_name]
    
    if opt_name in ['sgd', 'rmsprop']:
        # SGD and RMSprop accept 'momentum'
        optimizer = optimizer_cls(network.parameters(), 
                                  lr = config['lr'], 
                                  weight_decay=config['weight_decay'], 
                                  momentum=config['momentum'])
    
    elif opt_name in ['adam', 'adamw']:
        # Adam and AdamW do NOT accept 'momentum'
        optimizer = optimizer_cls(network.parameters(), 
                                  lr = config['lr'], 
                                  weight_decay=config['weight_decay'])
                                  # Note: 'betas', 'eps', etc., could also be
                                  # added here from the config if needed
    
    else:
        raise ValueError(f"Optimizer {opt_name} logic is not defined.")
    
    
    return optimizer

def setup_scheduler(config, optimizer):
    """ Initializing scheduler 
        - restart flag: whether reinitializing for a new task
    """
    # (This function was already correct and followed your desired pattern)
    if config['name']=="step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=int(config['step_scheduler_decay']), 
                                                    gamma=config['scheduler_step'])
    elif config['name'] == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=config['steps_per_task'], 
                                                               eta_min=1e-5)
    elif config['name'] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=config["patience"],        
                threshold=config["eps"],      
                threshold_mode='rel',
                min_lr=1e-5,
                verbose=True
            )
    else: 
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1) # Changed -1 to 1
    
    if config.get("warmup_on", False) and warmup:
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config['warmup'])   
        return scheduler, warmup_scheduler
        
    return scheduler, None

def get_flat_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p.view(-1)))
    return torch.cat(grads)

def get_flat_params(model):
    """Gets all model parameters as a single flat vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """Sets model parameters from a flat vector."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        # Get the slice
        param_data = flat_params[offset:offset + numel].view(param.size())
        # Copy data
        param.data.copy_(param_data)
        offset += numel

class MyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_size(obj):
    """Rough memory in bytes, recursively."""
    if isinstance(obj, dict):
        return sum(get_size(k) + get_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        return sum(get_size(x) for x in obj)
    else:
        return sys.getsizeof(obj)

def compute_flops(model, dataloader, device, max_batches=10):
    model.eval()
    total_flops = 0

    with torch.no_grad():
        for i, (x, *_) in enumerate(dataloader):
            x = x.to(device)
            if i >= max_batches:
                break
            x = x.to(device)
            flops = FlopCountAnalysis(model, x)
            total_flops += flops.total()
    return total_flops


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def MSE_residuals(outputs, targets):
    num_classes = outputs.shape[1]
    target_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    return outputs - target_onehot.to(outputs.device)

def CE_residuals(outputs, targets):
    num_classes = outputs.shape[1]
    target_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    return nn.functional.softmax(outputs, dim=1) - target_onehot.to(outputs.device)

def random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def has_batch_norm(net):
    for n, p in net.named_modules():
        if isinstance(p, torch.nn.BatchNorm2d): return True
    return False

def get_norm_distance(m1, m2):
    return torch.linalg.norm(m1-m2, 2).item()

def get_cosine_similarity(m1, m2):
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cosine(m1, m2)

def get_params(net):
    # Initialize an empty list to store the parameters
    params_list = []

    # Iterate over all the parameters of the network and append them to the list
    for param in net.parameters():
        params_list.append(param.data.view(-1))

    # Concatenate the list to a single tensor
    params_vector = torch.cat(params_list)

    return params_vector.detach().clone()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
 
def seed_worker(worker_id):
    """
    Worker init function for DataLoader to ensure workers are seeded correctly.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_model_checkpoint(model, optimizer, path, **kwargs):
    """Save a model checkpoint to the specified path."""
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    for k,v in kwargs.items():
        checkpoint[k] = v

    torch.save(checkpoint, path)

def load_model_checkpoint(model, optimizer, path, keys_to_load=[]):
    """Load a model checkpoint from the specified path."""
    # Load checkpoint
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    info = {}
    for k in keys_to_load:
        if k in checkpoint:
            info[k] = checkpoint[k] 
    config = checkpoint.get('config', None)
    return info, config



def set_seed(seed):
    """ Function to set the random seed for reproducibility across libraries. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

