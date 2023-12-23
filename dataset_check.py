import torch
import os
from tqdm import tqdm
import argparse
from utils import *
from rnn_model import *
from datasets import MNIST, Kink, Cifar10, Slab, SlabLinear, SlabNonlinear4
from fastargs import Section, Param, get_current_config
from fastargs.validation import OneOf
from fastargs.decorators import param, section
from optimizer import NelderMead, PatternSearch
from train_distributed import get_model_name
from fractions import Fraction
import time
import datetime
import json
import sys
#import matplotlib.pyplot as plt
from sql import *

def get_dataset(name, device, num_samples, seed, num_classes=None, noise=None, margin=0.25):
    if name == "mnist":
        name = MNIST(batch_size=num_samples, threads=1, aug='none', train_count=num_samples, num_classes=num_classes,
                     seed=seed)
        train_data, train_labels = next(iter(name.train))
        test_data, test_labels = next(iter(name.test))
        train_data, train_labels, test_data, test_labels = train_data.to(device), train_labels.to(device), test_data.to(
            device), test_labels.to(device)
        test_all_data, test_all_labels = name.test_all_data, name.test_all_labels
    elif name == "cifar10":
        name = Cifar10(batch_size=num_samples, threads=1, aug='none', train_count=num_samples, num_classes=num_classes,
                       seed=seed)
        train_data, train_labels = next(iter(name.train))
        test_data, test_labels = next(iter(name.test))
        train_data, train_labels, test_data, test_labels = train_data.to(device), train_labels.to(device), test_data.to(
            device), test_labels.to(device)
        test_all_data, test_all_labels = name.test_all_data, name.test_all_labels
    elif name == "kink":
        train_data = torch.tensor(
            Kink(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().to(device)
        train_labels = torch.tensor(
            Kink(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().to(device)
        test_data = torch.tensor(
            Kink(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().to(device)
        test_labels = torch.tensor(
            Kink(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().to(device)
        test_all_data, test_all_labels = test_data, test_labels
    elif name == "slab":
        train_data = torch.tensor(
            Slab(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().to(device)
        train_labels = torch.tensor(
            Slab(train=True, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().to(device)
        test_data = torch.tensor(
            Slab(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).data).float().to(device)
        test_labels = torch.tensor(
            Slab(train=False, samples=num_samples, seed=seed, noise=noise, margin=margin).labels).long().to(device)
        test_all_data, test_all_labels = test_data, test_labels
    elif name == "slab_nonlinear_4":
        dataset = SlabNonlinear4(samples=num_samples)
        train_data = torch.tensor(dataset.data).float().to(device)
        train_labels = torch.tensor(dataset.labels).long().to(device)
        test_data = train_data
        test_labels = train_labels
        test_all_data, test_all_labels = train_data, train_labels
    elif name == "slab_linear":
        dataset = SlabLinear(samples=num_samples)
        train_data = torch.tensor(dataset.data).float().to(device)
        train_labels = torch.tensor(dataset.labels).long().to(device)
        test_data = train_data
        test_labels = train_labels
        test_all_data, test_all_labels = train_data, train_labels
    return train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Cuda:0 is always the first visible GPU
cur_num_samples = 2
for data_seed in range(100,111):
    train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels = get_dataset(name="mnist", num_classes=2, num_samples=cur_num_samples, device=device, seed=data_seed)  # This operation takes time
