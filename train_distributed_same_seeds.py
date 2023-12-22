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

Section("dataset", "Dataset parameters").params(
    name=Param(str, OneOf(("mnist", "kink", "cifar10", "slab", "slab_nonlinear_3", "slab_nonlinear_4", "slab_linear")),
               default="kink"),
)
Section("dataset.kink", "Dataset parameters for kink").enable_if(
    lambda cfg: cfg['dataset.name'] == 'kink'
).params(
    margin=Param(float),
    noise=Param(float)
)
Section("dataset.mnistcifar", "Dataset parameters for mnist/cifar").params(
    num_classes=Param(int)
)
Section("model", "Model architecture parameters").params(
    arch=Param(str, OneOf(("rnn", "mlp", "lenet")), default="rnn"),
    model_count_times_batch_size=Param(int, default=20000 * 16),
    init=Param(str, OneOf(("uniform", "regular", "uniform2", "uniform5", "sphere100", "sphere200", "rnn")),
               default="uniform")
)
Section("model.rnn", "Model architecture parameters").params(
    N=Param(int),
    H_in=Param(int),
    H_out=Param(int),
    r_min=Param(float),
    r_max=Param(float),
    max_phase=Param(float),
    embedding_size=Param(int),
    complex=Param(bool),
    transition_matrix_parametrization=Param(str),
    gamma_normalization=Param(bool),
    official_glorot_init=Param(bool),
    linear_recurrent=Param(bool),
    embeddings_type=Param(str),
    guess_encoder_layer_params=Param(bool),
    enable_forward_normalize=Param(bool),
    model_count=Param(int),
    num_of_rnn_layers=Param(int),
    framework=Param(str),
    scale=Param(float),
    efficient_rnn_forward_pass=Param(bool)
)
Section("model.lenet", "Model architecture parameters").params(
    width=Param(float),
    feature_dim=Param(float)
)
Section("model.mlp", "Model architecture parameters").enable_if(lambda cfg: cfg['model.arch'] == 'mlp').params(
    hidden_units=Param(int),
    layers=Param(int)
)
Section("optimizer").params(
    name=Param(str, OneOf(["SGD", "SGDPoison", "Adam", "RMSProp", "guess", "GD"]), default='guess'),
    es_u=Param(float, default=float('inf')),
    es_l=Param(float, default=-float('inf')),
    grad_norm_thres=Param(float, desc='only accept models with gradient norm smaller than specified'),
    lr=Param(float, desc='learning rate'),
    momentum=Param(float, desc='momentum', default=0),
    epochs=Param(int, desc='number of epochs to optimize  for'),
    es_acc=Param(float, desc='stop the training when average training acc reaches this level'),
    batch_size=Param(int, desc='number of epochs ot optimize for', default=3),
    scheduler=Param(int, desc='whether to use a scheduler', default=False),
    poison_factor=Param(float, desc='level of poisoning applied'),
    print_intermediate_test_acc=Param(int, default=0, desc='whether to print intermediate test acc')
)
# TODO: write logic for excluded_cells
Section("distributed").params(
    loss_thres=Param(str, default="0.3,0.4,0.5"),
    num_samples=Param(str, default="2,4,8"),
    new_run=Param(bool, default=False),
    tmux_id=Param(int, default=-1),
    gpu_id=Param(str, default="0"),
    excluded_cells=Param(str, default="", desc='ex: 32_(0.3, 0.35)/16_(0.3, 0.35)'),
    target_model_count_subrun=Param(int, default=1),
    training_seed=Param(int, default=None,desc='If there is no training seed, then the training seed increment with every new runs'),
    data_seed=Param(int, default=None,desc='If there is no data seed, then the training seed increment with every new runs, otherwise, it is fix'))

Section("output", "arguments associated with output").params(
    target_model_count=Param(int, default=1),
    folder=Param(str, default='test_distributed')
)


@section('dataset')
@param('name')
@param('mnistcifar.num_classes')
@param('kink.noise')
@param('kink.margin')
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


def get_cur_batch_size_and_model_count(config, cur_num_samples):
    if config['optimizer.name'] in ["SGD"]:
        cur_batch_size = min(cur_num_samples // 2, config['optimizer.batch_size'])
        cur_model_count = config['model.model_count_times_batch_size'] // cur_batch_size
    elif config['optimizer.name'] in ["SGDPoison"]:
        cur_batch_size = config['optimizer.batch_size']
        cur_model_count = config['model.model_count_times_batch_size'] // cur_batch_size
    elif config['optimizer.name'] == "guess":
        cur_batch_size = None
        cur_model_count = config['model.model_count_times_batch_size'] // cur_num_samples
    else:
        cur_batch_size = None
        cur_model_count = config['model.model_count_times_batch_size'] // cur_num_samples
    return cur_batch_size, cur_model_count


@section('model')
@param('arch')
def get_model(arch, config, model_count, device):
    if arch == "mlp":
        kwargs = {"input_dim": 2, "output_dim": 2, "layers": config['model.mlp.layers'],
                  "hidden_units": config['model.mlp.hidden_units'], "model_count": model_count}
        model = MLPModels(**kwargs, device=device)

    elif arch == "lenet":
        kwargs = {"output_dim": config['dataset.mnistcifar.num_classes'], "width_factor": config['model.lenet.width'],
                  "model_count": model_count,
                  "dataset": config['dataset.name'], "feature_dim": config['model.lenet.feature_dim']}
        model = LeNetModels(**kwargs).to(device)

    elif arch == "linear":
        input_dim = 28 * 28 if config['dataset.name'] == "mnist" else 32 * 32 * 3
        kwargs = {"input_dim": input_dim, "output_dim": config['dataset.mnistcifar.num_classes'],
                  "model_count": model_count}
        model = LinearModels(**kwargs, device=device).to(device)

    elif arch == "rnn":
        kwargs = {"N": config['model.rnn.N'], "H_in": config['model.rnn.H_in'], "H_out": config['model.rnn.H_out'],
                  "output_dim": config['dataset.mnistcifar.num_classes'],"r_min": config['model.rnn.r_min'], "r_max": config['model.rnn.r_max'], "max_phase": config['model.rnn.max_phase'],
                  "embedding_size": config['model.rnn.embedding_size'],
                  "complex": config['model.rnn.complex'], "transition_matrix_parametrization": config['model.rnn.transition_matrix_parametrization'], "gamma_normalization": config['model.rnn.gamma_normalization'],
                  "official_glorot_init": config['model.rnn.official_glorot_init'], "linear_recurrent": config['model.rnn.linear_recurrent'],
                  "embeddings_type": config['model.rnn.embeddings_type'],
                  "enable_forward_normalize": config['model.rnn.enable_forward_normalize'],
                  "num_of_rnn_layers": config['model.rnn.num_of_rnn_layers'], "framework": config['model.rnn.framework'], "device": device, "model_count": model_count,
                  "scale": config['model.rnn.scale'], "efficient_rnn_forward_pass": config['model.rnn.efficient_rnn_forward_pass']}
        model = RNNModels(**kwargs)

    return model, kwargs


def reinitialize_modle(model, config):
    # TODO: update this section to take in variable mult + simplifying the way that initializaiton is selected
    if config['model.init'] == "uniform":
        model.reinitialize()
    elif config['model.init'] == "uniform2":
        model.reinitialize(mult=2)
    elif config['model.init'] == "uniform5":
        model.reinitialize(mult=5)
    elif config['model.init'] == "sphere100":
        model.reinitialize_sphere(mult=100)
    elif config['model.init'] == "sphere200":
        model.reinitialize_sphere(mult=200)
    elif config['model.init'] == "regular":
        model.reset_parameters()
    elif config['model.init'] == "rnn":
        print(f"DEBUG:reinitialize rnn model. scale={model.scale}")
        model.reinitialize_params()


@section('optimizer')
@param('name')
@param('lr')
@param('momentum')
@param('scheduler')
def get_optimizer_and_scheduler(name, model, scheduler=False, lr=None, momentum=0):
    if name in ["SGD", "GD", "RMSProp", "Adam", "SGDPoison"]:
        if name == "RMSProp":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif name == "SGDPoisons":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        if scheduler == False:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999999], gamma=0.2)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2000, 3000], gamma=0.5)
    elif name == "guess":
        optimizer = None
        scheduler = None
    elif name == "NelderMead":
        optimizer = NelderMead(model.parameters(), alpha=1, gamma=2, rho=0.5, sigma=0.5)
        scheduler = None
    elif name == "PatternSearch":
        optimizer = PatternSearch(model.parameters())
        scheduler = None
    else:
        optimizer = None
        scheduler = None
    return optimizer, scheduler


@section('optimizer')
@param('epochs')
@param('batch_size')
@param('es_u')
@param('es_acc')
@param('print_intermediate_test_acc')
def train_sgd(
        train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size, epochs,
        es_u, es_acc=1, print_intermediate_test_acc=0):
    for epoch in range(epochs):
        idx_list = torch.randperm(len(train_data))
        for st_idx in range(0, len(train_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(train_data))]
            train_loss, train_acc = calculate_loss_acc(train_data[idx], train_labels[idx], model, loss_func)

            if es_u != float('inf'):
                with torch.no_grad():
                    train_loss_all, train_acc_all = calculate_loss_acc(train_data, train_labels,
                                                                       model.forward_normalize,
                                                                       loss_func)
                train_loss = torch.where((train_loss_all > es_u) | (train_acc_all < 1), train_loss,
                                         torch.zeros_like(train_loss))

                train_loss = train_loss[~train_loss.isnan()]
            optimizer.zero_grad()
            train_loss.sum().backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
                if len(train_loss[train_acc == 1]) > 0:
                    print(
                        f"train loss range: {train_loss[train_acc == 1].max().item()} {train_loss[train_acc == 1].min().item()}")
                train_loss = train_loss[~train_loss.isnan()]
                test_loss = test_loss[~test_loss.isnan()]
                print(
                    f"epoch {epoch} -  train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}")
                print(
                    f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
                if print_intermediate_test_acc:
                    _, test_acc = calculate_loss_acc(test_all_data.to(device), test_all_labels.to(device), model,loss_func, batch_size=batch_size)
                    print("test acc (all):", test_acc)
                if train_acc.mean() >= es_acc:
                    break
    optimizer.zero_grad()


@section('optimizer')
@param('epochs')
@param('batch_size')
@param('es_u')
@param('es_acc')
@param('poison_factor')
def train_sgd_poison(
        train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size, epochs,
        es_u, es_acc=1, poison_factor=None,
        test_all_data=None, test_all_labels=None):
    test_all_data, test_all_labels = test_all_data.to(device), test_all_labels.to(device)
    poison_test_labels = torch.tensor([1, 0], device=test_all_labels.device)[test_all_labels]
    repeats = 10
    poison_data = torch.cat([train_data.repeat_interleave(repeats, dim=0), test_all_data], dim=0)
    poison_labels = torch.cat([train_labels.repeat_interleave(repeats), poison_test_labels], dim=0)
    for epoch in range(epochs):
        idx_list = torch.randperm(len(poison_data))
        for st_idx in range(0, len(poison_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(poison_data))]
            train_loss, train_acc = calculate_loss_acc(poison_data[idx], poison_labels[idx], model, loss_func)

            optimizer.zero_grad()
            train_loss.sum().backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_all_data, test_all_labels, model.forward_normalize, loss_func)
                poison_train_loss, poison_train_acc = calculate_loss_acc(poison_data, poison_labels, model.forward_normalize, loss_func)
                if len(train_loss[train_acc == 1]) > 0:
                    print(f"train loss range: {train_loss[train_acc == 1].max().item()} {train_loss[train_acc == 1].min().item()}")
                train_loss = train_loss[~train_loss.isnan()]
                test_loss = test_loss[~test_loss.isnan()]
                print(
                    f"epoch {epoch} -  train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}")
                print(
                    f"epoch {epoch} - test acc all (max, min): {test_acc.max().item(): 0.2f}, {test_acc.min().item(): 0.2f}")

                print(
                    f"epoch {epoch} -  poison_train_acc: {poison_train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {poison_train_loss.mean().cpu().detach().item(): 0.4f}")

                if poison_train_acc.mean() >= es_acc:
                    break
    optimizer.zero_grad()


@section('optimizer')
@param('epochs')
@param('es_acc')
@param('es_u')
def train_gd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, epochs, es_u,
             es_acc=2):
    for epoch in range(epochs):
        train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model, loss_func)
        if es_u != float('inf'):
            train_loss = torch.where((train_loss > es_u) | (train_acc < 1),
                                     train_loss, torch.zeros_like(train_loss))
        optimizer.zero_grad()
        train_loss.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
                print(
                    f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
                print(
                    f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc.mean() >= es_acc:
                break
    optimizer.zero_grad()


@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_nm(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        optimizer.step(lambda: calculate_loss_acc(train_data, train_labels, model, loss_func)[0][0])
        if epoch % (epochs // 100 + 1) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc >= es_acc:
                break


@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_ps(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        optimizer.step(lambda: calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)[0][0])
        if epoch % (epochs // 100) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc >= es_acc:
                break


@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_ps_fast(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs, es_acc=2):
    for epoch in range(epochs):
        model.pattern_search(train_data, train_labels, loss_func)
        if epoch % (epochs // 100) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(
                f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(
                f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc.mean() >= es_acc:
                break
    return model.get_model_subsets([0]).to(train_data.device)


@section('optimizer')
@param('epochs')
@param('es_acc')
@torch.no_grad()
def train_greedy_random(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, epochs,
                        es_acc=2):
    for epoch in range(epochs):
        model.greedy_random(train_data, train_labels, loss_func)
        if epoch % (epochs // 300) == 0:
            train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
            test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
            print(
                f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
            print(
                f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
            if train_acc.mean() >= es_acc:
                break
    return model.get_model_subsets([0]).to(train_data.device)


@section('optimizer')
@param('name')
def train(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, name,
          batch_size=None, es_u=None, test_all_data=None, test_all_labels=None):
    if name in ["SGD", "RMSProp", "Adam"]:
        train_sgd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler,
                  batch_size=batch_size, es_u=es_u)
    elif name in ["SGDPoison"]:
        train_sgd_poison(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler,
                         batch_size=batch_size, es_u=es_u, test_all_data=test_all_data, test_all_labels=test_all_labels)
    elif name == "GD":
        train_gd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, es_u=es_u)
    elif name == "NelderMead":
        train_nm(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    elif name == "PatternSearch":
        train_ps(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    elif name == "PatternSearchFast":
        model = train_ps_fast(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    elif name == "GreedyRandom":
        model = train_greedy_random(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer)
    else:
        pass
    return model


def convert_config_to_dict(config):
    config_dict = {}
    for path in config.entries.keys():
        try:
            value = config[path]
            if value is not None:
                config_dict['.'.join(path)] = config[path]
        except:
            pass
    return config_dict


def build_results_directory_path(config, training_seed, data_seed, cur_num_samples, results_directory_name, model_name):
    os.makedirs(os.path.join(config['output.folder'], results_directory_name), exist_ok=True)
    output_path = f"{config['output.folder']}/{results_directory_name}/"
    if config['model.arch'] == 'rnn':
        #forward_normalized_message = "using_forward_normalize" if config['model.rnn.enable_forward_normalize'] else "without_forward_normalize"
        #output_path += f"scale_{config['model.rnn.scale']}_{config['model.rnn.embeddings_type']}_embeddings_size_{config['model.rnn.embedding_size']}_{forward_normalized_message}/"
        output_path += model_name + "_"
        if results_directory_name != "models":
            return output_path

    output_path += f"{config['dataset.name']}_s{cur_num_samples}_"

    if config['model.arch'] == "lenet":
        output_path += f"lenet_w{config['model.lenet.width']}_"
    elif config['model.arch'] == 'linear':
        output_path += f"linear_"
    elif config['model.arch'] == 'mlp':
        output_path += f"mlp_h{config['model.mlp.hidden_units']}" \
                       f"l{config['model.mlp.layers']}_"

    output_path += f"opt{config['optimizer.name']}_"
    if config['optimizer.grad_norm_thres']:
        output_path += '_gnorm'
    output_path += f"dseed{data_seed}_tseed{training_seed}"
    return output_path


def print_and_save_loss_histogram(config, training_seed, loss_bin, data_seed, cur_num_samples, train_losses, model_name, scale=None):
    output_path = build_results_directory_path(config, training_seed, data_seed, cur_num_samples, "loss_histograms",model_name) + ".png"
    # bins = [i / 20 for i in range(21)] # lenet model
    bins = [i / 10 for i in range(0, 100, 1)]  # rnn model
    plt.hist(train_losses, bins=bins)
    plt.xlabel('Train Loss')
    plt.ylabel('Count')
    plt.title(f"train_loss histogram of num_samples:{cur_num_samples} calculated over {len(train_losses)} losses")
    plt.savefig(output_path)  # Save the plot to a file
    plt.show()


def print_model_details(config, model):
    if config['model.arch'] == 'rnn':
        embedding_size = "none" if model.embeddings_type == "none" else model.embedding_size
        print(f"############rnn experiment details############")
        print(f"N={model.N} H_in={model.H_in} H_out={model.H_out} scale={model.scale} embedding_size={embedding_size}")
        if model.transition_matrix_parametrization == "diag_stable_ring_init":
            print(f"r_min={model.r_min} r_max={model.r_max} max_phase={model.max_phase}")
        print(f"complex_model={model.complex} transition_matrix_parametrization={model.transition_matrix_parametrization} gamma_normalization={model.gamma_normalization} official_glorot_init={model.official_glorot_init} linear_recurrent={model.linear_recurrent} efficient_rnn_forward_pass={model.efficient_rnn_forward_pass} embeddings_type={model.embeddings_type} guess_encoder_layer_params={model.guess_encoder_layer_params} enable_forward_normalize={model.enable_forward_normalize}")
        print(f"num_of_rnn_layers={model.num_of_rnn_layers} framework={model.framework} device={model.device} model_count={model.model_count} target_model_count_subrun={config['distributed.target_model_count_subrun']} target_model_count={config['output.target_model_count']}")
        print(f"##############################################")


def get_models_norms(models):
    model_l2_norm = 0
    model_linf_norm = 0
    for para in models.parameters():
        model_l2_norm += (para ** 2).sum()
        para_abs = para.abs() if isinstance(para,torch.Tensor) else np.abs(para)
        model_linf_norm = max(para_abs.max(), model_linf_norm)

    model_l2_norm = model_l2_norm ** 0.5
    return model_linf_norm, model_l2_norm


def DEBUG_compare_models_and_print_result(model1, model2, models_indices_to_compare):
    models_are_identical = True
    if model1.embeddings_type != "none":
        print(f"printing encoder layer params:")
        for param_idx in range(len(model1.encoder_layer_params)):
            if not np.array_equal(model1.encoder_layer_params[param_idx][:, models_indices_to_compare, :, :],
                                  model2.encoder_layer_params[param_idx]):
                print(f"DEBUG_models_identical: models differ in encoder_layer_param{param_idx}")
                models_are_identical = False

    for layer_idx in range(len(model1.rnn_layers_params)):
        for param_idx in range(len(model1.rnn_layers_params[layer_idx])):
            if not np.array_equal(model1.rnn_layers_params[layer_idx][param_idx][:, models_indices_to_compare, :, :],
                                  model2.rnn_layers_params[layer_idx][param_idx]):
                print(f"DEBUG_models_identical: models differ in layer{layer_idx}_param{param_idx}")
                models_are_identical = False

    if models_are_identical:
        print(f"$$$ DEBUG_models_identical: models are identical $$$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the config needs to change to a vector of sample size (potential model counts as well) x loss bins??

    # it will then devote compute to the specific sample size & loss bins combination with the smallest number of trained models

    # always select a random training seed & data seed

    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.summary()
    config_ns = config.get()
    loss_thres = [float(v) for v in config['distributed.loss_thres'].split(",")]
    loss_bins = [(low, up) for low, up in zip(loss_thres[:-1], loss_thres[1:])]
    num_samples = [int(v) for v in config['distributed.num_samples'].split(",")]
    if -1 in num_samples: # '-1' will be in the list in case we want to run our program with num_samples = [16] for example.
        num_samples.remove(-1)
    # create the table with counts
    os.makedirs(config['output.folder'], exist_ok=True)
    #Setting GPU id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['distributed.gpu_id']
    # the following lines overcome "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory" error
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    print(f"Setting gpu_id to {config['distributed.gpu_id']}")
    model_name = get_model_name(config) + f"_same_seeds_{config['output.target_model_count']}"
    db_path = os.path.join(config['output.folder'], "databases", f"{model_name}_stats.db")
    # if config['distributed.tmux_id'] == -1:
    #     db_path = os.path.join(config['output.folder'],"databases", f"{model_name}_stats.db")
    # else:
    #     db_path = os.path.join(config['output.folder'],"databases", f"{model_name}_stats_tmux{config['distributed.tmux_id']}.db")
    create_model_stats_table(db_path)
    if config['distributed.new_run']:
        delete_all_records_from_model_stats(db_path)
    # get_model_stats_summary(db_path)
    max_data_seed_attemps = 10
    model_count_thresh_for_changing_data_seed = 3000000
    # max_data_seed_attemps = 2
    # model_count_thresh_for_changing_data_seed = 300000
    print_experiment_details = True
    ## DEBUG - start ##
    collect_first_data_seed = True
    single_guess_and_check_time_information = ""
    single_test_losses_and_accuracies_calculation = ""
    fist_data_seed = 0
    DEBUG_str = "Experiment details:\n" + f"model_name={model_name} max_data_seed_attemps={max_data_seed_attemps} model_count_thresh_for_changing_data_seed={model_count_thresh_for_changing_data_seed}"
    ## DEBUG - end ##
    program_start_time = time.time()
    program_current_time_for_DEBUG = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Cuda:0 is always the first visible GPU
    print(f"running calculations on: {device}")
    print("Experiment details:")
    print(f"model_name={model_name} max_data_seed_attemps={max_data_seed_attemps} model_count_thresh_for_changing_data_seed={model_count_thresh_for_changing_data_seed}")
    for cur_num_samples in num_samples:
        cur_batch_size, cur_model_count = get_cur_batch_size_and_model_count(config, cur_num_samples)
        model, _ = get_model(config=config, model_count=cur_model_count, device=device)
        model.guess_encoder_layer_params = config['model.rnn.guess_encoder_layer_params'] # After 'get_model' call the encoder params are initialized and from now we will update them only according to guess_encoder_layer_params flag
        for cur_loss_bin in loss_bins:
            DEBUG_str += f"cur_num_samples={cur_num_samples} cur_loss_bin={cur_loss_bin} combination details: +\n"
            data_seed = 100
            training_seed = 200
            data_seed_is_not_good = False  # For some dataseeds, the G&C model takes a very long time and is never able to find a solution for certain bins
            while True:
                if data_seed_is_not_good:
                    data_seed += 1
                    num_of_data_seed_attempts += 1
                    tested_model_count = 0
                    print(f"New data_seed is:{data_seed}")
                else:
                    failed_combinations_dict = get_failed_combinations_dict(db_path)
                    if (cur_num_samples, cur_loss_bin[0], cur_loss_bin[1]) in failed_combinations_dict:
                        print(f"DEBUG: combination {cur_num_samples},[{cur_loss_bin[0]},{cur_loss_bin[1]}] is failed")
                        break
                    model_id, cur_smallest_model_count = get_next_config_same_seeds(db_path, cur_loss_bin, cur_num_samples, data_seed, training_seed)
                    print(f"DEBUG: model_id={model_id} cur_smallest_model_count={cur_smallest_model_count}")
                    if collect_first_data_seed:
                        fist_data_seed = data_seed
                        collect_first_data_seed = False
                        print(f"fist_data_seed={fist_data_seed}")
                    if cur_smallest_model_count >= config['output.target_model_count']:
                        print(f"Found models greater than target model count:{config['output.target_model_count']}, so ending the search")
                        break

                    get_model_stats_summary(db_path)
                    print(f"current config: num_samples:{cur_num_samples} cur_batch_size={cur_batch_size} loss_bin:{cur_loss_bin} model_count:{cur_model_count} training_seed={training_seed} data_seed={data_seed}")
                    es_l, es_u = cur_loss_bin
                    program_current_time_for_DEBUG = time.time()
                    train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels = get_dataset(num_samples=cur_num_samples, device=device, seed=data_seed)  # This operation takes time
                    print(f"DEBUG: Time to get_dataset:{time.time() - program_current_time_for_DEBUG} train_data.shape={train_data.shape}  test_data.shape={test_all_data.shape} train_labels[0]={train_labels[0]} train_labels[1]={train_labels[1]}")
                    #print(f" DEBUG: test_labels[0]={test_labels[0]} test_labels[1]={test_labels[1]}")
                    torch.manual_seed(training_seed)
                    print(f"DEBUG: model details:")
                    print(f" N={model.N} linear_recurrent={model.linear_recurrent} complex={model.complex} efficient_rnn_forward_pass={model.efficient_rnn_forward_pass} transition_matrix_parametrization={model.transition_matrix_parametrization} gamma_normalization={model.gamma_normalization}")
                    if model.transition_matrix_parametrization == "diag_stable_ring_init":
                        print(f"r_min={model.r_min} r_max={model.r_max} max_phase={model.max_phase}")
                    perfect_model_count = 0
                    perfect_model_weights = []
                    target_model_count_subrun = config['distributed.target_model_count_subrun']
                    tested_model_count = 0
                    total_tested_model_count = 0
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    prior_max = 0
                    num_of_data_seed_attempts = 1
                    cur_train_losses = []
                    start_time = time.time()
                    if print_experiment_details:
                        print_model_details(config, model)
                        print_experiment_details = False

                while perfect_model_count < target_model_count_subrun:
                    if tested_model_count >= model_count_thresh_for_changing_data_seed:
                        print(f"DEBUG: data_seed is not good. tested_model_count={tested_model_count} num_of_data_seed_attempts={num_of_data_seed_attempts}")
                        data_seed_is_not_good = True
                        break
                    program_current_time_for_DEBUG = time.time()
                    reinitialize_modle(model, config)
                    print(f"DEBUG: time it takes to reinitialize_modle={time.time()-program_current_time_for_DEBUG}")
                    optimizer, scheduler = get_optimizer_and_scheduler(model=model)
                    model_result = train(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer,scheduler, batch_size=cur_batch_size, es_u=es_u, test_all_data=test_all_data,test_all_labels=test_all_labels)
                    with torch.no_grad():
                        print(f"DEBUG: calculating train losses and accuracies. train_data.shape={train_data.shape} batch_size={cur_batch_size}")
                        program_current_time_for_DEBUG = time.time()
                        train_loss, train_acc, output = calculate_loss_acc(train_data, train_labels, model_result.forward_normalize, loss_func, batch_size=cur_batch_size)
                        print(f"Time to calculate train losses and accuracies over {tested_model_count + cur_model_count} models: {time.time() - program_current_time_for_DEBUG}")
                    cur_train_losses.extend(train_loss[train_acc == 1].cpu().tolist())
                    if train_acc.max() > prior_max:
                        # print("max train acc:", train_acc.max().detach().cpu().item())
                        prior_max = train_acc.max()
                    if single_guess_and_check_time_information == "":
                       single_guess_and_check_time_information = f"Time to calculate train losses and accuracies over {tested_model_count + cur_model_count} models: {time.time() - program_current_time_for_DEBUG}"
                       #print(single_guess_and_check_time_information)
                       DEBUG_str += single_guess_and_check_time_information + "\n"
                    # filtering models based on loss threshold (creating 0/1 mask)
                    perfect_model_idxs = ((es_l < train_loss) & (train_loss <= es_u) & (train_acc == 1.0))  # If the loss is in the correct level (inside the bin) and train accuracy reached 100%
                    perfect_model_count_cur = perfect_model_idxs.sum().detach().cpu().item()
                    perfect_model_count += perfect_model_count_cur
                    tested_model_count += cur_model_count
                    total_tested_model_count += cur_model_count
                    print(f"total_tested_model_count: {total_tested_model_count}  tested_model_count: {tested_model_count} perfect_model_count={perfect_model_count}")
                    #print(f"total_tested_model_count: {total_tested_model_count}  tested_model_count: {tested_model_count} perfect_model_count={perfect_model_count} train_loss={train_loss}")
                    if perfect_model_idxs.sum() > 0:
                        if perfect_model_count > target_model_count_subrun:
                            remain_count = perfect_model_count_cur - (perfect_model_count - target_model_count_subrun)  # How many models we need to take
                            tested_model_count -= ((perfect_model_count - target_model_count_subrun) / perfect_model_count_cur) * cur_model_count
                            #total_tested_model_count -= ((perfect_model_count - target_model_count_subrun) / perfect_model_count_cur) * cur_model_count
                            #print(f"DEBUG: perfect_model_idxs.nonzero().squeeze(1).SHAPE={perfect_model_idxs.nonzero().squeeze(1)[:remain_count].shape}")
                            perfect_model_weights.append(model_result.get_weights_by_idx(perfect_model_idxs.nonzero().squeeze(1)[:remain_count]))
                            print(f"######DEBUG: entered 'perfect_model_count > target_model_count_subrun' condition. perfect_model_count={perfect_model_count} remain_count={remain_count} tested_model_count={tested_model_count} len(perfect_model_weights)={len(perfect_model_weights)}######")
                        else:
                            perfect_model_weights.append(model_result.get_weights_by_idx(perfect_model_idxs.nonzero().squeeze(1)))

                if data_seed_is_not_good:
                    if num_of_data_seed_attempts < max_data_seed_attemps:
                        print(f"Trying a new data seed - only {perfect_model_count} out of {target_model_count_subrun} perfect models were found for set up: cur_num_samples:{cur_num_samples} cur_loss_bin:{cur_loss_bin} with data_seed:{data_seed}")
                    else:
                        data_seed_is_not_good = False
                        print(f"Setting status in DB to Falied - only {perfect_model_count} out of {target_model_count_subrun} perfect models were found for set up: cur_num_samples:{cur_num_samples} cur_loss_bin:{cur_loss_bin} for {max_data_seed_attemps} data_seed attemps with {model_count_thresh_for_changing_data_seed} different models in each attemp")
                        # train_time = time.time() - start_time
                        set_all_combination_records_status_to_FAILED(db_path, cur_num_samples, es_l,es_u)  # marking all records of this {cur_num_samples,cur_loss_bin} combination as 'FAILED'
                        #if plot_loss_histogram_dict[cur_num_samples]:  # we want to plot only one histogram per 'cur_num_samples'
                            #print_and_save_loss_histogram(config, training_seed, cur_loss_bin, data_seed, cur_num_samples, cur_train_losses, model_name)
                            #plot_loss_histogram_dict[cur_num_samples] = False

                else:
                    train_time = time.time() - start_time
                    print("=" * 50)

                    # test that the model weights can be reloaded
                    # concatenating the weights learned into a single model
                    good_models_state_dict = dict()
                    cat_dim = 1 if config['model.arch'] in ["mlp", "linear", "rnn"] else 0
                    for k in perfect_model_weights[0].keys():
                        if isinstance(perfect_model_weights[0][k],torch.Tensor) : # pytorch
                            good_models_state_dict[k] = torch.cat([weights_dictionary[k].cpu() for weights_dictionary in perfect_model_weights], dim=cat_dim)
                        else:                                                      # jax
                            good_models_state_dict[k] = np.concatenate([weights_dictionary[k] for weights_dictionary in perfect_model_weights], axis=cat_dim)

                    new_models, kwargs = get_model(arch=config['model.arch'], config=config, model_count=target_model_count_subrun, device="cpu")
                    new_models.load_state_dict(good_models_state_dict)
                    # show norm of the model
                    model_linf_norm, model_l2_norm = get_models_norms(new_models)
                    print(f"model l2 norm: {model_l2_norm}")
                    print(f"model linf norm: {model_linf_norm}")

                    with torch.no_grad():
                        #print(f"train_data.shape={train_data.shape} test_data.shape={test_all_data.shape}")
                        program_current_time_for_DEBUG = time.time()
                        loss, acc, _ = calculate_loss_acc(train_data.cpu(), train_labels.cpu(), new_models.forward, loss_func, batch_size=1)
                        test_loss, test_acc, _ = calculate_loss_acc(test_all_data.cpu(), test_all_labels.cpu(), new_models.forward, loss_func, batch_size=64)  # we don't use forward_normalize here because we are only using test accuracy, where normalization doesn't have any impact. ??? test_acc.shape=1 ???
                        print(f"verify that train acc is 100%: {acc.mean().item()}")
                        print(f"test_acc={test_acc.mean().item()}")
                        print(f"test_acc statistics over {target_model_count_subrun} perfect models that were found: mean:{test_acc.mean().item(): 0.3f} std:{test_acc.std().item(): 0.3f} (min_value,max_value):({test_acc.min().item(): 0.3f} , {test_acc.max().item(): 0.3f} )")
                        print(f"Time to calculate test losses and accuracies: {time.time()-program_current_time_for_DEBUG}")
                        if single_test_losses_and_accuracies_calculation == "":
                            single_test_losses_and_accuracies_calculation = f"Time to calculate test losses and accuracies: {time.time()-program_current_time_for_DEBUG}"
                            print(single_test_losses_and_accuracies_calculation)
                            DEBUG_str += single_test_losses_and_accuracies_calculation + "\n"

                    # saving the models
                    output_path = build_results_directory_path(config, training_seed, data_seed, cur_num_samples, "models", model_name)
                    print(f"Saving models at: {output_path}")
                    # run specific features that are saved only for evaluate_minimas.py,these are used for resumming models
                    saveconfig = convert_config_to_dict(config)
                    saveconfig['dataset.num_samples'] = cur_num_samples
                    saveconfig['training.seed'] = training_seed
                    saveconfig['dataset.seed'] = data_seed
                    saveconfig['training.es_l'], saveconfig['training.es_u'] = cur_loss_bin

                    # save the model
                    #torch.save({"kwargs": kwargs,"good_models_state_dict": good_models_state_dict,"config": saveconfig},output_path)
                    update_model_stats_table(
                        db_path,
                        model_id=model_id, data_seed=data_seed,
                        training_seed=training_seed,
                        num_training_samples=cur_num_samples,
                        loss_bin_l=es_l,
                        loss_bin_u=es_u,
                        test_acc=test_acc.mean().item(),
                        perfect_models_percentage=perfect_model_count/total_tested_model_count,
                        train_time=train_time,
                        perfect_model_count=target_model_count_subrun,
                        tested_model_count=total_tested_model_count,
                        save_path=output_path, status="COMPLETE")
                    print(f"Finished successful combination of cur_num_samples={cur_num_samples} cur_loss_bin={cur_loss_bin} with data_seed={data_seed} training_seed={training_seed} perfect_models_percentage={perfect_model_count}/{total_tested_model_count}={perfect_model_count/total_tested_model_count}")
                    DEBUG_str += f"Finished successful combination of cur_num_samples={cur_num_samples} cur_loss_bin={cur_loss_bin} with data_seed={data_seed} training_seed={training_seed} avg_test_accuracy={test_acc} perfect_models_percentage={Fraction(perfect_model_count,total_tested_model_count)}={perfect_model_count/total_tested_model_count} \n"
                    data_seed += 1
                    training_seed += 1

    # writing results to a file
    DEBUG_str += f"Experiment total time={time.time() - program_start_time} seconds = {str(datetime.timedelta(seconds=time.time() - program_start_time))} [hours:minutes:seconds]" + "\n"
    DEBUG_log_output_path = build_results_directory_path(config, None, None, cur_num_samples, "debug_logs", model_name) + "debug_log.txt"
    output_path = build_results_directory_path(config, None, None, cur_num_samples, "model_stats_summary", model_name) + "model_stats.txt"
    output_str = f"Experiment name: {model_name}" + "\n"
    output_str = single_guess_and_check_time_information + "\n"
    output_str += f"first_data_seed={fist_data_seed} N={config['model.rnn.N']}" + "\n"
    output_str += f"max_data_seed_attempts={max_data_seed_attemps}  model_count_thresh_for_changing_data_seed={model_count_thresh_for_changing_data_seed} \n"
    output_str += f"Experiment total time={time.time() - program_start_time} seconds = {str(datetime.timedelta(seconds=time.time() - program_start_time))} [hours:minutes:seconds]" + "\n"
    output_str += get_model_stats_summary(db_path, verbose=True, return_print=True) + "\n"
    output_str += get_stds_of_avg_acuuracies(db_path, return_print=True) + "\n"
    output_str += "DB entries that failed"
    output_str += get_model_FAILED_stats_summary(db_path, return_print=True)
    with open(DEBUG_log_output_path, 'w') as file:
        file.write(DEBUG_str)
    with open(output_path, 'w') as file:
        file.write(output_str)
    print(f"Experiment name={model_name} Experiment total time={time.time() - program_start_time} seconds = {str(datetime.timedelta(seconds=time.time() - program_start_time))} [hours:minutes:seconds]")

