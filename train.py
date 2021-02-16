import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import model.model as model
from parse_config import ConfigParser
from trainer import Trainer

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, config):
    # setup DDP
    setup(rank, world_size)

    # data loader
    data_loader = config.init_data_loader(rank)

    # parameters used with other objects
    dims = data_loader.dims
    no_samples = data_loader.no_samples

    # model
    similarity_metric = config.init_obj('model', model)

    # losses
    losses = config.init_losses()

    # transformation and registration modules
    transformation_module, registration_module = config.init_transformation_and_registration_modules(dims)

    # metrics
    metrics = config.init_metrics(no_samples)

    # training
    trainer = Trainer(config, data_loader, similarity_metric, losses, transformation_module, registration_module, metrics, rank)
    trainer.train()

    # DDP
    cleanup()


if __name__ == '__main__':
    # parse arguments
    args = argparse.ArgumentParser(description='LearnSim')

    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    # config
    config = ConfigParser.from_args(args, [])

    # run the training script
    n = config['no_GPUs']
    mp.spawn(train, args=(n, config), nprocs=n, join=True)
