import argparse

import model.model as model
import numpy as np
import torch

import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import Trainer

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def main(config):
    data_loader = config.init_obj('data_loader', module_data, save_dirs=config.save_dirs)  # data loader

    # parameters used with other objects
    dims = data_loader.dims
    no_samples = data_loader.no_samples

    encoder = config.init_obj('model', model)  # model
    losses = config.init_losses()  # losses
    transformation_module, registration_module = config.init_transformation_and_registration_modules(dims)  # transformation and registration modules

    metrics = config.init_metrics(no_samples)  # metrics

    # train the model
    trainer = Trainer(config, data_loader, encoder, losses, transformation_module, registration_module, metrics)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LearnSim')

    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, [])
    main(config)
