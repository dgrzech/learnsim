from parse_config import ConfigParser
from trainer import Trainer

import argparse
import collections
import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.loss as model_loss
import utils.registration as registration
import utils.transformation as transformation

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


def main(config):
    # setup data_loader instance
    data_loader = config.init_obj('data_loader', module_data, save_dirs=config.save_dirs)

    # initialise the transformation model and registration modules
    dim_x = config['data_loader']['args']['dim_x']
    dim_y = config['data_loader']['args']['dim_y']
    dim_z = config['data_loader']['args']['dim_z']

    transformation_model = config.init_obj('transformation_model', transformation, dim_x, dim_y, dim_z)
    registration_module = config.init_obj('registration_module', registration)

    # losses
    data_loss = config.init_obj('data_loss', model_loss)
    reg_loss = config.init_obj('reg_loss', model_loss)
    entropy_loss = config.init_obj('entropy_loss', model_loss)

    # metrics
    metrics = ['data_term', 'reg_term', 'entropy_term', 'total_loss',
               'sample_data_term', 'sample_reg_term']

    # run training
    trainer = Trainer(data_loss, reg_loss, entropy_loss, transformation_model, registration_module, metrics,
                      config=config, data_loader=data_loader)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LearnSim')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
