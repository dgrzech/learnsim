import argparse
import collections
import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.loss as model_loss
import model.metric as module_metric
import model.model as module_arch
import utils.registration as registration
import utils.transformation as transformation

from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_phi = config.init_obj('optimizer_phi', torch.optim, trainable_params)

    # initialise the loss
    data_loss = config.init_obj('data_loss', model_loss)
    kl_loss = config.init_obj('kl_loss', model_loss)

    # initialise the transformation model
    transformation_model = config.init_obj('transformation_model', transformation)
    # initialise the registration module
    registration_module = config.init_obj('registration_module', registration)

    # get function handle of metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # run training
    trainer = Trainer(model, data_loss, kl_loss, transformation_model, registration_module, metrics,
                      optimizer_phi,
                      config=config,
                      data_loader=data_loader, valid_data_loader=valid_data_loader)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LearnSim')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
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
