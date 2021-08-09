import argparse
from datetime import datetime

import torch

from parse_config import ConfigParser
from trainer import Trainer

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def test(config):
    # data loader
    data_loader = config.init_data_loader()

    # model
    similarity_metric = config.init_model()

    # losses
    losses = config.init_losses()

    # transformation and registration modules
    transformation_module, registration_module = config.init_transformation_and_registration_modules()

    # metrics
    metrics = config.init_metrics()

    # test the model
    trainer = Trainer(config, data_loader, similarity_metric, losses, transformation_module, registration_module, metrics, is_test=True)
    trainer.test()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='learnsim')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    # config
    timestamp = datetime.now().strftime(r'%m%d_%H%M')
    config = ConfigParser.from_args(parser, timestamp=timestamp, test=True)
    test(config)
