import argparse
from datetime import datetime

import torch
import torch.distributed as dist

from parse_config import ConfigParser
from trainer import Trainer

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def train(config):
    # data loader
    data_loader = config.init_data_loader()

    # parameters used with other objects
    dims = data_loader.dims
    no_samples = data_loader.no_samples

    # model
    similarity_metric = config.init_model()

    # losses
    losses = config.init_losses()

    # transformation and registration modules
    transformation_module, registration_module = config.init_transformation_and_registration_modules(dims)

    # metrics
    metrics = config.init_metrics(no_samples)

    # training
    trainer = Trainer(config, data_loader, similarity_metric, losses, transformation_module, registration_module, metrics)
    trainer.train()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='LearnSim')

    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-l', '--local_rank', default=0, type=int)
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')

    args = parser.parse_args()
    rank = args.local_rank

    torch.cuda.set_device(rank)

    # config
    timestamp = datetime.now().strftime(r'%m%d_%H%M')
    config = ConfigParser.from_args(parser, timestamp=timestamp)

    # run training
    world_size = config['no_GPUs']
    dist.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)

    train(config)
    dist.destroy_process_group()
