import argparse

import torch
import torch.distributed as dist

import model.model as model
from parse_config import ConfigParser
from trainer import Trainer

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def train(config, rank):
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


if __name__ == '__main__':
    # parse arguments
    args = argparse.ArgumentParser(description='LearnSim')

    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-l', '--local_rank', default=0, type=int)
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')

    # config
    config = ConfigParser.from_args(args, [])

    rank = args.local_rank
    world_size = config['no_GPUs']

    # run training
    torch.cuda.set_device(rank)

    dist.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)
    train(config, rank)
    dist.destroy_process_group()
