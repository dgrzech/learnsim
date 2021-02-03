import argparse

import model.loss as model_loss
import model.model as model
import numpy as np
import torch

import data_loader.data_loaders as module_data
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

torch.autograd.set_detect_anomaly(False)  # TODO


def main(config):
    # setup the data_loader instance
    data_loader = config.init_obj('data_loader', module_data, save_dirs=config.save_dirs)

    # model
    encoder = config.init_obj('model', model)

    # losses
    data_loss = config.init_obj('data_loss', model_loss)
    reg_loss = config.init_obj('reg_loss', model_loss)
    entropy_loss = config.init_obj('entropy_loss', model_loss)

    losses = {'data': data_loss, 'regularisation': reg_loss, 'entropy': entropy_loss}

    # transformation model and registration module
    transformation_model = config.init_obj('transformation_model', transformation, data_loader.dims)
    registration_module = config.init_obj('registration_module', registration)

    # metrics
    structures_dict = {'left_thalamus': 10, 'left_caudate': 11, 'left_putamen': 12,
                       'left_pallidum': 13, 'brain_stem': 16, 'left_hippocampus': 17,
                       'left_amygdala': 18, 'left_accumbens': 26, 'right_thalamus': 49,
                       'right_caudate': 50, 'right_putamen': 51, 'right_pallidum': 52,
                       'right_hippocampus': 53, 'right_amygdala': 54, 'right_accumbens': 58}

    loss_terms = ['train/loss/data_term', 'train/loss/reg_term', 'train/loss/entropy_term']
    loss_vals = ['train/loss/q_v', 'train/loss/q_f_q_phi']

    no_samples = data_loader.no_samples

    ASD = ['train/ASD/im_pair_' + str(im_pair_idx) + '/' + structure for structure in structures_dict for im_pair_idx in range(no_samples)]
    DSC = ['train/DSC/im_pair_' + str(im_pair_idx) + '/' + structure for structure in structures_dict for im_pair_idx in range(no_samples)]

    ASD_avg = ['train/ASD/avg/' + structure for structure in structures_dict]  # TODO
    DSC_avg = ['train/DSC/avg/' + structure for structure in structures_dict]

    no_non_diffeomorphic_voxels = ['train/no_non_diffeomorphic_voxels/im_pair_' + str(im_pair_idx) for im_pair_idx in range(no_samples)]
    metrics = loss_terms + loss_vals + ASD + ASD_avg + DSC + DSC_avg + no_non_diffeomorphic_voxels

    # train the model
    trainer = Trainer(config, data_loader, encoder, losses, transformation_model, registration_module, metrics, structures_dict)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LearnSim')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, [])
    main(config)
