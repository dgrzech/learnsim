import argparse
import collections

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.distributions as model_distr
import model.loss as model_loss
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
    # setup data_loader instance
    data_loader = config.init_obj('data_loader', module_data, save_dirs=config.save_dirs)

    # initialise the transformation model and registration modules
    dim_x = config['data_loader']['args']['dim_x']
    dim_y = config['data_loader']['args']['dim_y']
    dim_z = config['data_loader']['args']['dim_z']

    dims = (dim_x, dim_y, dim_z)

    transformation_model = config.init_obj('transformation_model', transformation, dim_x, dim_y, dim_z)
    registration_module = config.init_obj('registration_module', registration)

    # losses
    data_loss = config.init_obj('data_loss', model_loss)
    data_loss_scale_prior = config.init_obj('data_loss_scale_prior', model_distr)
    data_loss_proportion_prior = config.init_obj('data_loss_proportion_prior', model_distr)

    reg_loss = config.init_obj('reg_loss', model_loss, dims)
    reg_loss_loc_prior = config.init_obj('reg_loss_loc_prior', model_distr)
    reg_loss_scale_prior = config.init_obj('reg_loss_scale_prior', model_distr)

    entropy_loss = config.init_obj('entropy_loss', model_loss)

    # metrics
    num_components = config['data_loss']['args']['num_components']

    sigmas_VI = ['VI/train/GMM/sigma_' + str(idx) for idx in range(num_components)]
    proportions_VI = ['VI/train/GMM/proportion_' + str(idx) for idx in range(num_components)]

    sigmas_MCMC = ['MCMC/GMM/sigma_' + str(idx) for idx in range(num_components)]
    proportions_MCMC = ['MCMC/GMM/proportion_' + str(idx) for idx in range(num_components)]

    structures_dict = {'left_thalamus': 10, 'left_caudate': 11, 'left_putamen': 12,
                       'left_pallidum': 13, 'brain_stem': 16, 'left_hippocampus': 17,
                       'left_amygdala': 18, 'left_accumbens': 26, 'right_thalamus': 49,
                       'right_caudate': 50, 'right_putamen': 51, 'right_pallidum': 52,
                       'right_hippocampus': 53, 'right_amygdala': 54, 'right_accumbens': 58}

    ASD_VI = ['VI/train/ASD/' + structure for structure in structures_dict] + \
             ['VI/test/ASD/' + structure for structure in structures_dict]
    DSC_VI = ['VI/train/DSC/' + structure for structure in structures_dict] + \
             ['VI/test/DSC/' + structure for structure in structures_dict]

    ASD_MCMC = ['MCMC/ASD/' + structure for structure in structures_dict]
    DSC_MCMC = ['MCMC/DSC/' + structure for structure in structures_dict]

    metrics_VI = ['VI/train/data_term', 'VI/train/reg_term', 'VI/train/entropy_term', 'VI/train/total_loss',
                  'VI/train/no_non_diffeomorphic_voxels', 'VI/test/no_non_diffeomorphic_voxels',
                  'VI/train/max_updates/mu_v', 'VI/train/max_updates/log_var_v', 'VI/train/max_updates/u_v',
                  'VI/train/alpha', 'VI/train/loc', 'VI/train/log_scale', 'VI/train/y'] + \
                 sigmas_VI + proportions_VI + ASD_VI + DSC_VI

    metrics_MCMC = ['MCMC/data_term', 'MCMC/reg_term', 'MCMC/no_non_diffeomorphic_voxels',
                    'MCMC/alpha', 'MCMC/loc', 'MCMC/log_scale', 'MCMC/y'] + \
                   sigmas_MCMC + proportions_MCMC + ASD_MCMC + DSC_MCMC

    # run the model
    trainer = Trainer(data_loss, data_loss_scale_prior, data_loss_proportion_prior,
                      reg_loss, reg_loss_loc_prior, reg_loss_scale_prior,
                      entropy_loss, transformation_model, registration_module,
                      metrics_VI, metrics_MCMC, structures_dict, config=config, data_loader=data_loader)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MCMC')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in the .json file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [CustomArgs(['-VI', '--variational_inference'], type=int, target='trainer;VI'),
               CustomArgs(['-MCMC', '--markov_chain_monte_carlo'], type=int, target='trainer;MCMC')]

    config = ConfigParser.from_args(args, options)
    main(config)
