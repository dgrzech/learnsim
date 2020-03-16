from parse_config import ConfigParser
from trainer import Trainer

import argparse
import collections
import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.distributions as model_distr
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

    dims = (dim_x, dim_y, dim_z)

    transformation_model = config.init_obj('transformation_model', transformation, dim_x, dim_y, dim_z)
    registration_module = config.init_obj('registration_module', registration)

    # losses
    num_components = config['data_loss']['args']['num_components']

    data_loss = config.init_obj('data_loss', model_loss)
    scale_prior = config.init_obj('scale_prior', model_distr)
    proportion_prior = config.init_obj('proportion_prior', model_distr)

    reg_loss = config.init_obj('reg_loss', model_loss, dims)
    reg_loss_prior_loc = config.init_obj('reg_loss_prior_loc', model_distr)
    reg_loss_prior_scale = config.init_obj('reg_loss_prior_scale', model_distr)

    entropy_loss = config.init_obj('entropy_loss', model_loss)

    # metrics
    structures_dict = {'left_thalamus': 10, 'left_caudate': 11, 'left_putamen': 12, 
                       'left_pallidum': 13, 'brain_stem': 16, 'left_hippocampus': 17, 
                       'left_amygdala': 18, 'left_accumbens': 26, 'right_thalamus': 49, 
                       'right_caudate': 50, 'right_putamen': 51, 'right_pallidum': 52, 
                       'right_hippocampus': 53, 'right_amygdala': 54, 'right_accumbens': 58}
    
    asd_vi = ['ASD/VI/' + structure for structure in structures_dict]
    dsc_vi = ['DSC/VI/' + structure for structure in structures_dict]
    sigmas_vi = ['GM/VI/sigma_' + str(idx) for idx in range(num_components)]
    proportions_vi = ['GM/VI/proportion_' + str(idx) for idx in range(num_components)]

    metrics_vi = ['VI/data_term', 'VI/reg_term', 'VI/entropy_term', 'VI/total_loss',
                  'other/max_updates/mu_v', 'other/max_updates/log_var_v', 'other/max_updates/u_v',
                  'other/VI/alpha', 'other/VI/loc', 'other/VI/log_scale', 'other/VI/y'] + sigmas_vi + proportions_vi + asd_vi + dsc_vi

    asd_mcmc = ['ASD/MCMC/' + structure for structure in structures_dict]
    dsc_mcmc = ['DSC/MCMC/' + structure for structure in structures_dict]
    sigmas_mcmc = ['GM/MCMC/sigma_' + str(idx) for idx in range(num_components)]
    proportions_mcmc = ['GM/MCMC/proportion_' + str(idx) for idx in range(num_components)]

    metrics_mcmc = ['MCMC/data_term', 'MCMC/reg_term',
                    'other/MCMC/alpha', 'other/MCMC/loc', 'other/MCMC/log_scale', 'other/MCMC/y'] \
                   + sigmas_mcmc + proportions_mcmc + asd_mcmc + dsc_mcmc

    # run the model
    trainer = Trainer(data_loss, scale_prior, proportion_prior, reg_loss, reg_loss_prior_loc, reg_loss_prior_scale,
                      entropy_loss, transformation_model, registration_module, metrics_vi, metrics_mcmc, structures_dict,
                      config=config, data_loader=data_loader)
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
    options = [
        CustomArgs(['-vi', '--variational_inference'], type=int, target='trainer;vi'),
        CustomArgs(['-mcmc', '--markov_chain_monte_carlo'], type=int, target='trainer;mcmc')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
