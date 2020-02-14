from os import path

from pathlib import Path
from utils import calc_norm, read_json, save_field_to_disk, save_grid_to_disk, save_im_to_disk

import logging
import logging.config

import torch


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    setup logging configuration
    """

    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)

        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


def print_log(logger, log):
    """
    print logged scalars
    """

    for key, value in log.items():
        if isinstance(value, int):
            logger.info(f'    {key:25s}: {value}')
        else:
            logger.info(f'    {key:25s}: {value:.5f}')

    print()


"""
images
"""


def save_im_fixed(save_dirs_dict, im_pair_idx, im_fixed):
    im_fixed_path = path.join(save_dirs_dict['images'], 'im_fixed_' + str(im_pair_idx) + '.nii.gz')
    
    if not path.exists(im_fixed_path):
        save_im_to_disk(im_fixed, im_fixed_path)

        
def save_im_moving(save_dirs_dict, im_pair_idx, im_moving):
    im_moving_path = path.join(save_dirs_dict['images'], 'im_moving_' + str(im_pair_idx) + '.nii.gz')
    
    if not path.exists(im_moving_path):
        save_im_to_disk(im_moving, im_moving_path)


def save_im_moving_warped(save_dirs_dict, im_pair_idx, im_moving_warped):
    im_moving_warped_path = path.join(save_dirs_dict['images'], 'im_moving_warped_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(im_moving_warped, im_moving_warped_path)


def save_images(save_dirs_dict, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch):
    """
    save input and output images to .nii.gz
    """

    im_pair_idxs = im_pair_idxs.tolist()

    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_fixed = im_fixed_batch[loop_idx, 0]
        save_im_fixed(save_dirs_dict, im_pair_idx, im_fixed)

        im_moving = im_moving_batch[loop_idx, 0]
        save_im_moving(save_dirs_dict, im_pair_idx, im_moving)

        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        save_im_moving_warped(save_dirs_dict, im_pair_idx, im_moving_warped)


"""
vector field norms
"""


def save_displacement_norm(save_dirs_dict, im_pair_idx, displacement_norm):
    displacement_norm_path = path.join(save_dirs_dict['norms'],
                                       'displacement_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(displacement_norm, displacement_norm_path)


def save_mu_v_norm(save_dirs_dict, im_pair_idx, v_norm):
    v_norm_path = path.join(save_dirs_dict['norms'], 'v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(v_norm, v_norm_path)


def save_sigma_v_norm(save_dirs_dict, im_pair_idx, sigma_v_norm):
    sigma_v_norm_path = path.join(save_dirs_dict['norms'], 'sigma_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(sigma_v_norm, sigma_v_norm_path)


def save_u_v_norm(save_dirs_dict, im_pair_idx, u_v_norm):
    u_v_norm_path = path.join(save_dirs_dict['norms'], 'u_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(u_v_norm, u_v_norm_path)


def save_norms(save_dirs_dict, im_pair_idxs, var_params_batch, displacement_batch):
    """
    save norms of output vector fields to .nii.gz
    """

    im_pair_idxs = im_pair_idxs.tolist()

    mu_v_batch = var_params_batch['mu_v']
    log_var_v_batch = var_params_batch['log_var_v']
    sigma_v_batch = torch.exp(0.5 * log_var_v_batch)
    u_v_batch = var_params_batch['u_v']

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        temp = calc_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()
        save_mu_v_norm(save_dirs_dict, im_pair_idx, mu_v_norm)

        temp = calc_norm(sigma_v_batch[loop_idx])
        sigma_v_norm = temp[0].cpu().numpy()
        save_sigma_v_norm(save_dirs_dict, im_pair_idx, sigma_v_norm)

        temp = calc_norm(u_v_batch[loop_idx])
        u_v_norm = temp[0].cpu().numpy()
        save_u_v_norm(save_dirs_dict, im_pair_idx, u_v_norm)

        temp = calc_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()
        save_displacement_norm(save_dirs_dict, im_pair_idx, displacement_norm)


"""
vector fields
"""


def save_log_det_J(save_dirs_dict, im_pair_idx, log_det_J):
    log_det_J_path = path.join(save_dirs_dict['fields'], 'log_det_J_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(log_det_J, log_det_J_path)


def save_displacement_field(save_dirs_dict, im_pair_idx, displacement):
    displacement_path = path.join(save_dirs_dict['fields'], 'displacement_' + str(im_pair_idx) + '.vtk')
    save_field_to_disk(displacement, displacement_path)


def save_mu_v_field(save_dirs_dict, im_pair_idx, v):
    v_path = path.join(save_dirs_dict['fields'], 'mu_v_' + str(im_pair_idx) + '.vtk')
    save_field_to_disk(v, v_path)


def save_sigma_v_field(save_dirs_dict, im_pair_idx, sigma_v):
    sigma_v_field_path = path.join(save_dirs_dict['fields'], 'sigma_v_' + str(im_pair_idx) + '.vtk')
    save_field_to_disk(sigma_v, sigma_v_field_path)


def save_u_v_field(save_dirs_dict, im_pair_idx, u_v):
    u_v_field_path = path.join(save_dirs_dict['fields'], 'u_v_' + str(im_pair_idx) + '.vtk')
    save_field_to_disk(u_v, u_v_field_path)


def save_fields(save_dirs_dict, im_pair_idxs, var_params_batch, displacement_batch, log_det_J_batch):
    """
    save output vector fields to .vtk
    """

    im_pair_idxs = im_pair_idxs.tolist()

    mu_v_batch = var_params_batch['mu_v']
    log_var_v_batch = var_params_batch['log_var_v']
    sigma_v_batch = torch.exp(0.5 * log_var_v_batch)
    u_v_batch = var_params_batch['u_v']
    log_det_J_batch = log_det_J_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        mu_v = mu_v_batch[loop_idx]
        save_mu_v_field(save_dirs_dict, im_pair_idx, mu_v)

        sigma_v = sigma_v_batch[loop_idx]
        save_sigma_v_field(save_dirs_dict, im_pair_idx, sigma_v)

        u_v = u_v_batch[loop_idx]
        save_u_v_field(save_dirs_dict, im_pair_idx, u_v)

        displacement = displacement_batch[loop_idx]
        save_displacement_field(save_dirs_dict, im_pair_idx, displacement)

        log_det_J = log_det_J_batch[loop_idx]
        save_log_det_J(save_dirs_dict, im_pair_idx, log_det_J)


"""
transformation grids
"""


def save_grids(save_dirs_dict, im_pair_idxs, grids):
    """
    save output structured grids to .vtk
    """

    im_pair_idxs = im_pair_idxs.tolist()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        grid = grids[loop_idx]

        grid_path = path.join(save_dirs_dict['grids'], 'grid_' + str(im_pair_idx) + '.vtk')
        save_grid_to_disk(grid, grid_path)


"""
samples
"""


def save_sample(save_dirs_dict, im_pair_idxs, sample_no, im_moving_warped_batch, v_batch):
    """
    save output images and vector fields related to a sample from MCMC
    """

    im_pair_idxs = im_pair_idxs.tolist()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        im_moving_warped_path = \
            path.join(save_dirs_dict['samples'],
                      'sample_' + str(sample_no) + '_im_moving_warped_' + str(im_pair_idx) + '.nii.gz')
        save_im_to_disk(im_moving_warped, im_moving_warped_path)

        v = v_batch[loop_idx]
        v_path = path.join(save_dirs_dict['samples'], 'sample_' + str(sample_no) + '_v_' + str(im_pair_idx) + '.vtk')
        save_field_to_disk(v, v_path)
