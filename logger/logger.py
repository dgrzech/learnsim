from os import path

from pathlib import Path
from utils import compute_norm, read_json, save_field_to_disk, save_grid_to_disk, save_im_to_disk

import logging
import logging.config


def registration_print(logger, iter_no, no_steps_v, loss_q_v, data_term, reg_term, entropy_term):
    """
    print value of the energy function at a given step of registration
    """

    logger.info(f'ITERATION ' + str(iter_no) + '/' + str(no_steps_v - 1) +
                f', TOTAL ENERGY: {loss_q_v:.5f}' +
                f'\ndata: {data_term:.5f}' +
                f', regularisation: {reg_term:.5f}' +
                f', entropy: {entropy_term:.5f}'
                )


def mixing_print(logger, sample_idx, no_samples, loss_q_v, data_term, reg_term):
    """
    print value of the energy function at a given current state
    """

    logger.info(f'SAMPLE ' + str(sample_idx) + '/' + str(no_samples - 1) +
                f', TOTAL ENERGY: {loss_q_v:.5f}' +
                f'\ndata: {data_term:.5f}' +
                f', regularisation: {reg_term:.5f}'
                )


def save_grids(save_dirs_dict, im_pair_idxs, grids):
    im_pair_idxs = im_pair_idxs.tolist()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        grid = grids[loop_idx]
        grid_path = path.join(save_dirs_dict['grids'], 'grid_' + str(im_pair_idx) + '.vtk')

        save_grid_to_disk(grid, grid_path)


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


def save_v_norm(save_dirs_dict, im_pair_idx, v_norm):
    v_norm_path = path.join(save_dirs_dict['norms'], 'v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(v_norm, v_norm_path)


def save_displacement_norm(save_dirs_dict, im_pair_idx, displacement_norm):
    displacement_norm_path = path.join(save_dirs_dict['norms'],
                                       'displacement_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(displacement_norm, displacement_norm_path)


def save_log_det_J(save_dirs_dict, im_pair_idx, log_det_J):
    log_det_J_path = path.join(save_dirs_dict['log_det_J'], 'log_det_J_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(log_det_J, log_det_J_path)


def save_images(save_dirs_dict, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch,
                v_batch, displacement_batch, log_det_J_batch):
    """
    save the input and output images as well as norms of vectors in the vector fields to disk
    """

    im_pair_idxs = im_pair_idxs.tolist()
    
    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()
    log_det_J_batch = log_det_J_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_fixed = im_fixed_batch[loop_idx, 0]
        im_moving = im_moving_batch[loop_idx, 0]
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]

        v_field_path = path.join(save_dirs_dict['v_field'], 'v_' + str(im_pair_idx) + '.vtk')
        save_field_to_disk(v_batch[loop_idx], v_field_path)

        temp = compute_norm(v_batch[loop_idx])
        v_norm = temp[0].cpu().numpy()

        temp = compute_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()
        
        save_im_fixed(save_dirs_dict, im_pair_idx, im_fixed)
        save_im_moving(save_dirs_dict, im_pair_idx, im_moving)
        save_im_moving_warped(save_dirs_dict, im_pair_idx, im_moving_warped)
        save_v_norm(save_dirs_dict, im_pair_idx, v_norm)
        save_displacement_norm(save_dirs_dict, im_pair_idx, displacement_norm)

        log_det_J = log_det_J_batch[loop_idx]
        save_log_det_J(save_dirs_dict, im_pair_idx, log_det_J)


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
