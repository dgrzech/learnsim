from os import path

import torch

from pathlib import Path
from utils import compute_norm, read_json, save_field_to_disk, save_im_to_disk

import logging
import logging.config


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


def save_mu_v_norm(save_dirs_dict, im_pair_idx, mu_v_norm):
    mu_v_norm_path = path.join(save_dirs_dict['norms'], 'mu_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(mu_v_norm, mu_v_norm_path)


def save_displacement_norm(save_dirs_dict, im_pair_idx, displacement_norm):
    displacement_norm_path = path.join(save_dirs_dict['norms'],
                                       'displacement_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(displacement_norm, displacement_norm_path)


def save_log_var_v_norm(save_dirs_dict, im_pair_idx, log_var_v_norm):
    log_var_v_norm_path = path.join(save_dirs_dict['norms'], 'log_var_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(log_var_v_norm, log_var_v_norm_path)


def save_u_v_norm(save_dirs_dict, im_pair_idx, u_v_norm):
    u_v_norm_path = path.join(save_dirs_dict['norms'], 'u_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(u_v_norm, u_v_norm_path)


def save_images_q_v(save_dirs_dict, im_pair_idx, mu_v_norm, displacement_norm, log_var_v_norm, u_v_norm):
    save_mu_v_norm(save_dirs_dict, im_pair_idx, mu_v_norm)
    save_displacement_norm(save_dirs_dict, im_pair_idx, displacement_norm)
    save_log_var_v_norm(save_dirs_dict, im_pair_idx, log_var_v_norm)
    save_u_v_norm(save_dirs_dict, im_pair_idx, u_v_norm)


def save_log_var_f_image(save_dirs_dict, log_var_f):
    log_var_f_im_path = path.join(save_dirs_dict['images'], 'log_var_f.nii.gz')
    save_im_to_disk(log_var_f, log_var_f_im_path)


def save_u_f_image(save_dirs_dict, u_f):
    u_f_im_path = path.join(save_dirs_dict['images'], 'u_f.nii.gz')
    save_im_to_disk(u_f, u_f_im_path)


def save_images_q_f(save_dirs_dict, log_var_f, u_f):
    save_log_var_f_image(save_dirs_dict, log_var_f)
    save_u_f_image(save_dirs_dict, u_f)


def save_seg_fixed(save_dirs_dict, im_pair_idx, seg_fixed):
    seg_fixed_path = path.join(save_dirs_dict['segs'], 'seg_fixed_' + str(im_pair_idx) + '.nii.gz')

    if not path.exists(seg_fixed_path):
        save_im_to_disk(seg_fixed, seg_fixed_path)


def save_seg_moving(save_dirs_dict, im_pair_idx, seg_moving):
    seg_moving_path = path.join(save_dirs_dict['segs'], 'seg_moving_' + str(im_pair_idx) + '.nii.gz')

    if not path.exists(seg_moving_path):
        save_im_to_disk(seg_moving, seg_moving_path)


def save_seg_moving_warped(save_dirs_dict, im_pair_idx, seg_moving_warped):
    save_im_to_disk(seg_moving_warped,
                    path.join(save_dirs_dict['segs'], 'seg_moving_warped_' + str(im_pair_idx) + '.nii.gz'))


def save_images(save_dirs_dict, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch,
                mu_v_batch, log_var_v_batch, u_v_batch, log_var_f_batch, u_f_batch, displacement_batch,
                seg_fixed_batch=None, seg_moving_batch=None, seg_moving_warped_batch=None):
    """
    save the input and output images as well as norms of vectors in the vector fields to disk
    """

    im_pair_idxs = im_pair_idxs.tolist()
    
    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    log_var_f, u_f = torch.mean(log_var_f_batch, dim=0), torch.mean(u_f_batch, dim=0)
    log_var_f, u_f = log_var_f[0].cpu().numpy(), u_f[0].cpu().numpy()

    save_images_q_f(save_dirs_dict, log_var_f, u_f)

    if seg_fixed_batch is not None:
        seg_fixed_batch = seg_fixed_batch.cpu().numpy()

    if seg_moving_batch is not None:
        seg_moving_batch = seg_moving_batch.cpu().numpy()

    if seg_moving_warped_batch is not None:
        seg_moving_warped_batch = seg_moving_warped_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_fixed = im_fixed_batch[loop_idx, 0]
        im_moving = im_moving_batch[loop_idx, 0]
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]

        mu_v_field_path = path.join(save_dirs_dict['mu_v_field'], 'mu_v_' + str(im_pair_idx) + '.vtk')
        save_field_to_disk(mu_v_batch[loop_idx], mu_v_field_path)

        temp = compute_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()

        temp = compute_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()

        temp = compute_norm(log_var_v_batch[loop_idx])
        log_var_v_norm = temp[0].cpu().numpy()

        temp = compute_norm(u_v_batch[loop_idx])
        u_v_norm = temp[0].cpu().numpy()

        save_im_fixed(save_dirs_dict, im_pair_idx, im_fixed)
        save_im_moving(save_dirs_dict, im_pair_idx, im_moving)
        save_im_moving_warped(save_dirs_dict, im_pair_idx, im_moving_warped)
        save_images_q_v(save_dirs_dict, im_pair_idx, mu_v_norm, displacement_norm, log_var_v_norm, u_v_norm)

        if seg_fixed_batch is not None:
            seg_fixed = seg_fixed_batch[loop_idx, 0]
            save_seg_fixed(save_dirs_dict, im_pair_idx, seg_fixed)

        if seg_moving_batch is not None:
            seg_moving = seg_moving_batch[loop_idx, 0]
            save_seg_moving(save_dirs_dict, im_pair_idx, seg_moving)

        if seg_moving_warped_batch is not None:
            seg_moving_warped = seg_moving_warped_batch[loop_idx, 0]
            save_seg_moving_warped(save_dirs_dict, im_pair_idx, seg_moving_warped)


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
