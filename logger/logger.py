from os import path

import matplotlib.pyplot as plt

from pathlib import Path
from utils import compute_norm, read_json, save_field_to_disk, save_im_to_disk

import logging
import logging.config


def im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices):
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['im_fixed', 'im_moving', 'im_moving_warped']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_fixed_slices[i])
        axs[1, i].imshow(im_moving_slices[i])
        axs[2, i].imshow(im_moving_warped_slices[i])

    return fig


def var_params_q_f_grid(log_var_f_slices, u_f_slices):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['log_var_f', 'u_f']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(log_var_f_slices[i])
        axs[1, i].imshow(u_f_slices[i])

    return fig


def var_params_q_v_grid(mu_v_norm_slices, deformation_field_norm_slices, log_var_v_norm_slices, u_v_norm_slices):
    fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['mu_v_norm', 'deformation_field_norm', 'log_var_v_norm', 'u_v_norm']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(mu_v_norm_slices[i])
        axs[1, i].imshow(deformation_field_norm_slices[i])
        axs[2, i].imshow(log_var_v_norm_slices[i])
        axs[3, i].imshow(u_v_norm_slices[i])

    return fig


def log_images(writer, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_fixed = im_fixed_batch[loop_idx, 0]
        im_fixed_slices = [im_fixed[:, :, 64], im_fixed[:, 64, :], im_fixed[64, :, :]]

        im_moving = im_moving_batch[loop_idx, 0]
        im_moving_slices = [im_moving[:, :, 64], im_moving[:, 64, :], im_moving[64, :, :]]

        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        im_moving_warped_slices = [im_moving_warped[:, :, 64], im_moving_warped[:, 64, :], im_moving_warped[64, :, :]]

        writer.add_figure('im_pair_' + str(im_pair_idx),
                          im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices))


def log_q_v(writer, im_pair_idxs, mu_v_batch, deformation_field_batch, log_var_v_batch, u_v_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        temp = compute_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()
        mu_v_norm_slices = [mu_v_norm[:, :, 64],
                            mu_v_norm[:, 64, :],
                            mu_v_norm[64, :, :]]

        temp = compute_norm(deformation_field_batch[loop_idx])
        deformation_field_norm = temp[0].cpu().numpy()
        deformation_field_norm_slices = [deformation_field_norm[:, :, 64],
                                         deformation_field_norm[:, 64, :],
                                         deformation_field_norm[64, :, :]]

        temp = compute_norm(log_var_v_batch[loop_idx])
        log_var_v_norm = temp[0].cpu().numpy()
        log_var_v_norm_slices = [log_var_v_norm[:, :, 64],
                                 log_var_v_norm[:, 64, :],
                                 log_var_v_norm[64, :, :]]

        temp = compute_norm(u_v_batch[loop_idx])
        u_v_norm = temp[0].cpu().numpy()
        u_v_norm_slices = [u_v_norm[:, :, 64],
                           u_v_norm[:, 64, :],
                           u_v_norm[64, :, :]]

        writer.add_figure('q_v_' + str(im_pair_idx),
                          var_params_q_v_grid(mu_v_norm_slices, deformation_field_norm_slices,
                                              log_var_v_norm_slices, u_v_norm_slices))


def log_q_f(writer, im_pair_idxs, log_var_f_batch, u_f_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    log_var_f_batch = log_var_f_batch.cpu().numpy()
    u_f_batch = u_f_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        log_var_f = log_var_f_batch[loop_idx, 0]
        log_var_f_slices = [log_var_f[:, :, 64], log_var_f[:, 64, :], log_var_f[64, :, :]]

        u_f = u_f_batch[loop_idx, 0]
        u_f_slices = [u_f[:, :, 64], u_f[:, 64, :], u_f[64, :, :]]

        writer.add_figure('q_f_' + str(im_pair_idx),
                          var_params_q_f_grid(log_var_f_slices, u_f_slices))


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


def save_deformation_field_norm(save_dirs_dict, im_pair_idx, deformation_field_norm):
    deformation_field_norm_path = path.join(save_dirs_dict['norms'],
                                            'deformation_field_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(deformation_field_norm, deformation_field_norm_path)


def save_log_var_v_norm(save_dirs_dict, im_pair_idx, log_var_v_norm):
    log_var_v_norm_path = path.join(save_dirs_dict['norms'], 'log_var_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(log_var_v_norm, log_var_v_norm_path)


def save_u_v_norm(save_dirs_dict, im_pair_idx, u_v_norm):
    u_v_norm_path = path.join(save_dirs_dict['norms'], 'u_v_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(u_v_norm, u_v_norm_path)


def save_images_q_v(save_dirs_dict, im_pair_idx, mu_v_norm, deformation_field_norm, log_var_v_norm, u_v_norm):
    save_mu_v_norm(save_dirs_dict, im_pair_idx, mu_v_norm)
    save_deformation_field_norm(save_dirs_dict, im_pair_idx, deformation_field_norm)
    save_log_var_v_norm(save_dirs_dict, im_pair_idx, log_var_v_norm)
    save_u_v_norm(save_dirs_dict, im_pair_idx, u_v_norm)


def save_log_var_f_image(save_dirs_dict, im_pair_idx, log_var_f):
    log_var_f_im_path = path.join(save_dirs_dict['images'], 'log_var_f_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(log_var_f, log_var_f_im_path)


def save_u_f_image(save_dirs_dict, im_pair_idx, u_f):
    u_f_im_path = path.join(save_dirs_dict['images'], 'u_f_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(u_f, u_f_im_path)


def save_images_q_f(save_dirs_dict, im_pair_idx, log_var_f, u_f):
    save_log_var_f_image(save_dirs_dict, im_pair_idx, log_var_f)
    save_u_f_image(save_dirs_dict, im_pair_idx, u_f)


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
                mu_v_batch, deformation_field_batch, log_var_v_batch, u_v_batch, log_var_f_batch, u_f_batch,
                seg_fixed_batch=None, seg_moving_batch=None, seg_moving_warped_batch=None):
    """
    save the input and output images as well as norms of vectors in the vector fields to disk
    """

    im_pair_idxs = im_pair_idxs.tolist()
    
    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    log_var_f_batch = log_var_f_batch.cpu().numpy()
    u_f_batch = u_f_batch.cpu().numpy()

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

        log_var_f = log_var_f_batch[loop_idx, 0]
        u_f = u_f_batch[loop_idx, 0]

        mu_v_field_path = path.join(save_dirs_dict['mu_v_field'], 'mu_v_' + str(im_pair_idx) + '.nii.gz')
        save_field_to_disk(mu_v_batch[loop_idx], mu_v_field_path)

        temp = compute_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()

        temp = compute_norm(deformation_field_batch[loop_idx])
        deformation_field_norm = temp[0].cpu().numpy()

        temp = compute_norm(log_var_v_batch[loop_idx])
        log_var_v_norm = temp[0].cpu().numpy()

        temp = compute_norm(u_v_batch[loop_idx])
        u_v_norm = temp[0].cpu().numpy()

        save_im_fixed(save_dirs_dict, im_pair_idx, im_fixed)
        save_im_moving(save_dirs_dict, im_pair_idx, im_moving)
        save_im_moving_warped(save_dirs_dict, im_pair_idx, im_moving_warped)

        save_images_q_v(save_dirs_dict, im_pair_idx, mu_v_norm, deformation_field_norm, log_var_v_norm, u_v_norm)
        save_images_q_f(save_dirs_dict, im_pair_idx, log_var_f, u_f)

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
