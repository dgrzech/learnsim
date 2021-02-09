import logging
import logging.config
from os import path
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tvtk.api import tvtk, write_data

from utils import calc_norm, read_json


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
        if 'DSC' not in key and 'ASD' not in key and 'GMM' not in key and 'test' not in key:
            if isinstance(value, int):
                logger.info(f'    {key:50s}: {value}')
            else:
                logger.info(f'    {key:50s}: {value:.5f}')

    print()


def save_field_to_disk(field, file_path, spacing=(1, 1, 1)):
    """
    save a vector field to a .vtk file

    :param field: field to save
    :param file_path: path to use
    :param spacing: voxel spacing
    """

    spacing = spacing.numpy()
    field_x, field_y, field_z = field[0], field[1], field[2]

    vectors = np.empty(field_x.shape + (3,), dtype=float)
    vectors[..., 0], vectors[..., 1], vectors[..., 2] = field_x, field_y, field_z
    vectors = vectors.transpose(2, 1, 0, 3).copy()
    vectors.shape = vectors.size // 3, 3

    im_vtk = tvtk.ImageData(spacing=spacing, origin=(0, 0, 0), dimensions=field_x.shape)
    im_vtk.point_data.vectors = vectors
    im_vtk.point_data.vectors.name = 'field'

    write_data(im_vtk, file_path)


def save_grid_to_disk(grid, file_path):
    """
    save a VTK structured grid to a .vtk file

    :param grid: grid to save
    :param file_path: path to use
    """

    grid = grid.cpu().numpy()

    x, y, z = grid[0, ...], grid[1, ...], grid[2, ...]

    pts = np.empty(x.shape + (3,), dtype=float)
    pts[..., 0], pts[..., 1], pts[..., 2] = x, y, z
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size // 3, 3

    sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
    write_data(sg, file_path)


def save_im_to_disk(im, file_path, spacing=(1, 1, 1)):
    """
    save an image stored in a numpy array to a .nii.gz file

    :param im: 3D image
    :param file_path: path to use
    :param spacing: voxel spacing
    """

    im = nib.Nifti1Image(im, np.eye(4))
    im.header.set_xyzt_units(2)

    try:
        spacing = spacing.numpy()
        im.header.set_zooms(spacing)
    except:
        im.header.set_zooms(spacing)

    im.to_filename(file_path)


"""
(vector) fields
"""


def save_field(im_pair_idx, save_dirs, spacing, field, field_name, sample=False):
    folder = save_dirs['samples'] if sample else save_dirs['fields']
    field_path = path.join(folder, field_name + '_' + str(im_pair_idx) + '.vtk')
    save_field_to_disk(field, field_path, spacing)


def save_fields(im_pair_idxs, save_dirs, spacing, **kwargs):
    for field_name, field_batch in kwargs.items():
        field_batch = field_batch * spacing[0]
        field_batch = field_batch.cpu().numpy()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            field_norm = field_batch[loop_idx]
            save_field(save_dirs, im_pair_idx, field_norm, spacing, field_name)


"""
grids
"""


def save_grids(im_pair_idxs, save_dirs, grids):
    """
    save output structured grids to .vtk
    """

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        grid_path = path.join(save_dirs['grids'], 'grid_' + str(im_pair_idx) + '.vtk')
        grid = grids[loop_idx]
        save_grid_to_disk(grid, grid_path)


"""
images
"""


def save_fixed_image(save_dirs, spacing, im_fixed):
    """
    save the input fixed image to .nii.gz
    """

    im_fixed = im_fixed[0, 0].cpu().numpy()
    im_path = path.join(save_dirs['images'], 'im_fixed.nii.gz')
    save_im_to_disk(im_fixed, im_path, spacing)


def save_im(im_pair_idx, save_dirs, spacing, im, name, sample=False):
    folder = save_dirs['samples'] if sample else save_dirs['images']
    im_path = path.join(folder, name + '_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(im, im_path, spacing)


def save_moving_images(im_pair_idxs, save_dirs, spacing, im_moving_batch):
    """
    save input moving images to .nii.gz
    """

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_moving = im_moving_batch[loop_idx, 0].cpu().numpy()
        save_im(im_pair_idx, save_dirs, spacing, im_moving, 'im_moving')


"""
optimizers
"""


def save_optimizer(batch_idx, save_dirs, optimizer, optimizer_name):
    """
    save an optimiser state to a .pth file
    """

    optimizer_path = path.join(save_dirs['optimizers'], optimizer_name + '_' + str(batch_idx) + '.pt')
    state_dict = optimizer.state_dict()
    torch.save(state_dict, optimizer_path)


"""
samples
"""


def save_sample(im_pair_idxs, save_dirs, spacing, sample_no, im_moving_warped_batch, displacement_batch):
    """
    save output images and vector fields related to a sample from MCMC
    """

    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    displacement_batch = displacement_batch * spacing[0]
    displacement_batch = displacement_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        name = 'sample_' + str(sample_no) + '_im_moving_warped'
        save_im(im_pair_idx, save_dirs, spacing, im_moving_warped, name, sample=True)

        displacement = displacement_batch[loop_idx]
        name = 'sample_' + str(sample_no) + '_displacement'
        save_field(im_pair_idx, save_dirs, spacing, displacement, name, sample=True)


"""
tensors
"""


def save_tensors(im_pair_idxs, save_dirs, var_params_q_v):
    mu_v = var_params_q_v['mu']
    log_var_v = var_params_q_v['log_var']
    u_v = var_params_q_v['u']

    def save_tensor(im_pair_idx, tensor, tensor_name):
        tensor_path = path.join(save_dirs['tensors'], tensor_name + '_' + str(im_pair_idx) + '.pt')
        torch.save(tensor.detach().cpu(), tensor_path)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        save_tensor(im_pair_idx, mu_v[loop_idx], 'mu_v')
        save_tensor(im_pair_idx, log_var_v[loop_idx], 'log_var_v')
        save_tensor(im_pair_idx, u_v[loop_idx], 'u_v')