import logging
import logging.config
from os import path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from tvtk.api import tvtk, write_data

from utils import read_json


class Logger(logging.Logger):
    def __init__(self, name, level=logging.DEBUG):
        super(Logger, self).__init__(name, level)

    def debug(self, msg, *args, **kwargs):
        if dist.get_rank() == 0:
            return super(Logger, self).debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if dist.get_rank() == 0:
            return super(Logger, self).info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if dist.get_rank() == 0:
            return super(Logger, self).warning(msg, *args, **kwargs)


def setup_logging(log_dir):
    log_config_path = 'logger/logger_config.json'
    config = read_json(log_config_path)

    # modify logging paths based on run config
    for _, handler in config['handlers'].items():
        if 'filename' in handler:
            filename = handler['filename']
            handler['filename'] = f'{log_dir}/{filename}'

    logging.config.dictConfig(config)


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
    field_path = path.join(folder, f'{field_name}_{im_pair_idx}.vtk')
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
        grid_path = path.join(save_dirs['grids'], f'grid_{im_pair_idx}.vtk')
        grid = grids[loop_idx]
        save_grid_to_disk(grid, grid_path)


"""
images
"""


def save_im(im_pair_idx, save_dirs, spacing, im, name, sample=False):
    folder = save_dirs['samples'] if sample else save_dirs['images']
    im_path = path.join(folder, f'{name}_{im_pair_idx}.nii.gz')
    save_im_to_disk(im, im_path, spacing)


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
        name = f'sample_{sample_no}_im_moving_warped'
        save_im(im_pair_idx, save_dirs, spacing, im_moving_warped, name, sample=True)

        displacement = displacement_batch[loop_idx]
        name = f'sample_{sample_no}_displacement'
        save_field(im_pair_idx, save_dirs, spacing, displacement, name, sample=True)


"""
variational parameters
"""


@torch.no_grad()
def save_var_params(im_pair_idxs, save_dirs, var_params_q_v):
    mu_v = var_params_q_v['mu']
    log_var_v = var_params_q_v['log_var']
    u_v = var_params_q_v['u']

    def save_state_dict(im_pair_idx, state_dict, name):
        state_dict_path = path.join(save_dirs['var_params'], f'{name}_{im_pair_idx}.pt')
        torch.save(state_dict, state_dict_path)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_pair_state_dict = {'mu': mu_v[loop_idx].cpu(), 'log_var': log_var_v[loop_idx].cpu(), 'u': u_v[loop_idx].cpu()}
        save_state_dict(im_pair_idx, im_pair_state_dict, 'var_params')
