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


def save_optimiser_to_disk(optimiser, file_path):
    """
    save an optimiser state to a .pth file

    :param optimiser: optimiser
    :param file_path: path to use
    """

    state_dict = optimiser.state_dict()
    torch.save(state_dict, file_path)


"""
images
"""


def save_im(save_dirs_dict, im_pair_idx, im, spacing, name, sample=False):
    folder = save_dirs_dict['samples'] if sample else save_dirs_dict['images']
    im_path = path.join(folder, name + '_' + str(im_pair_idx) + '.nii.gz')

    if not path.exists(im_path) or name == 'im_moving_warped':
        save_im_to_disk(im, im_path, spacing)


def save_images(data_loader, im_pair_idxs, **kwargs):
    """
    save input and output images to .nii.gz
    """

    save_dirs_dict = data_loader.save_dirs
    spacing = data_loader.dataset.spacing

    for im_name, im_batch in kwargs.items():
        if im_batch.type() in {'torch.BoolTensor', 'torch.cuda.BoolTensor',
                               'torch.ShortTensor', 'torch.cuda.ShortTensor'}:
            im_batch = im_batch.float()

        im_batch = im_batch.cpu().numpy()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs.tolist()):
            im = im_batch[loop_idx, 0]
            save_im(save_dirs_dict, im_pair_idx, im, spacing, im_name)


"""
vector field norms
"""


def save_norm(save_dirs_dict, im_pair_idx, norm, spacing, name):
    norm_path = path.join(save_dirs_dict['norms'], name + '_norm_' + str(im_pair_idx) + '.nii.gz')
    save_im_to_disk(norm, norm_path, spacing)


def save_norms(data_loader, im_pair_idxs, **kwargs):
    """
    save input and output images to .nii.gz
    """

    save_dirs_dict = data_loader.save_dirs
    spacing = data_loader.dataset.spacing

    for field_name, field_batch in kwargs.items():
        field_batch = calc_norm(field_batch * spacing[0])
        field_batch = field_batch.cpu().numpy()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs.tolist()):
            field_norm = field_batch[loop_idx, 0]
            save_norm(save_dirs_dict, im_pair_idx, field_norm, spacing, field_name)


"""
vector fields
"""


def save_field(save_dirs_dict, im_pair_idx, field, spacing, field_name, sample=False):
    folder = save_dirs_dict['samples'] if sample else save_dirs_dict['fields']
    field_path = path.join(folder, field_name + '_' + str(im_pair_idx) + '.vtk')
    save_field_to_disk(field, field_path, spacing)


def save_fields(data_loader, im_pair_idxs, **kwargs):
    save_dirs_dict = data_loader.save_dirs
    spacing = data_loader.dataset.spacing

    for field_name, field_batch in kwargs.items():
        field_batch = field_batch * spacing[0]
        field_batch = field_batch.cpu().numpy()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs.tolist()):
            field_norm = field_batch[loop_idx]
            save_field(save_dirs_dict, im_pair_idx, field_norm, spacing, field_name)


"""
transformation grids
"""


def save_grids(data_loader, im_pair_idxs, grids):
    """
    save output structured grids to .vtk
    """

    save_dirs_dict = data_loader.save_dirs

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs.tolist()):
        grid_path = path.join(save_dirs_dict['grids'], 'grid_' + str(im_pair_idx) + '.vtk')
        grid = grids[loop_idx]
        save_grid_to_disk(grid, grid_path)


"""
samples
"""


def save_sample(data_loader, im_pair_idxs, sample_no, im_moving_warped_batch, displacement_batch, model='MCMC'):
    """
    save output images and vector fields related to a sample from MCMC
    """

    save_dirs_dict = data_loader.save_dirs
    spacing = data_loader.dataset.spacing

    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    displacement_batch = displacement_batch * spacing[0]
    displacement_batch = displacement_batch.cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs.tolist()):
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        name = 'sample_' + model + '_' + str(sample_no) + '_im_moving_warped'
        save_im(save_dirs_dict, im_pair_idx, im_moving_warped, spacing, name, sample=True)

        displacement = displacement_batch[loop_idx]
        name = 'sample_' + model + '_' + str(sample_no) + '_displacement'
        save_field(save_dirs_dict, im_pair_idx, displacement, spacing, name, sample=True)
