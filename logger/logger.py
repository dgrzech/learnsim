import json
import logging
import logging.config
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tvtk.api import tvtk, write_data

from utils import read_json


class Logger(logging.Logger):
    def __init__(self, name, level=logging.DEBUG):
        super(Logger, self).__init__(name, level)

        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0

    def debug(self, msg, *args, **kwargs):
        if self.rank == 0:
            return super(Logger, self).debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.rank == 0:
            return super(Logger, self).info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.rank == 0:
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


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self._data = pd.DataFrame(index=keys, columns=['value'])
        self.writer = writer

        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0

        self.reset()

    def reset(self):
        if self.rank == 0:
            for col in self._data.columns:
                self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.rank == 0:
            if self.writer is not None:
                self.writer.add_scalar(key, value / n)

            self._data.value[key] = value

    def update_ASD_and_DSC(self, structures_dict, ASD, DSC, im_pair_idx=None):
        if im_pair_idx is None:
            if self.rank == 0:
                ASD_dict = dict(zip(structures_dict.keys(), ASD.mean(dim=0)))
                DSC_dict = dict(zip(structures_dict.keys(), DSC.mean(dim=0)))

                for key in ASD_dict:
                    self.writer.add_scalar(f'ASD/{key}', ASD_dict[key])
                    self.writer.add_scalar(f'DSC/{key}', DSC_dict[key])
        else:
            ASD_dict = dict(zip(structures_dict.keys(), ASD))
            DSC_dict = dict(zip(structures_dict.keys(), DSC))

            for key in ASD_dict:
                self.writer.add_scalar(f'test/ASD/im_pair_{im_pair_idx}/{key}', ASD_dict[key])
                self.writer.add_scalar(f'test/DSC/im_pair_{im_pair_idx}/{key}', DSC_dict[key])


class TensorboardWriter:
    def __init__(self, log_dir):
        self.step = 0

        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0

        if self.rank == 0:
            self.writer = SummaryWriter(log_dir)

        self.tb_writer_ftns = {'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_figure', 'add_text', 'add_histogram'}

    def set_step(self, step):
        self.step = step

    def write_graph(self, model):
        if self.rank == 0:
            im_fixed = im_moving = torch.randn([1, 1, 128, 128, 128], device=self.rank)
            mask = torch.ones([1, 1, 128, 128, 128], device=self.rank).bool()
            inputs = (im_fixed, im_moving, mask)

            self.writer.add_graph(model, input_to_model=inputs)

    def write_hparams(self, config):  # NOTE (DG): should use add_hparams but it's not working with DDP..
        if self.rank == 0:
            text = json.dumps(config.config, indent=4, sort_keys=False)
            self.writer.add_text('hparams', text)

    def __getattr__(self, name):
        """
        if visualization is configured, return add_data() methods of tensorboard with additional information (step, tag) added; else return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:  # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


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
    folder = save_dirs['samples_dir'] if sample else save_dirs['fields_dir']
    field_path = os.path.join(folder, f'{field_name}_{im_pair_idx}.vtk')
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
        grid_path = os.path.join(save_dirs['grids_dir'], f'grid_{im_pair_idx}.vtk')
        grid = grids[loop_idx]
        save_grid_to_disk(grid, grid_path)


"""
images
"""


def save_im(im_pair_idx, save_dirs, spacing, im, name, sample=False):
    folder = save_dirs['samples_dir'] if sample else save_dirs['images_dir']
    im_path = os.path.join(folder, f'{name}_{im_pair_idx}.nii.gz')
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
        state_dict_path = os.path.join(save_dirs['var_params_dir'], f'{name}_{im_pair_idx}.pt')
        torch.save(state_dict, state_dict_path)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_pair_state_dict = {'mu': mu_v[loop_idx].cpu(), 'log_var': log_var_v[loop_idx].cpu(), 'u': u_v[loop_idx].cpu()}
        save_state_dict(im_pair_idx, im_pair_state_dict, 'var_params')
