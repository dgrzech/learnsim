from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from torch import nn
from tvtk.api import tvtk, write_data

import json
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """
    wrapper function for endless data loader.
    """

    for loader in repeat(data_loader):
        yield from loader


def calc_det_J(nabla_x, nabla_y, nabla_z):
    det_J = nabla_x[:, 0] * nabla_y[:, 1] * nabla_z[:, 2] + \
            nabla_y[:, 0] * nabla_z[:, 1] * nabla_x[:, 2] + \
            nabla_z[:, 0] * nabla_x[:, 1] * nabla_y[:, 2] - \
            nabla_x[:, 2] * nabla_y[:, 1] * nabla_z[:, 0] - \
            nabla_y[:, 2] * nabla_z[:, 1] * nabla_x[:, 0] - \
            nabla_z[:, 2] * nabla_x[:, 1] * nabla_y[:, 0]

    return det_J


def compute_norm(v):
    return torch.norm(v, p=2, dim=0, keepdim=True)


def get_module_attr(module, name):
    if isinstance(module, nn.DataParallel):
        return getattr(module.module, name)

    return getattr(module, name)


def init_identity_grid_2d(nx, ny):
    """
    initialise a 2D grid to use with grid_sample
    """

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)

    x = x.expand(ny, -1)
    y = y.expand(nx, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(3)
    y.unsqueeze_(0).unsqueeze_(3)

    return torch.cat((x, y), 3)


def init_identity_grid_3d(nx, ny, nz):
    """
    initialise a 3D grid to use with grid_sample
    """

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)
    z = torch.linspace(-1, 1, steps=nz)

    x = x.expand(ny, -1).expand(nz, -1, -1)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(4)
    y.unsqueeze_(0).unsqueeze_(4)
    z.unsqueeze_(0).unsqueeze_(4)

    return torch.cat((x, y, z), 4)


def pixel_to_normalised_2d(px_idx_x, px_idx_y, dim_x, dim_y):
    """
    normalise the coordinates of a pixel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)

    return x, y


def pixel_to_normalised_3d(px_idx_x, px_idx_y, px_idx_z, dim_x, dim_y, dim_z):
    """
    normalise the coordinates of a pixel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)
    z = -1.0 + 2.0 * px_idx_z / (dim_z - 1.0)

    return x, y, z


def normalised_3d_to_pixel_idx(x, y, z, dim_x, dim_y, dim_z):
    px_idx_x = (x + 1.0) * (dim_x - 1.0) / 2.0
    px_idx_y = (y + 1.0) * (dim_y - 1.0) / 2.0
    px_idx_z = (z + 1.0) * (dim_z - 1.0) / 2.0

    return px_idx_x, px_idx_y, px_idx_z


def resample_im_to_be_isotropic(im):
    im_spacing = im.GetSpacing()
    im_size = im.GetSize()

    min_spacing = min(im_spacing)

    new_spacing = [min_spacing, min_spacing, min_spacing]
    new_size = [int(round(im_size[0] * (im_spacing[0] / min_spacing))),
                int(round(im_size[1] * (im_spacing[1] / min_spacing))),
                int(round(im_size[2] * (im_spacing[2] / min_spacing)))]

    resampled_im = sitk.Resample(im, new_size, sitk.Transform(),
                                 sitk.sitkLinear, im.GetOrigin(),
                                 new_spacing, im.GetDirection(), 0.0,
                                 im.GetPixelID())

    return resampled_im


def rescale_im(im, range_min=-1.0, range_max=1.0):
    """
    rescale the intensity of image pixels to a given range
    """

    im_min, im_max = torch.min(im), torch.max(im)

    im = (range_max - range_min) * (im - im_min) / (im_max - im_min) + range_min
    return im


def standardise_im(im):
    """
    standardise an image to zero mean and unit variance 
    """

    im_mean, im_std = torch.mean(im), torch.std(im)
    
    im -= im_mean
    im /= im_std

    return im


def save_field_to_disk(field, file_path):
    field = field.cpu().numpy()

    field_x = field[0]
    field_y = field[1]
    field_z = field[2]

    dim = field.shape[1:]  # FIXME: why is the no. cells < no. points?

    vectors = np.empty(field_x.shape + (3,), dtype=float)
    vectors[..., 0] = field_x
    vectors[..., 1] = field_y
    vectors[..., 2] = field_z

    vectors = vectors.transpose(2, 1, 0, 3).copy()
    vectors.shape = vectors.size // 3, 3

    im_vtk = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=dim)
    im_vtk.point_data.vectors = vectors
    im_vtk.point_data.vectors.name = 'field'

    write_data(im_vtk, file_path)


def save_grid_to_disk(grid, file_path):
    grid = grid.cpu().numpy()
    
    x = grid[0, :, :, :]
    y = grid[1, :, :, :]
    z = grid[2, :, :, :]

    pts = np.empty(x.shape + (3,), dtype=float)

    pts[..., 0] = x
    pts[..., 1] = y
    pts[..., 2] = z

    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size // 3, 3

    sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
    write_data(sg, file_path)


def save_im_to_disk(im, file_path):
    im = nib.Nifti1Image(im, np.eye(4))
    im.to_filename(file_path)


def save_optimiser_to_disk(optimiser, file_path):
    state_dict = optimiser.state_dict()
    torch.save(state_dict, file_path)


def separable_conv_3d(field_out, *args, **kwargs):
    if len(args) == 2:
        kernel = args[0]
        padding_sz = args[1]

        N, C, D, H, W = field_out.size()

        padding_3d = (padding_sz, padding_sz, 0, 0, 0, 0)

        field_out = F.pad(field_out, padding_3d, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # depth
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # permute depth, height, and width

        field_out = F.pad(field_out, padding_3d, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # height
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))

        field_out = F.pad(field_out, padding_3d, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # width
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # back to the orig. dimensions

    elif len(args) == 4:
        kernel_x = args[0]
        kernel_y = args[1]
        kernel_z = args[2]

        padding_sz = args[3]

        padding = (padding_sz, padding_sz, padding_sz, padding_sz, padding_sz, padding_sz)
        field_out = F.pad(field_out, padding, mode='replicate')

        field_out = F.conv3d(field_out, kernel_z, groups=3)
        field_out = F.conv3d(field_out, kernel_y, groups=3)
        field_out = F.conv3d(field_out, kernel_x, groups=3)

    return field_out


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average', 'last_value'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
