from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import json
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


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


def compute_norm(v):
    return torch.norm(v, p=2, dim=1, keepdim=True)


def init_identity_grid_2d(nx, ny):
    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)

    x = x.expand(ny, -1)
    y = y.expand(nx, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(3)
    y.unsqueeze_(0).unsqueeze_(3)

    return torch.cat((x, y), 3)


def init_identity_grid_3d(nx, ny, nz):
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
    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)

    return x, y


def pixel_to_normalised_3d(px_idx_x, px_idx_y, px_idx_z, dim_x, dim_y, dim_z):
    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)
    z = -1.0 + 2.0 * px_idx_z / (dim_z - 1.0)

    return x, y, z


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
    im_min, im_max = torch.min(im), torch.max(im)

    im = (range_max - range_min) * (im - im_min) / (im_max - im_min) + range_min
    return im


def standardise_im(im):
    im_mean, im_std = torch.mean(im), torch.std(im)
    
    im -= im_mean
    im /= im_std

    return im


def save_im_to_disk(im, file_path):
    im = im[0, 0, :, :, :].cpu().numpy()
    im = nib.Nifti1Image(im, np.eye(4))
    im.to_filename(file_path)


def save_field_to_disk(field, file_path):
    field = field[0, :, :, :, :].cpu().numpy()
    field = nib.Nifti1Image(field, np.eye(4))
    field.to_filename(file_path)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
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
