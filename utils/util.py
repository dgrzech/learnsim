import json
import math
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


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
    wrapper function for endless data loader
    """

    for loader in repeat(data_loader):
        yield from loader


def add_noise_uniform(field, alpha):
    return field + get_noise_uniform(field, alpha)


def add_noise_Langevin(field, sigma, tau):
    return field + get_noise_Langevin(sigma, tau)


def get_noise_uniform(field, alpha):
    epsilon = -2.0 * alpha * torch.rand(field.shape, device=field.device) + alpha
    return transform_coordinates(epsilon)


def get_noise_Langevin(sigma, tau):
    eps = torch.randn(sigma.shape, device=sigma.device)
    return math.sqrt(2.0 * tau) * sigma * eps


def calc_det_J(nabla):
    """
    calculate the Jacobian determinant of a vector field

    :param nabla: field gradients
    :return: Jacobian determinant
    """

    nabla_x = nabla[..., 0]
    nabla_y = nabla[..., 1]
    nabla_z = nabla[..., 2]

    det_J = nabla_x[:, 0] * nabla_y[:, 1] * nabla_z[:, 2] + \
            nabla_y[:, 0] * nabla_z[:, 1] * nabla_x[:, 2] + \
            nabla_z[:, 0] * nabla_x[:, 1] * nabla_y[:, 2] - \
            nabla_x[:, 2] * nabla_y[:, 1] * nabla_z[:, 0] - \
            nabla_y[:, 2] * nabla_z[:, 1] * nabla_x[:, 0] - \
            nabla_z[:, 2] * nabla_x[:, 1] * nabla_y[:, 0]

    return det_J


@torch.no_grad()
def calc_DSC_GPU(im_pair_idxs, seg_fixed, seg_moving, structures_dict):
    """
    calculate the Dice scores
    """

    DSC_batch = dict()  # dict. with Dice scores for each image pair and segmentation

    for idx, im_pair in enumerate(im_pair_idxs):
        DSC = dict()  # dict. with Dice scores for the image pair

        seg_fixed_im_pair = seg_fixed[idx]
        seg_moving_im_pair = seg_moving[idx]

        for structure in structures_dict:
            label = structures_dict[structure]

            numerator = 2.0 * ((seg_fixed_im_pair == label) * (seg_moving_im_pair == label)).sum().item()
            denominator = (seg_fixed_im_pair == label).sum().item() + (seg_moving_im_pair == label).sum().item()

            score = numerator / denominator
            DSC[structure] = score

        DSC_batch[im_pair] = DSC

    return DSC_batch


@torch.no_grad()
def calc_metrics(im_pair_idxs, seg_fixed, seg_moving, structures_dict, spacing, GPU=True):
    """
    calculate average surface distances and Dice scores
    """

    metrics = dict()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    if GPU:
        DSCs = calc_DSC_GPU(im_pair_idxs, seg_fixed, seg_moving, structures_dict)
    else:
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    seg_fixed_arr = seg_fixed[0].squeeze().cpu().numpy()
    seg_moving = seg_moving.cpu().numpy()
    spacing = spacing.numpy().tolist()

    def calc_ASD(seg_fixed_im, seg_moving_im):
        seg_fixed_contour = sitk.LabelContour(seg_fixed_im)
        seg_moving_contour = sitk.LabelContour(seg_moving_im)

        hausdorff_distance_filter.Execute(seg_fixed_contour, seg_moving_contour)
        return hausdorff_distance_filter.GetAverageHausdorffDistance()

    def calc_DSC(seg_fixed_im, seg_moving_im):
        overlap_measures_filter.Execute(seg_fixed_im, seg_moving_im)
        return overlap_measures_filter.GetDiceCoefficient()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        seg_moving_arr = seg_moving[loop_idx].squeeze()
        metrics_im_pair = {'ASD': dict(), 'DSC': dict()}

        for structure in structures_dict:
            label = structures_dict[structure]

            seg_fixed_structure = np.where(seg_fixed_arr == label, 1, 0)
            seg_moving_structure = np.where(seg_moving_arr == label, 1, 0)

            seg_fixed_im = sitk.GetImageFromArray(seg_fixed_structure)
            seg_moving_im = sitk.GetImageFromArray(seg_moving_structure)

            seg_fixed_im.SetSpacing(spacing)
            seg_moving_im.SetSpacing(spacing)

            try:
                metrics_im_pair['ASD'][structure] = calc_ASD(seg_fixed_im, seg_moving_im)
            except:
                metrics_im_pair['ASD'][structure] = np.inf

            if GPU:
                metrics_im_pair['DSC'][structure] = DSCs[im_pair_idx][structure]
            else:
                metrics_im_pair['DSC'][structure] = calc_DSC(seg_fixed_im, seg_moving_im)

        metrics[im_pair_idx] = metrics_im_pair

    return metrics


def calc_no_non_diffeomorphic_voxels(transformation, diff_op):
    nabla = diff_op(transformation, transformation=True)
    log_det_J_transformation = torch.log(calc_det_J(nabla))
    return torch.sum(torch.isnan(log_det_J_transformation), dim=(1, 2, 3)), log_det_J_transformation


def calc_norm(field):
    """
    calculate the voxel-wise norm of vectors in a batch of 3D fields
    """

    norms = torch.empty(size=(field.shape[0], 1, field.shape[2], field.shape[3], field.shape[4]), device=field.device)

    for batch_idx in range(field.shape[0]):
        norms[batch_idx, ...] = torch.norm(field[batch_idx], p=2, dim=0)

    return norms


def get_module_attr(module, name):
    if isinstance(module, nn.DataParallel):
        return getattr(module.module, name)

    return getattr(module, name)


def im_flip(array):
    return np.fliplr(np.flipud(np.transpose(array, (1, 0))))


def init_identity_grid_2D(dims):
    """
    initialise a 2D identity grid
    """

    nx, ny = dims[0], dims[1]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)

    x = x.expand(ny, -1).unsqueeze(0).unsqueeze(3)
    y = y.expand(nx, -1).transpose(0, 1).unsqueeze(0).unsqueeze(3)

    return torch.cat((x, y), 3)


def init_identity_grid_3D(dims):
    """
    initialise a 3D identity grid
    """

    nx, ny, nz = dims[0], dims[1], dims[2]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)
    z = torch.linspace(-1, 1, steps=nz)

    x = x.expand(ny, -1).expand(nz, -1, -1).unsqueeze(0).unsqueeze(4)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2).unsqueeze(0).unsqueeze(4)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1).unsqueeze(0).unsqueeze(4)

    return torch.cat((x, y, z), 4)


def max_field_update(field_old, field_new):
    """
    calculate the largest voxel-wise update to a vector field in terms of the L2 norm

    :param field_old: vector field before the update
    :param field_new: vector field after the update

    :return: voxel index and value of the largest update
    """

    norm_old = calc_norm(field_old)
    norm_new = calc_norm(field_new)

    diff = torch.abs(norm_new - norm_old)
    return torch.max(diff), torch.argmax(diff)


def pixel_to_normalised_2D(px_idx_x, px_idx_y, dim_x, dim_y):
    """
    transform the coordinates of a pixel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)

    return x, y


def pixel_to_normalised_3D(px_idx_x, px_idx_y, px_idx_z, dim_x, dim_y, dim_z):
    """
    transform the coordinates of a voxel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)
    z = -1.0 + 2.0 * px_idx_z / (dim_z - 1.0)

    return x, y, z


def rescale_im(im, range_min=-1.0, range_max=1.0):
    """
    rescale the intensity of image pixels/voxels to a given range
    """

    im_min, im_max = torch.min(im), torch.max(im)
    return (range_max - range_min) * (im - im_min) / (im_max - im_min) + range_min


def separable_conv_3D(field, *args):
    """
    implements separable convolution over a three-dimensional vector field either as three 1D convolutions
    with a 1D kernel, or as three 3D convolutions with three 3D kernels of sizes kx1x1, 1xkx1, and 1x1xk

    :param field: input vector field
    :param args: input kernel(s) and the size of padding to use
    :return: input vector field convolved with the kernel
    """

    field_out = field.clone()

    if len(args) == 2:
        kernel = args[0]
        padding_sz = args[1]

        N, C, D, H, W = field_out.shape

        padding_3D = (padding_sz, padding_sz, 0, 0, 0, 0)

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # depth
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # permute depth, height, and width

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # height
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # width
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # back to the orig. dimensions

    elif len(args) == 4:
        kernel_x = args[0]
        kernel_y = args[1]
        kernel_z = args[2]
        padding = args[3]

        field_out = F.pad(field_out, padding, mode='replicate')

        field_out = F.conv3d(field_out, kernel_z, groups=3)
        field_out = F.conv3d(field_out, kernel_y, groups=3)
        field_out = F.conv3d(field_out, kernel_x, groups=3)

    return field_out


def standardise_im(im):
    """
    standardise image to zero mean and unit variance
    """

    im_mean, im_std = torch.mean(im), torch.std(im)

    im -= im_mean
    im /= im_std

    return im


def transform_coordinates(field):
    """
    coordinate transformation from absolute coordinates (0, 1, ..., n) to normalised (-1, ..., 1)
    """

    field_out = field.clone()

    no_dims = field.shape[1]
    dims = field.shape[2:]

    for idx in range(no_dims):
        field_out[:, idx] = field_out[:, idx] * 2.0 / float(dims[idx] - 1)

    return field_out


def transform_coordinates_inv(field):
    """
    coordinate transformation from normalised coordinates (-1, ..., 1) to absolute (0, 1, ..., n)
    """

    field_out = field.clone()

    no_dims = field.shape[1]
    dims = field.shape[2:]

    for idx in range(no_dims):
        field_out[:, idx] = field_out[:, idx] * float(dims[idx] - 1) / 2.0

    return field_out


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
            self.writer.add_scalar(key, value / n)

        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_ASD_and_DSC(self, metrics_im_pairs, test=False):
        for im_pair_idx in metrics_im_pairs:
            ASDs = metrics_im_pairs[im_pair_idx]['ASD']
            DSCs = metrics_im_pairs[im_pair_idx]['DSC']

            for structure in ASDs:
                name = 'ASD/im_pair_' + str(im_pair_idx) + '/' + structure if not test else 'test/ASD/im_pair_' + str(im_pair_idx) + '/' + structure
                self.update(name, ASDs[structure])
            for structure in DSCs:
                name = 'DSC/im_pair_' + str(im_pair_idx) + '/' + structure if not test else 'test/DSC/im_pair_' + str(im_pair_idx) + '/' + structure
                self.update(name, DSCs[structure])

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
