import json
from collections import OrderedDict
from pathlib import Path

import SimpleITK as sitk
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def add_noise_uniform_field(field, alpha):
    return field + transform_coordinates(get_noise_uniform(field.shape, field.device, alpha))


def add_noise_Langevin(field, sigma, tau):
    return field + get_noise_Langevin(sigma, tau)


def get_noise_uniform(shape, device, alpha):
    return -2.0 * alpha * torch.rand(shape, device=device) + alpha


def get_noise_Langevin(sigma, tau):
    eps = torch.randn_like(sigma)
    return math.sqrt(2.0 * tau) * sigma * eps


def get_control_grid_size(dims, cps):
    """
    calculate the control grid size
    :param dims: image dimensions
    :param cps: control point spacing
    """

    return tuple([int(math.ceil((sz - 1) / c) + 1 + 2) for sz, c in zip(dims, cps)])


def calc_det_J(nabla):
    """
    calculate the Jacobian determinant of a vector field

    :param nabla: field gradients
    :return: Jacobian determinant
    """

    # _, N, D, H, W, _ = nabla.shape
    # Jac = nabla.permute([0, 2, 3, 4, 1, 5]).reshape([-1, N, N])
    # return torch.det(Jac).reshape(-1, D, H, W)  # NOTE (DG): for some reason causes an illegal memory access

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

    DSC = torch.zeros(len(im_pair_idxs), len(structures_dict), device=seg_fixed.device)

    for idx, im_pair in enumerate(im_pair_idxs):
        seg_fixed_im_pair = seg_fixed[idx]
        seg_moving_im_pair = seg_moving[idx]

        for structure_idx, structure in enumerate(structures_dict):
            label = structures_dict[structure]

            numerator = 2.0 * ((seg_fixed_im_pair == label) * (seg_moving_im_pair == label)).sum()
            denominator = (seg_fixed_im_pair == label).sum() + (seg_moving_im_pair == label).sum()

            try:
                score = numerator / denominator
            except:
                score = 0.0

            DSC[idx, structure_idx] = score

    return DSC


@torch.no_grad()
def calc_metrics(im_pair_idxs, seg_fixed, seg_moving, structures_dict, spacing, GPU=True):
    """
    calculate average surface distances and Dice scores
    """

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    ASD, DSC = torch.zeros(len(im_pair_idxs), len(structures_dict), device=seg_fixed.device), \
               torch.zeros(len(im_pair_idxs), len(structures_dict), device=seg_fixed.device)

    if GPU:
        DSC = calc_DSC_GPU(im_pair_idxs, seg_fixed, seg_moving, structures_dict)
    else:
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    seg_fixed = seg_fixed.cpu().numpy()
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
        seg_fixed_arr = seg_fixed[loop_idx].squeeze()
        seg_moving_arr = seg_moving[loop_idx].squeeze()

        for structure_idx, structure_name in enumerate(structures_dict):
            label = structures_dict[structure_name]

            seg_fixed_structure = np.where(seg_fixed_arr == label, 1, 0)
            seg_moving_structure = np.where(seg_moving_arr == label, 1, 0)

            seg_fixed_im = sitk.GetImageFromArray(seg_fixed_structure)
            seg_moving_im = sitk.GetImageFromArray(seg_moving_structure)

            seg_fixed_im.SetSpacing(spacing)
            seg_moving_im.SetSpacing(spacing)

            try:
                ASD[loop_idx, structure_idx] = calc_ASD(seg_fixed_im, seg_moving_im)
            except:
                ASD[loop_idx, structure_idx] = np.inf

            if not GPU:
                DSC[loop_idx, structure_idx] = calc_DSC(seg_fixed_im, seg_moving_im)

    return ASD, DSC


def calc_no_non_diffeomorphic_voxels(transformation, diff_op):
    nabla = diff_op(transformation, transformation=True)
    log_det_J_transformation = torch.log(calc_det_J(nabla))
    return torch.sum(torch.isnan(log_det_J_transformation), dim=(1, 2, 3), dtype=torch.float), log_det_J_transformation


def get_log_path_from_run_ID(save_path, run_ID):
    return f'{save_path}/{run_ID}/log'


def get_module_attr(module, name):
    if isinstance(module, nn.DataParallel) or isinstance(module, nn.parallel.DistributedDataParallel):
        return getattr(module.module, name)

    return getattr(module, name)


def get_samples_path_from_run_ID(save_path, run_ID):
    return f'{save_path}/{run_ID}/samples'


def init_grid_im(size, spacing=2):
    if len(size) == 2:
        im = torch.zeros([1, 1, *size], dtype=torch.float)
        im[:, :, ::spacing, :] = 1
        im[:, :, :, ::spacing] = 1

        return im
    elif len(size) == 3:
        im = torch.zeros([1, 3, *size], dtype=torch.float)  # NOTE (DG): stack in the channel dimension

        im[:, 0, :, ::spacing, :] = 1
        im[:, 0, :, :, ::spacing] = 1

        im[:, 1, ::spacing, :, :] = 1
        im[:, 1, :, :, ::spacing] = 1

        im[:, 2, ::spacing, :, :] = 1
        im[:, 2, :, ::spacing, :] = 1

        return im

    raise NotImplementedError


def init_identity_grid_2D(dims):
    """
    initialise a 2D identity grid
    """

    nx, ny = dims[0], dims[1]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)

    grid_x, grid_y = torch.meshgrid(x, y)
    return torch.stack([grid_y, grid_x]).unsqueeze(0).float()


def init_identity_grid_3D(dims):
    """
    initialise a 3D identity grid
    """

    nx, ny, nz = dims[0], dims[1], dims[2]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)
    z = torch.linspace(-1, 1, steps=nz)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    return torch.stack([grid_z, grid_y, grid_x]).unsqueeze(0).float()


def max_field_update(field_old, field_new):
    """
    calculate the largest voxel-wise update to a vector field in terms of the L2 norm

    :param field_old: vector field before the update
    :param field_new: vector field after the update

    :return: voxel index and value of the largest update
    """

    norm_old = torch.norm(field_old, dim=1, keepdim=True, p=2)
    norm_new = torch.norm(field_new, dim=1, keepdim=True, p=2)

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


def rescale_im_intensity(im, range_min=0.0, range_max=1.0):
    """
    rescale the intensity of image pixels/voxels to a given range
    """

    im_min, im_max = torch.min(im), torch.max(im)
    return (range_max - range_min) * (im - im_min) / (im_max - im_min) + range_min


def separable_conv_3D(field, *args):
    """
    implements separable convolution over a three-dimensional image or vector field either as three 1D convolutions
    with a 1D kernel, or as three 3D convolutions with three 3D kernels of sizes kx1x1, 1xkx1, and 1x1xk

    :param field: input vector field
    :param args: input kernel(s) and the size of padding to use
    :return: input vector field convolved with the kernel
    """

    field_out = field.clone()
    ndim = field_out.shape[1]

    if len(args) == 2:
        kernel, padding_sz = args[0], args[1]
        padding_3D = (padding_sz, padding_sz, 0, 0, 0, 0)

        N, C, D, H, W = field_out.shape

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=ndim)  # depth
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # permute depth, height, and width

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=ndim)  # height
        field_out = field_out.reshape(N, C, H, W, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=ndim)  # width
        field_out = field_out.reshape(N, C, W, D, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # back to the orig. dimensions

    elif len(args) == 4:
        kernel_x, kernel_y, kernel_z = args[0], args[1], args[2]
        padding = args[3]

        field_out = F.pad(field_out, padding, mode='replicate')

        field_out = F.conv3d(field_out, kernel_z, groups=ndim)
        field_out = F.conv3d(field_out, kernel_y, groups=ndim)
        field_out = F.conv3d(field_out, kernel_x, groups=ndim)

    return field_out


def standardise_im_intensity(im):
    """
    standardise image to zero mean and unit variance
    """

    im_mean, im_std = torch.mean(im), torch.std(im)
    return (im - im_mean) / im_std


def transform_coordinates(field):
    """
    coordinate transformation from absolute coordinates (0, 1, ..., n) to normalised (-1, ..., 1)
    """

    field_out = field.clone()
    no_dims, dims = field.shape[1], field.shape[2:]

    for idx in range(no_dims):
        field_out[:, idx] = field_out[:, idx] * 2.0 / float(dims[idx] - 1)

    return field_out


def transform_coordinates_inv(field):
    """
    coordinate transformation from normalised coordinates (-1, ..., 1) to absolute (0, 1, ..., n)
    """

    field_out = field.clone()
    no_dims, dims = field.shape[1], field.shape[2:]

    for idx in range(no_dims):
        field_out[:, idx] = field_out[:, idx] * float(dims[idx] - 1) / 2.0

    return field_out
