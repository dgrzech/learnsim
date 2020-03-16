from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from torch import nn
from tvtk.api import tvtk, write_data

import json
import math
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
    wrapper function for endless data loader
    """

    for loader in repeat(data_loader):
        yield from loader


def calc_det_J(nabla_x, nabla_y, nabla_z):
    """
    calculate the Jacobian determinant of a vector field

    :param nabla_x: field gradients in the x direction
    :param nabla_y: field gradients in the y direction
    :param nabla_z: field gradients in the z direction

    :return: Jacobian determinant
    """

    det_J = nabla_x[:, 0] * nabla_y[:, 1] * nabla_z[:, 2] + \
            nabla_y[:, 0] * nabla_z[:, 1] * nabla_x[:, 2] + \
            nabla_z[:, 0] * nabla_x[:, 1] * nabla_y[:, 2] - \
            nabla_x[:, 2] * nabla_y[:, 1] * nabla_z[:, 0] - \
            nabla_y[:, 2] * nabla_z[:, 1] * nabla_x[:, 0] - \
            nabla_z[:, 2] * nabla_x[:, 1] * nabla_y[:, 0]

    return det_J


def calc_asd(seg_fixed, seg_moving, structures_dict, spacing):
    """
    calculate the symmetric average surface distance
    """

    asd = dict()  # dict. with the output for each segmentation
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    seg_fixed_arr = seg_fixed.squeeze().cpu().numpy()
    seg_moving_arr = seg_moving.squeeze().cpu().numpy()
    spacing = spacing.numpy().tolist()
    
    for structure in structures_dict:
        label = structures_dict[structure]

        seg_fixed_structure, seg_moving_structure = np.where(seg_fixed_arr == label, 1, 0), \
                                                    np.where(seg_moving_arr == label, 1, 0)
        seg_fixed_im, seg_moving_im = sitk.GetImageFromArray(seg_fixed_structure), \
                                      sitk.GetImageFromArray(seg_moving_structure)

        seg_fixed_im.SetSpacing(spacing)
        seg_moving_im.SetSpacing(spacing)

        seg_fixed_contour, seg_moving_contour = sitk.LabelContour(seg_fixed_im), \
                                                sitk.LabelContour(seg_moving_im)

        hausdorff_distance_filter.Execute(seg_fixed_contour, seg_moving_contour)
        asd[structure] = hausdorff_distance_filter.GetAverageHausdorffDistance()

    return asd


def calc_dice(seg_fixed, seg_moving, structures_dict):
    dsc = dict()  # dict. with dice scores for each segmentation
    
    for structure in structures_dict:
        label = structures_dict[structure]

        numerator = 2.0 * ((seg_fixed == label) * (seg_moving == label)).sum().item()
        denominator = (seg_fixed == label).sum().item() + (seg_moving == label).sum().item()
    
        score = numerator / denominator
        dsc[structure] = score
    
    return dsc


def calc_norm(v):
    """
    calculate the voxel-wise norm of vectors in a 3D field
    """
    return torch.norm(v, p=2, dim=0, keepdim=True)


def get_module_attr(module, name):
    if isinstance(module, nn.DataParallel):
        return getattr(module.module, name)

    return getattr(module, name)


def init_identity_grid_2d(nx, ny):
    """
    initialise a 2D identity grid

    :param nx: number of voxels in the x direction
    :param ny: number of voxels in the y direction
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
    initialise a 3D identity grid

    :param nx: number of voxels in the x direction
    :param ny: number of voxels in the y direction
    :param nz: number of voxels in the z direction
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


def max_field_update(field_old, field_new):
    """
    calculate the largest voxel-wise update to a vector field in terms of the L2 norm

    :param field_old: vector field before the update
    :param field_new: vector field after the update

    :return: voxel index and value of the largest update
    """

    norm_old = calc_norm(field_old)
    norm_new = calc_norm(field_new)

    max_update = torch.max(torch.abs(norm_new - norm_old))
    max_update_idx = torch.argmax(torch.abs(norm_new - norm_old))
    return max_update, max_update_idx


def pixel_to_normalised_2d(px_idx_x, px_idx_y, dim_x, dim_y):
    """
    transform the coordinates of a pixel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)

    return x, y


def pixel_to_normalised_3d(px_idx_x, px_idx_y, px_idx_z, dim_x, dim_y, dim_z):
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


def rescale_residuals(res, mask, data_loss):
    """
    rescale residuals by the estimated voxel-wise standard deviation
    """

    res_masked = torch.where(mask, res, torch.zeros_like(res))
    res_masked_flattened = res_masked.view(1, -1, 1)

    scaled_res = res_masked_flattened * torch.exp(-data_loss.log_std)
    scaled_res.requires_grad_(True)
    scaled_res.retain_grad()

    loss_vd = -1.0 * torch.sum(data_loss.log_pdf_vd(scaled_res))
    loss_vd.backward()

    return torch.sum(scaled_res * scaled_res.grad, dim=-1).view(res.shape)


def save_field_to_disk(field, file_path, spacing=(1, 1, 1)):
    """
    save a vector field to a .vtk file

    :param field: field to save
    :param file_path: path to use
    :param spacing: voxel spacing
    """

    field = field.cpu().numpy()
    spacing = spacing.numpy()

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

    im_vtk = tvtk.ImageData(spacing=spacing, origin=(0, 0, 0), dimensions=dim)
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


def separable_conv_3d(field, *args):
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
    dims = field.shape[2:]

    for idx in range(3):
        field_out[:, idx] = field_out[:, idx] * 2.0 / float(dims[idx] - 1)

    return field_out


def transform_coordinates_inv(field):
    """
    coordinate transformation from normalised coordinates (-1, ..., 1) to absolute (0, 1, ..., n)
    """

    field_out = field.clone()
    dims = field.shape[2:]

    for idx in range(3):
        field_out[:, idx] = field_out[:, idx] * float(dims[idx] - 1) / 2.0

    return field_out


def vd(residual, mask):
    """
    virtual decimation

    input x = residual (Gaussian-ish) field with stationary covariance, e.g. residual map (I-J) / sigma,
    where sigma is the noise sigma if you use SSD/Gaussian model or else the EM voxel-wise estimate if you use a GMM.

    EM voxel-wise estimate of precision = var^(-1) is sum_k rho_k precision_k,
    where rho_k is the component responsible for the voxel.

    The general idea is that each voxel-wise observation now only counts for "VD < 1 of an observation";
    imagine sampling a z ~ bernoulli(VD) at each voxel and you only add the voxel's loss if z == 1.

    In practice you do that in expectation. In the simplest case it looks like VD * data_loss,
    and goes well in a VB framework, as if you added a q(z) = Bernoulli(VD) to a VB approximation
    and took the expectation wrt q(z).
    """

    with torch.no_grad():
        # variance
        residual_masked = residual[mask]
        var_res = torch.mean(residual_masked ** 2)

        # covariance..
        no_unmasked_voxels = torch.sum(mask)
        residual_masked = torch.where(mask, residual, torch.zeros_like(residual))

        cov_x = torch.sum(residual_masked[:, :, :-1] * residual_masked[:, :, 1:]) / no_unmasked_voxels
        cov_y = torch.sum(residual_masked[:, :, :, :-1] * residual_masked[:, :, :, 1:]) / no_unmasked_voxels
        cov_z = torch.sum(residual_masked[:, :, :, :, :-1] * residual_masked[:, :, :, :, 1:]) / no_unmasked_voxels

        corr_x = cov_x / var_res
        corr_y = cov_y / var_res
        corr_z = cov_z / var_res

        sq_vd_x = torch.clamp(-2.0 / math.pi * torch.log(corr_x), max=1.0)
        sq_vd_y = torch.clamp(-2.0 / math.pi * torch.log(corr_y), max=1.0)
        sq_vd_z = torch.clamp(-2.0 / math.pi * torch.log(corr_z), max=1.0)

        return torch.sqrt(sq_vd_x * sq_vd_y * sq_vd_z)


def vd_reg(nabla_vx, nabla_vy, nabla_vz, mask):
    with torch.no_grad():
        mask_stacked = torch.cat((mask, mask, mask), dim=1)

        # variance
        nabla_vx_masked = nabla_vx[mask_stacked]
        nabla_vy_masked = nabla_vy[mask_stacked]
        nabla_vz_masked = nabla_vz[mask_stacked]

        var_nabla_vx = torch.mean(nabla_vx_masked ** 2)
        var_nabla_vy = torch.mean(nabla_vy_masked ** 2)
        var_nabla_vz = torch.mean(nabla_vz_masked ** 2)

        # covariance..
        no_unmasked_voxels = torch.sum(mask_stacked)

        nabla_vx_masked = torch.where(mask_stacked, nabla_vx, torch.zeros_like(nabla_vx))
        nabla_vy_masked = torch.where(mask_stacked, nabla_vy, torch.zeros_like(nabla_vy))
        nabla_vz_masked = torch.where(mask_stacked, nabla_vz, torch.zeros_like(nabla_vz))

        cov_nabla_vx_x = torch.sum(nabla_vx_masked[:, :, :-1] * nabla_vx_masked[:, :, 1:]) / no_unmasked_voxels
        cov_nabla_vx_y = torch.sum(nabla_vx_masked[:, :, :, :-1] * nabla_vx_masked[:, :, :, 1:]) / no_unmasked_voxels
        cov_nabla_vx_z = \
            torch.sum(nabla_vx_masked[:, :, :, :, :-1] * nabla_vx_masked[:, :, :, :, 1:]) / no_unmasked_voxels

        cov_nabla_vy_x = torch.sum(nabla_vy_masked[:, :, :-1] * nabla_vy_masked[:, :, 1:]) / no_unmasked_voxels
        cov_nabla_vy_y = torch.sum(nabla_vy_masked[:, :, :, :-1] * nabla_vy_masked[:, :, :, 1:]) / no_unmasked_voxels
        cov_nabla_vy_z = \
            torch.sum(nabla_vy_masked[:, :, :, :, :-1] * nabla_vy_masked[:, :, :, :, 1:]) / no_unmasked_voxels

        cov_nabla_vz_x = torch.sum(nabla_vz_masked[:, :, :-1] * nabla_vz_masked[:, :, 1:]) / no_unmasked_voxels
        cov_nabla_vz_y = torch.sum(nabla_vz_masked[:, :, :, :-1] * nabla_vz_masked[:, :, :, 1:]) / no_unmasked_voxels
        cov_nabla_vz_z = \
            torch.sum(nabla_vz_masked[:, :, :, :, :-1] * nabla_vz_masked[:, :, :, :, 1:]) / no_unmasked_voxels

        corr_vx_x = cov_nabla_vx_x / var_nabla_vx
        corr_vx_y = cov_nabla_vx_y / var_nabla_vx
        corr_vx_z = cov_nabla_vx_z / var_nabla_vx

        corr_vy_x = cov_nabla_vy_x / var_nabla_vy
        corr_vy_y = cov_nabla_vy_y / var_nabla_vy
        corr_vy_z = cov_nabla_vy_z / var_nabla_vy

        corr_vz_x = cov_nabla_vz_x / var_nabla_vz
        corr_vz_y = cov_nabla_vz_y / var_nabla_vz
        corr_vz_z = cov_nabla_vz_z / var_nabla_vz

        sq_vd_nabla_vx_x = torch.clamp(-2.0 / math.pi * torch.log(corr_vx_x), max=1.0)
        sq_vd_nabla_vx_y = torch.clamp(-2.0 / math.pi * torch.log(corr_vx_y), max=1.0)
        sq_vd_nabla_vx_z = torch.clamp(-2.0 / math.pi * torch.log(corr_vx_z), max=1.0)

        sq_vd_nabla_vy_x = torch.clamp(-2.0 / math.pi * torch.log(corr_vy_x), max=1.0)
        sq_vd_nabla_vy_y = torch.clamp(-2.0 / math.pi * torch.log(corr_vy_y), max=1.0)
        sq_vd_nabla_vy_z = torch.clamp(-2.0 / math.pi * torch.log(corr_vy_z), max=1.0)

        sq_vd_nabla_vz_x = torch.clamp(-2.0 / math.pi * torch.log(corr_vz_x), max=1.0)
        sq_vd_nabla_vz_y = torch.clamp(-2.0 / math.pi * torch.log(corr_vz_y), max=1.0)
        sq_vd_nabla_vz_z = torch.clamp(-2.0 / math.pi * torch.log(corr_vz_z), max=1.0)

        vd_nabla_vx = torch.sqrt(sq_vd_nabla_vx_x * sq_vd_nabla_vx_y * sq_vd_nabla_vx_z)
        vd_nabla_vy = torch.sqrt(sq_vd_nabla_vy_x * sq_vd_nabla_vy_y * sq_vd_nabla_vy_z)
        vd_nabla_vz = torch.sqrt(sq_vd_nabla_vz_x * sq_vd_nabla_vz_y * sq_vd_nabla_vz_z)

        return (vd_nabla_vx + vd_nabla_vy + vd_nabla_vz) / 3.0


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
