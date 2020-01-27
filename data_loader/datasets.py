from os import listdir, path
from skimage import io, transform
from torch.utils.data import Dataset

from utils import rescale_im, standardise_im

import numpy as np
import SimpleITK as sitk
import torch


def init_mu_v(dims):
    return torch.zeros(dims)


def init_log_var_v(dims):
    len_voxel = float(dims[1] / 2.0)
    var_v = float(len_voxel ** (-2)) * torch.ones(dims)

    return torch.log(var_v)


def init_u_v(dims):
    return torch.zeros(dims)


def init_log_var_f(dims):
    var_f = (0.1 ** 2) * torch.ones(dims)
    return torch.log(var_f)


def init_u_f(dims):
    return torch.zeros(dims)


class BiobankDataset(Dataset):
    def __init__(self, im_paths, save_paths, dim_x, dim_y, dim_z):
        self.im_paths = im_paths
        self.save_paths = save_paths

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        self.dims_im = (1, self.dim_x, self.dim_y, self.dim_z)
        self.dims_v = (3, self.dim_x, self.dim_y, self.dim_z)

        # image filenames
        im_filenames = sorted([path.join(im_paths, f)
                               for f in listdir(im_paths) if path.isfile(path.join(im_paths, f))])

        # segmentation filenames
        seg_paths = path.join(im_paths, 'seg')

        if listdir(seg_paths):
            seg_filenames = sorted([path.join(seg_paths, f)
                                    for f in listdir(seg_paths) if path.isfile(path.join(seg_paths, f))])
        else:
            seg_filenames = ['' for _ in range(len(im_filenames))]

        # mask filenames
        mask_paths = path.join(im_paths, 'masks')

        if listdir(mask_paths):
            mask_filenames = sorted([path.join(mask_paths, f)
                                    for f in listdir(mask_paths) if path.isfile(path.join(mask_paths, f))])
        else:
            mask_filenames = ['' for _ in range(len(im_filenames))]

        im_seg_pairs = list(zip(im_filenames, seg_filenames, mask_filenames))

        # all-to-one
        atlas = im_seg_pairs[0]
        self.im_seg_pairs = []

        for other_im_seg_pair in im_seg_pairs:
            if other_im_seg_pair == atlas:
                continue

            self.im_seg_pairs.append((atlas, other_im_seg_pair))

        # pre-load im_fixed and the segmentation
        im_fixed_path = self.im_seg_pairs[0][0][0]
        seg_fixed_path = self.im_seg_pairs[0][0][1]
        mask_fixed_path = self.im_seg_pairs[0][0][2]

        im_fixed = sitk.ReadImage(im_fixed_path, sitk.sitkFloat32)
        im_fixed = torch.from_numpy(
            transform.resize(
                np.transpose(sitk.GetArrayFromImage(im_fixed), (2, 1, 0)), (self.dim_x, self.dim_y, self.dim_z)))

        if seg_fixed_path is not '':
            seg_fixed = sitk.ReadImage(seg_fixed_path, sitk.sitkFloat32)
            seg_fixed = torch.from_numpy(
                transform.resize(
                    np.transpose(sitk.GetArrayFromImage(seg_fixed), (2, 1, 0)),
                    (self.dim_x, self.dim_y, self.dim_z), order=0))

        if mask_fixed_path is not '':
            mask_fixed = sitk.ReadImage(mask_fixed_path, sitk.sitkFloat32)
            mask_fixed = torch.from_numpy(
                transform.resize(
                    np.transpose(sitk.GetArrayFromImage(mask_fixed), (2, 1, 0)),
                    (self.dim_x, self.dim_y, self.dim_z), order=0))

        im_fixed = standardise_im(im_fixed)  # standardise image
        im_fixed = rescale_im(im_fixed)  # rescale to range (-1, 1)

        self.im_fixed = im_fixed.unsqueeze(0)
        self.mask_fixed = mask_fixed.unsqueeze(0)
        self.seg_fixed = seg_fixed.unsqueeze(0)

    def __len__(self):
        return len(self.im_seg_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_seg_pair = self.im_seg_pairs[idx]

        im_moving_path = im_seg_pair[1][0]
        seg_moving_path = im_seg_pair[1][1]
        mask_moving_path = im_seg_pair[1][2]

        im_moving = sitk.ReadImage(im_moving_path, sitk.sitkFloat32)
        im_moving = torch.from_numpy(
            transform.resize(
                np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0)),
                (self.dim_x, self.dim_y, self.dim_z)))

        if seg_moving_path is not '':
            seg_moving = sitk.ReadImage(seg_moving_path, sitk.sitkFloat32)
            seg_moving = torch.from_numpy(
                transform.resize(
                    np.transpose(sitk.GetArrayFromImage(seg_moving), (2, 1, 0)),
                    (self.dim_x, self.dim_y, self.dim_z), order=0))

        if mask_moving_path is not '':
            mask_moving = sitk.ReadImage(mask_moving_path, sitk.sitkFloat32)
            mask_moving = torch.from_numpy(
                transform.resize(
                    np.transpose(sitk.GetArrayFromImage(mask_moving), (2, 1, 0)),
                    (self.dim_x, self.dim_y, self.dim_z), order=0))

        im_moving = standardise_im(im_moving)  # standardise image
        im_moving = rescale_im(im_moving)  # rescale to range (-1, 1)

        im_moving.unsqueeze_(0)
        seg_moving.unsqueeze_(0)

        assert self.im_fixed.shape == im_moving.shape, "images don't have the same dimensions"

        """
        if already exist then load them from a file, otherwise initialise new ones
        """

        # velocity field
        if path.exists(path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt')):
            mu_v = torch.load(path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt'))
        else:
            mu_v = init_mu_v(self.dims_v)

        # sigma_v
        if path.exists(path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt')):
            log_var_v = torch.load(path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt'))
        else:
            log_var_v = init_log_var_v(self.dims_v)
        # u_v
        if path.exists(path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt')):
            u_v = torch.load(path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt'))
        else:
            u_v = init_u_v(self.dims_v)

        # sigma_f
        if path.exists(path.join(self.save_paths['log_var_f'], 'log_var_f.pt')):
            log_var_f = torch.load(path.join(self.save_paths['log_var_f'], 'log_var_f.pt'))
        else:
            log_var_f = init_log_var_f(self.dims_im)
        # u_f
        if path.exists(path.join(self.save_paths['u_f'], 'u_f.pt')):
            u_f = torch.load(path.join(self.save_paths['u_f'], 'u_f.pt'))
        else:
            u_f = init_u_f(self.dims_im)

        return idx, self.im_fixed, self.seg_fixed, self.mask_fixed, \
               im_moving, seg_moving, mu_v, log_var_v, u_v, log_var_f, u_f
