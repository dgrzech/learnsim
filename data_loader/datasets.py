from utils import get_omega_norm_sq

from os import listdir, path
from torch.utils.data import Dataset

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def init_mu_hat(dims):
    return torch.zeros(dims)


def init_log_var_hat(dims):
    dims_omega = (dims[1] + 1, dims[2] + 1, dims[3] + 1, dims[4])
    omega_norm_sq = get_omega_norm_sq(dims_omega)[:, 1:, 1:, 1:]

    return torch.log(torch.ones(dims) * omega_norm_sq)


def init_u_hat(dims):
    return torch.zeros(dims)


class BiobankDataset(Dataset):
    def __init__(self, im_paths, save_paths, dim_x, dim_y, dim_z):
        self.im_paths = im_paths
        self.save_paths = save_paths

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        self.dims_im = (1, self.dim_x, self.dim_y, self.dim_z)
        self.dims_v = (3, self.dim_x - 1, self.dim_y - 1, self.dim_z // 2, 2)

        # image filenames
        im_filenames = sorted([path.join(im_paths, f)
                               for f in listdir(im_paths) if path.isfile(path.join(im_paths, f))])

        # mask filenames
        mask_paths = path.join(im_paths, 'masks')

        if listdir(mask_paths):
            mask_filenames = sorted([path.join(mask_paths, f)
                                     for f in listdir(mask_paths) if path.isfile(path.join(mask_paths, f))])
        else:
            mask_filenames = ['' for _ in range(len(im_filenames))]
        
        # segmentations filenames
        seg_paths = path.join(im_paths, 'seg')

        if listdir(seg_paths):
            seg_filenames = sorted([path.join(seg_paths, f)
                                    for f in listdir(seg_paths) if path.isfile(path.join(seg_paths, f))])
        else:
            seg_filenames = ['' for _ in range(len(im_filenames))]

        im_mask_seg_triples = list(zip(im_filenames, mask_filenames, seg_filenames))

        # all-to-one
        atlas = im_mask_seg_triples[0]
        self.im_mask_seg_triples = []

        for other_im_mask_seg_triple in im_mask_seg_triples:
            if other_im_mask_seg_triple == atlas:
                continue

            self.im_mask_seg_triples.append((atlas, other_im_mask_seg_triple))

        # pre-load im_fixed and the segmentation
        im_fixed_path = self.im_mask_seg_triples[0][0][0]
        mask_fixed_path = self.im_mask_seg_triples[0][0][1]
        seg_fixed_path = self.im_mask_seg_triples[0][0][2]

        im_fixed = sitk.ReadImage(im_fixed_path, sitk.sitkFloat32)
        im_fixed_arr = np.transpose(sitk.GetArrayFromImage(im_fixed), (2, 1, 0))

        longest_dim = max(im_fixed_arr.shape)
        self.pad_x = (longest_dim - im_fixed_arr.shape[0]) // 2
        self.pad_z = (longest_dim - im_fixed_arr.shape[2]) // 2

        # pad
        im_fixed_arr_padded = np.pad(im_fixed_arr,
                                     ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='minimum')
        # resize
        im_fixed = torch.from_numpy(im_fixed_arr_padded)
        im_fixed.unsqueeze_(0).unsqueeze_(0)

        im_fixed = F.interpolate(im_fixed,
                                 size=(self.dim_x, self.dim_y, self.dim_z), mode='trilinear', align_corners=True)

        if mask_fixed_path is not '':
            mask_fixed = sitk.ReadImage(mask_fixed_path, sitk.sitkFloat32)
            mask_fixed_arr = np.transpose(sitk.GetArrayFromImage(mask_fixed), (2, 1, 0))

            # pad
            mask_fixed_arr_padded = np.pad(mask_fixed_arr,
                                           ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='minimum')
            # resize
            mask_fixed = torch.from_numpy(mask_fixed_arr_padded)
            mask_fixed.unsqueeze_(0).unsqueeze_(0)

            mask_fixed = F.interpolate(mask_fixed, size=(self.dim_x, self.dim_y, self.dim_z), mode='nearest').bool()
        else:
            mask_fixed = torch.ones_like(im_fixed).bool()
        
        if seg_fixed_path is not '':
            seg_fixed = sitk.ReadImage(seg_fixed_path, sitk.sitkFloat32)
            seg_fixed_arr = np.transpose(sitk.GetArrayFromImage(seg_fixed), (2, 1, 0))

            # pad
            seg_fixed_arr_padded = np.pad(seg_fixed_arr,
                                           ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='minimum')
            # resize
            seg_fixed = torch.from_numpy(seg_fixed_arr_padded)
            seg_fixed.unsqueeze_(0).unsqueeze_(0)

            seg_fixed = F.interpolate(seg_fixed, size=(self.dim_x, self.dim_y, self.dim_z), mode='nearest').short()
        else:
            seg_fixed = torch.ones_like(im_fixed).short()

        self.im_fixed = im_fixed.squeeze(0)
        self.mask_fixed = mask_fixed.squeeze(0)
        self.seg_fixed = seg_fixed.squeeze(0)

        self.spacing = torch.tensor([longest_dim / self.dim_x, longest_dim / self.dim_y, longest_dim / self.dim_z],
                                    dtype=torch.float32)

    def __len__(self):
        return len(self.im_mask_seg_triples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_mask_seg_triple = self.im_mask_seg_triples[idx]

        im_moving_path = im_mask_seg_triple[1][0]
        mask_moving_path = im_mask_seg_triple[1][1]
        seg_moving_path = im_mask_seg_triple[1][2]

        im_moving = sitk.ReadImage(im_moving_path, sitk.sitkFloat32)
        im_moving_arr = np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0))

        # pad
        im_moving_arr_padded = np.pad(im_moving_arr,
                                      ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='minimum')
        # resize
        im_moving = torch.from_numpy(im_moving_arr_padded)
        im_moving.unsqueeze_(0).unsqueeze_(0)

        im_moving = F.interpolate(im_moving,
                                  size=(self.dim_x, self.dim_y, self.dim_z), mode='trilinear', align_corners=True)

        if mask_moving_path is not None:
            mask_moving = sitk.ReadImage(mask_moving_path, sitk.sitkFloat32)
            mask_moving_arr = np.transpose(sitk.GetArrayFromImage(mask_moving), (2, 1, 0))

            # pad
            mask_moving_arr_padded = np.pad(mask_moving_arr,
                                            ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='minimum')
            # resize
            mask_moving = torch.from_numpy(mask_moving_arr_padded)
            mask_moving.unsqueeze_(0).unsqueeze_(0)

            mask_moving = F.interpolate(mask_moving, size=(self.dim_x, self.dim_y, self.dim_z), mode='nearest').bool()
        else:
            mask_moving = torch.ones_like(im_moving).bool()
        
        if seg_moving_path is not None:
            seg_moving = sitk.ReadImage(seg_moving_path, sitk.sitkFloat32)
            seg_moving_arr = np.transpose(sitk.GetArrayFromImage(seg_moving), (2, 1, 0))

            # pad
            seg_moving_arr_padded = np.pad(seg_moving_arr,
                                           ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='minimum')
            # resize
            seg_moving = torch.from_numpy(seg_moving_arr_padded)
            seg_moving.unsqueeze_(0).unsqueeze_(0)

            seg_moving = F.interpolate(seg_moving, size=(self.dim_x, self.dim_y, self.dim_z), mode='nearest').short()
        else:
            seg_moving = torch.ones_like(im_moving).short()

        im_moving.squeeze_(0)
        mask_moving.squeeze_(0)
        seg_moving.squeeze_(0)

        assert self.im_fixed.shape == im_moving.shape, "images don't have the same dimensions"

        mu_hat = init_mu_hat(self.dims_v)
        log_var_hat = init_log_var_hat(self.dims_v)
        u_hat = init_u_hat(self.dims_v)

        return idx, self.im_fixed, self.mask_fixed, self.seg_fixed, im_moving, mask_moving, seg_moving, mu_hat, log_var_hat, u_hat
