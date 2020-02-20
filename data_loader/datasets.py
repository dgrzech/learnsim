from os import listdir, path
from torch.utils.data import Dataset

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def init_mu_v(dims):
    return torch.zeros(dims)


def init_log_var_v(dims):
    var_v = (0.5 ** 2) * torch.ones(dims)
    return torch.log(var_v)


def init_u_v(dims):
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

        # mask filenames
        mask_paths = path.join(im_paths, 'masks')

        if listdir(mask_paths):
            mask_filenames = sorted([path.join(mask_paths, f)
                                    for f in listdir(mask_paths) if path.isfile(path.join(mask_paths, f))])
        else:
            mask_filenames = ['' for _ in range(len(im_filenames))]

        im_mask_pairs = list(zip(im_filenames, mask_filenames))

        # all-to-one
        atlas = im_mask_pairs[0]
        self.im_mask_pairs = []

        for other_im_mask_pair in im_mask_pairs:
            if other_im_mask_pair == atlas:
                continue

            self.im_mask_pairs.append((atlas, other_im_mask_pair))

        # pre-load im_fixed and the segmentation
        im_fixed_path = self.im_mask_pairs[0][0][0]
        mask_fixed_path = self.im_mask_pairs[0][0][1]

        im_fixed = sitk.ReadImage(im_fixed_path, sitk.sitkFloat32)
        im_fixed_arr = np.transpose(sitk.GetArrayFromImage(im_fixed), (2, 1, 0))

        longest_dim = max(im_fixed_arr.shape)
        self.pad_x = (longest_dim - im_fixed_arr.shape[0]) // 2
        self.pad_z = (longest_dim - im_fixed_arr.shape[2]) // 2

        # pad
        im_fixed_arr_padded = np.pad(im_fixed_arr,
                                     ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='edge')
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
                                           ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='edge')
            # resize
            mask_fixed = torch.from_numpy(mask_fixed_arr_padded)
            mask_fixed.unsqueeze_(0).unsqueeze_(0)

            mask_fixed = F.interpolate(mask_fixed, size=(self.dim_x, self.dim_y, self.dim_z), mode='nearest').bool()
        else:
            mask_fixed = torch.ones_like(im_fixed).bool()

        self.im_fixed = im_fixed.squeeze(0)
        self.mask_fixed = mask_fixed.squeeze(0)

    def __len__(self):
        return len(self.im_mask_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_mask_pair = self.im_mask_pairs[idx]
        im_moving_path = im_mask_pair[1][0]

        im_moving = sitk.ReadImage(im_moving_path, sitk.sitkFloat32)
        im_moving_arr = np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0))

        # pad
        im_moving_arr_padded = np.pad(im_moving_arr,
                                     ((self.pad_x, self.pad_x), (0, 0), (self.pad_z, self.pad_z)), mode='edge')
        # resize
        im_moving = torch.from_numpy(im_moving_arr_padded)
        im_moving.unsqueeze_(0).unsqueeze_(0)

        im_moving = F.interpolate(im_moving,
                                  size=(self.dim_x, self.dim_y, self.dim_z), mode='trilinear', align_corners=True)

        im_moving.squeeze_(0)
        assert self.im_fixed.shape == im_moving.shape, "images don't have the same dimensions"

        mu_v = init_mu_v(self.dims_v)
        log_var_v = init_log_var_v(self.dims_v)
        u_v = init_u_v(self.dims_v)

        return idx, self.im_fixed, self.mask_fixed, im_moving, mu_v, log_var_v, u_v
