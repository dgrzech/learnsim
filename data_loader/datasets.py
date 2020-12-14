from os import listdir, path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BiobankDataset(Dataset):
    def __init__(self, im_paths, save_paths, dim_x, dim_y, dim_z):
        self.im_paths = im_paths
        self.save_paths = save_paths

        self.dims = (dim_x, dim_y, dim_z)
        self.dims_im = (1, dim_x, dim_y, dim_z)
        self.dims_v = (3, dim_x, dim_y, dim_z)

        self.padding, self.spacing = None, None

        # image filenames
        im_filenames = self._get_filenames(im_paths)
        mask_filenames = self._get_filenames(path.join(im_paths, 'masks'))
        seg_filenames = self._get_filenames(path.join(im_paths, 'segs'))

        # all-to-one
        self.im_mask_seg_fixed_triples = list()

        for triple in list(zip(im_filenames, mask_filenames, seg_filenames)):
            self.im_mask_seg_fixed_triples.append({'im': triple[0], 'mask': triple[1], 'seg': triple[2]})

        # pre-load im_fixed and the segmentation
        im_fixed_path = self.im_mask_seg_fixed_triples[0]['im']
        mask_fixed_path = self.im_mask_seg_fixed_triples[0]['mask']
        seg_fixed_path = self.im_mask_seg_fixed_triples[0]['seg']

        self.im_fixed = self.get_image(im_fixed_path)
        self.mask_fixed = self.get_mask(mask_fixed_path)
        self.seg_fixed = self.get_seg(seg_fixed_path)

    def __len__(self):
        return 1

    @staticmethod
    def _get_filenames(p):
        if listdir(p):
            return sorted([path.join(p, f) for f in listdir(p) if path.isfile(path.join(p, f))])

        return ['' for _ in range(2)]

    @staticmethod
    def _init_mu_v(dims):
        return torch.zeros(dims)

    @staticmethod
    def _init_log_var_v(dims):
        var_v = (0.5 ** 2) * torch.ones(dims)
        return torch.log(var_v)

    @staticmethod
    def _init_u_v(dims):
        return torch.zeros(dims)

    def get_image(self, im_path):
        im = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im_arr = np.transpose(sitk.GetArrayFromImage(im), (2, 1, 0))

        if self.spacing is None:
            self.spacing = torch.tensor(max(im_arr.shape) / np.asarray(self.dims), dtype=torch.float32)
        if self.padding is None:
            padding = (max(im_arr.shape) - np.asarray(im_arr.shape)) // 2
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))

        # pad
        im_arr_padded = np.pad(im_arr, self.padding, mode='minimum')
        im = torch.from_numpy(im_arr_padded).unsqueeze(0).unsqueeze(0)

        return F.interpolate(im, size=self.dims, mode='trilinear', align_corners=True).squeeze(0)

    def get_mask(self, mask_path):
        if mask_path is '':
            return torch.ones_like(self.im_fixed).bool()

        mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask), (2, 1, 0))

        # pad
        mask_arr_padded = np.pad(mask_arr, self.padding, mode='minimum')
        mask = torch.from_numpy(mask_arr_padded).unsqueeze(0).unsqueeze(0)

        return F.interpolate(mask, size=self.dims, mode='nearest').bool().squeeze(0)

    def get_seg(self, seg_path):
        if seg_path is '':
            return torch.ones_like(self.im_fixed).short()

        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        seg_arr = np.transpose(sitk.GetArrayFromImage(seg), (2, 1, 0))

        # pad
        seg_arr_padded = np.pad(seg_arr, self.padding, mode='minimum')
        seg = torch.from_numpy(seg_arr_padded).unsqueeze(0).unsqueeze(0)

        return F.interpolate(seg, size=self.dims, mode='nearest').short().squeeze(0)

    def __getitem__(self, idx):
        im_moving_path = self.im_mask_seg_fixed_triples[1]['im']
        mask_moving_path = self.im_mask_seg_fixed_triples[1]['mask']
        seg_moving_path = self.im_mask_seg_fixed_triples[1]['seg']

        im_moving = self.get_image(im_moving_path)
        mask_moving = self.get_mask(mask_moving_path)
        seg_moving = self.get_seg(seg_moving_path)

        assert self.im_fixed.shape == im_moving.shape, 'images don\'t have the same dimensions'

        mu_v = self._init_mu_v(self.dims_v)
        log_var_v = self._init_log_var_v(self.dims_v)
        u_v = self._init_u_v(self.dims_v)

        return idx, self.im_fixed, self.mask_fixed, self.seg_fixed, im_moving, mask_moving, seg_moving, \
               mu_v, log_var_v, u_v
