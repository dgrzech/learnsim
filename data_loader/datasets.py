import json
import os
from os import listdir, path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import rescale_im


class BiobankDataset(Dataset):
    def __init__(self, dims, im_paths, save_paths, sigma_v_init, u_v_init, rescale_im=False, rank=0):
        self.im_paths = im_paths
        self.save_paths = save_paths
        self.rescale_im = rescale_im

        self.sigma_v_init, self.u_v_init = sigma_v_init, u_v_init

        self.dims, self.dims_im, self.dims_v = dims, (1, *dims), (3, *dims)
        self.padding, self.im_spacing = None, None

        # image filenames
        im_filenames = self._get_filenames(im_paths)
        mask_filenames = self._get_filenames(path.join(im_paths, 'masks'))
        seg_filenames = self._get_filenames(path.join(im_paths, 'segs'))

        # all-to-one
        self.im_mask_seg_triples = list()

        for triple in list(zip(im_filenames, mask_filenames, seg_filenames)):
            self.im_mask_seg_triples.append({'im': triple[0], 'mask': triple[1], 'seg': triple[2]})

        if rank == 0:
            txt_file_path = os.path.join(self.save_paths['dir'], 'idx_to_biobank_ID.json')

            with open(txt_file_path, 'w') as out:
                json.dump(dict(enumerate(self.im_mask_seg_triples)), out, indent=4, sort_keys=True)

        # pre-load the fixed image, the segmentation, and the mask
        fixed_triple = self.im_mask_seg_triples.pop(0)

        im_fixed_path = fixed_triple['im']
        mask_fixed_path = fixed_triple['mask']
        seg_fixed_path = fixed_triple['seg']

        im_fixed = self._get_image(im_fixed_path).unsqueeze(0)
        mask_fixed = self._get_mask(mask_fixed_path).unsqueeze(0)
        seg_fixed = self._get_seg(seg_fixed_path).unsqueeze(0)

        self.fixed = {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}

    def __len__(self):
        return len(self.im_mask_seg_triples)

    @staticmethod
    def _get_filenames(p):
        if listdir(p):
            return sorted([path.join(p, f) for f in listdir(p) if path.isfile(path.join(p, f))])

        return ['' for _ in range(2)]

    def _init_mu_v(self):
        return torch.zeros(self.dims_v)

    def _get_mu_v(self, idx):
        tensor_path = path.join(self.save_paths['tensors'], 'mu_v_' + str(idx) + '.pt')

        if path.exists(tensor_path):
            return torch.load(tensor_path)
        else:
            return self._init_mu_v()

    def _init_log_var_v(self):
        sigma_v = self.sigma_v_init
        var_v = (sigma_v ** 2) * torch.ones(self.dims_v)

        return torch.log(var_v)

    def _get_log_var_v(self, idx):
        tensor_path = path.join(self.save_paths['tensors'], 'log_var_v_' + str(idx) + '.pt')

        if path.exists(tensor_path):
            return torch.load(tensor_path)
        else:
            return self._init_log_var_v()

    def _init_u_v(self):
        u_v = self.u_v_init  * torch.ones(self.dims_v)
        return u_v

    def _get_u_v(self, idx):
        tensor_path = path.join(self.save_paths['tensors'], 'u_v_' + str(idx) + '.pt')

        if path.exists(tensor_path):
            return torch.load(tensor_path)
        else:
            return self._init_u_v()

    def _get_image(self, im_path):
        im = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im_arr = np.transpose(sitk.GetArrayFromImage(im), (2, 1, 0))

        if self.im_spacing is None:
            self.im_spacing = torch.tensor(max(im_arr.shape) / np.asarray(self.dims), dtype=torch.float32)
        if self.padding is None:
            padding = (max(im_arr.shape) - np.asarray(im_arr.shape)) // 2
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))

        # pad
        im_arr_padded = np.pad(im_arr, self.padding, mode='minimum')
        im_tensor = torch.from_numpy(im_arr_padded).unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im_tensor, size=self.dims, mode='trilinear', align_corners=True).squeeze(0)

        if self.rescale_im:
            im = rescale_im(im)

        return im

    def _get_mask(self, mask_path):
        mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask), (2, 1, 0))

        # pad
        mask_arr_padded = np.pad(mask_arr, self.padding, mode='minimum')
        mask = torch.from_numpy(mask_arr_padded).unsqueeze(0).unsqueeze(0)

        return F.interpolate(mask, size=self.dims, mode='nearest').bool().squeeze(0)  # NOTE (DG): no align_corners cause short

    def _get_seg(self, seg_path):
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        seg_arr = np.transpose(sitk.GetArrayFromImage(seg), (2, 1, 0))

        # pad
        seg_arr_padded = np.pad(seg_arr, self.padding, mode='minimum')
        seg = torch.from_numpy(seg_arr_padded).unsqueeze(0).unsqueeze(0)

        return F.interpolate(seg, size=self.dims, mode='nearest').short().squeeze(0)  # NOTE (DG): no align_corners cause short

    def __getitem__(self, idx):
        im_moving_path = self.im_mask_seg_triples[idx]['im']
        mask_moving_path = self.im_mask_seg_triples[idx]['mask']
        seg_moving_path = self.im_mask_seg_triples[idx]['seg']

        im_moving = self._get_image(im_moving_path)
        mask_moving = self._get_mask(mask_moving_path)
        seg_moving = self._get_seg(seg_moving_path)

        mu_v = self._get_mu_v(idx)
        log_var_v = self._get_log_var_v(idx)
        u_v = self._get_u_v(idx)

        moving = {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}
        var_params_q_v = {'mu': mu_v, 'log_var': log_var_v, 'u': u_v}

        return idx, moving, var_params_q_v
