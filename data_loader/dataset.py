import json
import os
from abc import abstractmethod
from os import path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import get_control_grid_size, rescale_im


class ImageRegistrationDataset(Dataset):
    def __init__(self, dims, im_paths, save_paths, im_mask_seg_triples, structures_dict, sigma_v_init, u_v_init,
                pad=False, rescale=True, resize=False, cps=None):
        self.im_paths, self.save_paths = im_paths, save_paths
        self.im_mask_seg_triples = im_mask_seg_triples

        self.im_spacing, self.padding = None, None
        self.structures_dict = structures_dict
        self.pad, self.rescale, self.resize = pad, rescale, resize

        self.dims, self.dims_im, = dims, (1, *dims)

        if cps is None:
            self.dims_v = (3, *dims)
        else:
            control_grid_sz = get_control_grid_size(dims, cps)
            self.dims_v = (3, *control_grid_sz)

        self.sigma_v_init, self.u_v_init = sigma_v_init, u_v_init

    def __len__(self):
        return len(self.im_mask_seg_triples)

    @property
    @abstractmethod
    def atlas_mode(self):
        pass

    def _get_image(self, im_path):
        im = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im_arr = sitk.GetArrayFromImage(im)
        im_spacing = im.GetSpacing()

        if self.pad:
            if self.im_spacing is None:
                self.im_spacing = torch.tensor(max(im_arr.shape) / np.asarray(self.dims), dtype=torch.float32)
            if self.padding is None:
                padding = (max(im_arr.shape) - np.asarray(im_arr.shape)) // 2
                self.padding = ((padding[0],) * 2, (padding[1],) * 2, (padding[2],) * 2)

            im_arr = np.pad(np.transpose(im_arr, (2, 1, 0)), self.padding, mode='minimum')

        im = torch.from_numpy(im_arr).unsqueeze(0)

        if self.resize:
            im = F.interpolate(im.unsqueeze(0), size=self.dims, mode='trilinear', align_corners=True).squeeze(0)

        if self.rescale:
            im = rescale_im(im)

        return im, im_spacing

    def _get_mask(self, mask_path):
        mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask_arr = np.transpose(sitk.GetArrayFromImage(mask), (2, 1, 0))

        # pad
        mask_arr_padded = np.pad(mask_arr, self.padding, mode='minimum')
        mask = torch.from_numpy(mask_arr_padded).unsqueeze(0).unsqueeze(0)

        return F.interpolate(mask, size=self.dims, mode='nearest').bool().squeeze(0)  # NOTE (DG): no align_corners cause short

    def _get_seg(self, seg_path):
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        seg_arr = sitk.GetArrayFromImage(seg)

        if self.pad:
            seg_arr = np.pad(np.transpose(seg_arr, (2, 1, 0)), self.padding, mode='minimum')

        seg = torch.from_numpy(seg_arr).unsqueeze(0)

        if self.resize:
            seg = F.interpolate(seg.unsqueeze(0), size=self.dims, mode='nearest').squeeze(0)  # NOTE (DG): no align_corners cause short

        return seg.short()

    def _init_mu_v(self):
        return torch.zeros(self.dims_v)

    def _init_log_var_v(self):
        sigma_v = self.sigma_v_init
        var_v = (sigma_v ** 2) * torch.ones(self.dims_v)

        return var_v.log()

    def _init_u_v(self):
        return self.u_v_init * torch.ones(self.dims_v)

    def _get_mu_v(self, idx):
        tensor_path = path.join(self.save_paths['tensors'], f'mu_v_{idx}.pt')

        if path.exists(tensor_path):
            return torch.load(tensor_path)
        else:
            return self._init_mu_v()

    def _get_log_var_v(self, idx):
        tensor_path = path.join(self.save_paths['tensors'], f'log_var_v_{idx}pt')

        if path.exists(tensor_path):
            return torch.load(tensor_path)
        else:
            return self._init_log_var_v()

    def _get_u_v(self, idx):
        tensor_path = path.join(self.save_paths['tensors'], f'u_v_{idx}.pt')

        if path.exists(tensor_path):
            return torch.load(tensor_path)
        else:
            return self._init_u_v()

    def write_idx_to_ID_json(self):
        txt_file_path = os.path.join(self.save_paths['dir'], 'idx_to_dataset_ID.json')

        with open(txt_file_path, 'w') as out:
            json.dump(dict(enumerate(self.im_mask_seg_triples)), out, indent=4, sort_keys=True)

