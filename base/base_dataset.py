import json
import os
import random
from abc import abstractmethod, abstractproperty
from os import path

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import get_control_grid_size, rescale_im


class BaseImageRegistrationDataset(Dataset):
    def __init__(self, dims, im_paths, save_paths, im_mask_seg_triples, structures_dict, sigma_v_init, u_v_init, cps=None):
        self.dims, self.im_paths, self.save_paths = dims, im_paths, save_paths
        self.im_mask_seg_triples, self.structures_dict = im_mask_seg_triples, structures_dict
        self.cps, self.sigma_v_init, self.u_v_init = cps, sigma_v_init, u_v_init

        self.__set_im_spacing()

    def __len__(self):
        return len(self.im_mask_seg_triples)

    @property
    def dims_im(self):
        return 1, *dims

    @property
    def dims_v(self):
        parameter_dims = self.dims if self.cps is None else get_control_grid_size(self.dims, self.cps)
        return 3, *parameter_dims

    @abstractmethod
    def __preprocess(self, im_or_mask_or_seg):
        pass

    @abstractmethod
    def _preprocess_im(self, im):
        pass

    @abstractmethod
    def _preprocess_mask_or_seg(self, mask_or_seg):
        pass

    @staticmethod
    def _load_im_or_mask_or_seg_file(path):
        im_or_mask_or_seg = sitk.ReadImage(path)
        spacing = im_or_mask_or_seg.GetSpacing()
        im_or_mask_or_seg = sitk.GetArrayFromImage(im_or_mask_or_seg)
        im_or_mask_or_seg = torch.from_numpy(im_or_mask_or_seg)

        return im_or_mask_or_seg.unsqueeze(0).unsqueeze(0), spacing

    def _get_im(self, im_path):
        im, spacing = self._load_im_or_mask_or_seg_file(im_path)
        im = self._preprocess_im(im)

        return im, spacing

    def _get_mask_or_seg(self, mask_or_seg_path):
        mask_or_seg, spacing = self._load_im_or_mask_or_seg_file(mask_or_seg_path)
        mask_or_seg = self._preprocess_mask_or_seg(mask_or_seg)

        return mask_or_seg, spacing

    def _init_mu_v(self):
        return torch.zeros(self.dims_v)

    def _init_log_var_v(self):
        var_v = (self.sigma_v_init ** 2) + torch.zeros(self.dims_v)
        return var_v.log()

    def _init_u_v(self):
        return self.u_v_init + torch.zeros(self.dims_v)

    def _get_var_params(self, idx):
        state_dict_path = path.join(self.save_paths['var_params'], f'var_params_{idx}.pt')

        if path.exists(tensor_path):
            state_dict = torch.load(state_dict_path)
            mu_v, log_var_v, u_v = state_dict['mu'], state_dict['log_var'], state_dict['u']
            return mu_v, log_var_v, u_v

        return self._init(mu_v), self._init_log_var_v(), self._init_u_v()

    def __set_im_spacing(self):
        im_path = random.choice(self.im_mask_seg_triples)['im']
        im, im_spacing = self._load_im_or_mask_or_seg_file(im_path)
        self.im_spacing = torch.tensor(im_spacing).float()

    def write_idx_to_ID_json(self):
        txt_file_path = os.path.join(self.save_paths['dir'], 'idx_to_dataset_ID.json')

        with open(txt_file_path, 'w') as out:
            json.dump(dict(enumerate(self.im_mask_seg_triples)), out, indent=4, sort_keys=True)
