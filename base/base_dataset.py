from abc import abstractmethod
from os import path

import SimpleITK as sitk
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import get_control_grid_size


class BaseImageRegistrationDataset(Dataset):
    def __init__(self, data_path, save_paths, im_pairs, im_filename, mask_filename, seg_filename,
                 dims, mu_v_init=0.0, sigma_v_init=1e-5, u_v_init=0.0, cps=None, structures_dict=None):
        self.data_path, self.save_paths = data_path, save_paths
        self.dims = dims
        self.mu_v_init, self.sigma_v_init, self.u_v_init, self.cps = mu_v_init, sigma_v_init, u_v_init, cps
        self.structures_dict = structures_dict

        self.im_filename, self.mask_filename, self.seg_filename = im_filename, mask_filename, seg_filename
        self.im_pairs = pd.read_csv(im_pairs, names=['fixed', 'moving']).applymap(str)

        self.__set_im_spacing()

    def __len__(self):
        return len(self.im_pairs.index)

    @property
    def dims_im(self):
        return 1, *self.dims

    @property
    def dims_v(self):
        parameter_dims = self.dims if self.cps is None else get_control_grid_size(self.dims, self.cps)
        return 3, *parameter_dims

    """
    images, masks, and segmentations
    """

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
    def _load_im_or_mask_or_seg_file(im_or_mask_or_seg_path):
        im_or_mask_or_seg = sitk.ReadImage(im_or_mask_or_seg_path)
        spacing = im_or_mask_or_seg.GetSpacing()
        im_or_mask_or_seg = sitk.GetArrayFromImage(im_or_mask_or_seg)
        im_or_mask_or_seg = torch.from_numpy(im_or_mask_or_seg)

        return im_or_mask_or_seg.unsqueeze(0).unsqueeze(0), spacing

    def _get_im_path_from_ID(self, ID):
        return path.join(path.join(self.data_path, ID), self.im_filename)

    def _get_mask_path_from_ID(self, ID):
        return path.join(path.join(self.data_path, ID), self.mask_filename)

    def _get_seg_path_from_ID(self, ID):
        return path.join(path.join(self.data_path, ID), self.seg_filename)

    def _get_im(self, ID):
        im_path = self._get_im_path_from_ID(ID)
        im, spacing = self._load_im_or_mask_or_seg_file(im_path)
        im = self._preprocess_im(im)

        return im, spacing

    def _get_mask(self, ID):
        if self.mask_filename == '':
            return torch.ones(self.dims_im).bool(), None

        mask_path = self._get_mask_path_from_ID(ID)
        mask, spacing = self._load_im_or_mask_or_seg_file(mask_path)
        mask = self._preprocess_mask_or_seg(mask)

        return mask.bool(), spacing

    def _get_seg(self, ID):
        seg_path = self._get_seg_path_from_ID(ID)
        seg, spacing = self._load_im_or_mask_or_seg_file(seg_path)
        seg = self._preprocess_mask_or_seg(seg)

        return seg.long(), spacing

    def _get_fixed(self, idx):
        ID_fixed = self.im_pairs['fixed'].iloc[idx]

        # moving image
        im_fixed, _ = self._get_im(ID_fixed)
        mask_fixed, _ = self._get_mask(ID_fixed)
        seg_fixed, _ = self._get_seg(ID_fixed)

        return {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}

    def _get_moving(self, idx):
        ID_moving = self.im_pairs['moving'].iloc[idx]

        # moving image
        im_moving, _ = self._get_im(ID_moving)
        mask_moving, _ = self._get_mask(ID_moving)
        seg_moving, _ = self._get_seg(ID_moving)
        
        return {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}

    """
    variational parameters
    """

    def _init_mu_v(self):
        return self.mu_v_init + torch.zeros(self.dims_v)

    def _init_log_var_v(self):
        var_v = (self.sigma_v_init ** 2) + torch.zeros(self.dims_v)
        return var_v.log()

    def _init_u_v(self):
        return self.u_v_init + torch.zeros(self.dims_v)

    def _get_var_params(self, idx):
        state_dict_path = path.join(self.save_paths['var_params'], f'var_params_{idx}.pt')

        if path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path)
            return state_dict

        return {'mu': self._init_mu_v(), 'log_var': self._init_log_var_v(), 'u': self._init_u_v()}

    def __set_im_spacing(self):
        idx = self.im_pairs['fixed'].sample().iloc[0]
        im_path = self._get_im_path_from_ID(idx)
        im, im_spacing = self._load_im_or_mask_or_seg_file(im_path)
        self.im_spacing = torch.tensor(im_spacing).float()
