import itertools
from os import path

import pandas as pd
import torch

from base import BaseImageRegistrationDataset
from utils import rescale_im
from .oasis_structures_dicts import structures_dict_35


class OasisDataset(BaseImageRegistrationDataset):
    def __init__(self, save_paths, im_pairs, dims, mu_v_init=0.0, sigma_v_init=1e-5, u_v_init=0.0, cps=None):
        data_path = '/vol/biodata/data/learn2reg/2021/task03'

        im_filename = 'aligned_norm.nii.gz'
        seg_filename = 'aligned_seg35.nii.gz'
        mask_filename = ''

        structures_dict = structures_dict_35  # segmentation IDs
        im_pairs = self._get_im_pairs(im_pairs, save_paths)
        
        super().__init__(data_path, save_paths, im_pairs, im_filename, mask_filename, seg_filename,
                         dims, mu_v_init=mu_v_init, sigma_v_init=sigma_v_init, u_v_init=u_v_init,
                         cps=cps, structures_dict=structures_dict)

    @property
    def atlas_mode(self):
        return False

    @staticmethod
    def _get_im_pairs(im_pairs, save_paths):
        if im_pairs == '':
            missing_IDs = [8, 24, 36, 48, 89, 93,
                           100, 118, 128, 149, 154, 171, 172, 175, 187, 194, 196,
                           215, 219, 225, 242, 245, 248, 251, 252, 253, 257, 276, 297,
                           306, 320, 324, 334, 347, 360, 364, 369, 391, 393,
                           412, 414, 427, 436]
            existing_IDs = [idx for idx in range(1, 458) if idx not in missing_IDs]

            train_pairs = []
            val_pairs = [(idx, idx + 1) for idx in range(438, 457)]

            for IDs in itertools.combinations(existing_IDs, 2):
                if IDs not in val_pairs and reversed(IDs) not in val_pairs:
                    train_pairs.append({'fixed': IDs[0], 'moving': IDs[1]})

            train_pairs = pd.DataFrame(train_pairs)
            im_pairs = path.join(save_paths['dir'], 'train_pairs.csv')
            train_pairs.to_csv(im_pairs, header=False, index=False)

        return im_pairs

    def _get_im_path_from_ID(self, subject_idx):
        return path.join(path.join(self.data_path, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1'), self.im_filename)

    def _get_seg_path_from_ID(self, subject_idx):
        return path.join(path.join(self.data_path, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1'), self.seg_filename)

    def _preprocess(self, im_or_mask_or_seg):
        im_or_mask_or_seg = torch.rot90(im_or_mask_or_seg, dims=(3, 2))
        return im_or_mask_or_seg

    def _preprocess_im(self, im):
        im = self._preprocess(im)
        return rescale_im(im).squeeze(0)

    def _preprocess_mask_or_seg(self, mask_or_seg):
        mask_or_seg = self._preprocess(mask_or_seg)
        return mask_or_seg.squeeze(0)

    def __getitem__(self, idx):
        fixed = self._get_fixed(idx)
        moving = self._get_moving(idx)
        var_params_q_v = self._get_var_params(idx)

        fixed['mask'] = fixed['im'] > 1e-3

        return idx, fixed, moving, var_params_q_v

