import itertools
from os import path

import pandas as pd
import torch

from base import BaseImageRegistrationDataset
from utils import rescale_im


class OasisDataset(BaseImageRegistrationDataset):
    def __init__(self, save_paths, im_pairs, dims, mu_v_init=0.0, sigma_v_init=1e-5, u_v_init=0.0, cps=None):
        data_path = '/vol/biodata/data/learn2reg/2021/task03'

        im_filename = 'aligned_norm.nii.gz'
        seg_filename = 'aligned_seg35.nii.gz'
        mask_filename = ''
        
        # segmentation IDs
        structures_dict_04 = {'cortex': 1, 'subcortical_grey_matter': 2, 'white_matter': 3, 'CSF': 4}

        structures_dict_24 = {'left_cerebral_white_matter': 1, 'left_cerebral_cortex': 2,
                              'left_lateral_ventricle': 3, 'left_inf_lateral_ventricle': 4,
                              'left_thalamus': 5, 'left_caudate': 6, 'left_putamen': 7,
                              'left_pallidum': 8,
                              '3rd_ventricle': 9, 'brain_stem': 10,
                              'left_hippocampus': 11, 'left_ventral_dorsal_cord': 12,
                              'left choroid_plexus': 13,
                              'right_cerebral_white_matter': 14, 'right_cerebral_cortex': 15,
                              'right_lateral_ventricle': 16, 'right_inf_lateral_ventricle': 17,
                              'right_thalamus': 18, 'right_caudate': 19, 'right_putamen': 20,
                              'right_pallidum': 21,
                              'right_hippocampus': 22, 'right_ventrcal_dorsal_cord': 23,
                              'right_choroid_plexus': 24}

        structures_dict_35 = {'left_cerebral_white_matter': 1, 'left_cerebral_cortex': 2,
                              'left_lateral_ventricle': 3, 'left_inf_lateral_ventricle': 4,
                              'left_cerebellum_white_matter': 5, 'left_cerebellum_cortex': 6,
                              'left_thalamus': 7, 'left_caudate': 8, 'left_putamen': 9,
                              'left_pallidum': 10, '3rd_ventricle': 11, '4th_ventricle': 12,
                              'brain_stem': 13, 'left_hippocampus': 14, 'left_amygdala': 15,
                              'left_accumbens': 16, 'left_ventral_dorsal_cord': 17, 'left_vessel': 18,
                              'left_choroid_plexus': 19,
                              'right_cerebral_white_matter': 20, 'right_cerebral_cortex': 21,
                              'right_lateral_ventricle': 22, 'right_inf_lateral_ventricle': 23,
                              'right_cerebellum_white_matter': 24, 'right_cerebellum_cortex': 25,
                              'right_thalamus': 26, 'right_caudate': 27, 'right_putamen': 28,
                              'right_pallidum': 29, 'right_hippocampus': 30, 'right_amygdala': 31,
                              'right_accumbens': 32, 'right_ventral_dorsal_cord': 33, 'right_vessel': 34,
                              'right_choroid_plexus': 35}

        if im_pairs == '':
            # NOTE (DG): these IDs are missing
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
        
        super().__init__(data_path, save_paths, im_pairs, im_filename, mask_filename, seg_filename, dims, cps=cps, structures_dict=structures_dict_35)

    @property
    def atlas_mode(self):
        return False
    
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

        return idx, fixed, moving, var_params_q_v

