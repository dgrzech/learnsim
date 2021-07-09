import itertools
from os import path

import torch

from base import BaseImageRegistrationDataset
from utils import rescale_im


class OasisDataset(BaseImageRegistrationDataset):
    def __init__(self, dims, im_paths, save_paths, sigma_v_init, u_v_init, cps=None):
        # NOTE (DG): these IDs are missing
        missing_IDs = [8, 24, 36, 48, 89, 93,
                       100, 118, 128, 149, 154, 171, 172, 175, 187, 194, 196,
                       215, 219, 225, 242, 245, 248, 251, 252, 253, 257, 276, 297,
                       306, 320, 324, 334, 347, 360, 364, 369, 391, 393,
                       412, 414, 427, 436]
        existing_IDs = [str(idx) for idx in range(1, 458) if idx not in missing_IDs]

        im_mask_seg_triples = list()
        self.train_pairs_idxs, self.val_pairs_idxs = list(), list()

        for IDs_idx, IDs in enumerate(itertools.combinations(existing_IDs, 2)):
            if IDs in self.val_pairs or reversed(IDs) in self.val_pairs:
                self.val_pairs_idxs.append(IDs_idx)
            else:
                self.train_pairs_idxs.append(IDs_idx)

            fixed_idx, moving_idx = IDs[0], IDs[1]

            fixed_im_path, fixed_seg_path = self._get_im_filename_from_subject_ID(im_paths, fixed_idx), \
                                            self._get_seg_filename_from_subject_ID(im_paths, fixed_idx)
            moving_im_path, moving_seg_path = self._get_im_filename_from_subject_ID(im_paths, moving_idx), \
                                              self._get_seg_filename_from_subject_ID(im_paths, moving_idx)

            im_mask_seg_triples.append({'fixed': {'im': fixed_im_path, 'mask': '', 'seg': fixed_seg_path},
                                        'moving': {'im': moving_im_path, 'mask': '', 'seg': moving_seg_path}})

        # segmentation labels
        structures_4_dict = {'cortex': 1, 'subcortical_grey_matter': 2, 'white_matter': 3, 'CSF': 4}

        structures_24_dict = {'left_cerebral_white_matter': 1, 'left_cerebral_cortex': 2,
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

        structures_35_dict = {'left_cerebral_white_matter': 1, 'left_cerebral_cortex': 2,
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

        super().__init__(dims, im_paths, save_paths, im_mask_seg_triples, structures_35_dict, sigma_v_init, u_v_init, cps=cps)

    @property
    def atlas_mode(self):
        return False

    @staticmethod
    def _get_im_filename_from_subject_ID(im_paths, subject_idx):
        return path.join(path.join(im_paths, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1'), 'aligned_norm.nii.gz')

    @staticmethod
    def _get_seg_filename_from_subject_ID(im_paths, subject_idx):
        return path.join(path.join(im_paths, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1'), 'aligned_seg35.nii.gz')

    @property
    def val_pairs(self):
        return [(str(idx), str(idx + 1)) for idx in range(438, 457)]

    def __preprocess(self, im_or_mask_or_seg):
        im_or_mask_or_seg = torch.rot90(im_or_mask_or_seg, dims=(3, 2))
        return im_or_mask_or_seg

    def _preprocess_im(self, im):
        im = self.__preprocess(im)
        return rescale_im(im).squeeze(0)

    def _preprocess_mask_or_seg(self, mask_or_seg):
        mask_or_seg = self.__preprocess(mask_or_seg)
        return mask_or_seg.squeeze(0)

    def __getitem__(self, idx):
        # fixed image
        im_fixed_path = self.im_mask_seg_triples[idx]['fixed']['im']
        seg_fixed_path = self.im_mask_seg_triples[idx]['fixed']['seg']

        im_fixed, im_fixed_spacing = self._get_im(im_fixed_path)
        mask_fixed = torch.ones_like(im_fixed).bool()
        seg_fixed, _ = self._get_mask_or_seg(seg_fixed_path)

        # moving image
        im_moving_path = self.im_mask_seg_triples[idx]['moving']['im']
        seg_moving_path = self.im_mask_seg_triples[idx]['moving']['seg']

        im_moving, im_moving_spacing = self._get_im(im_moving_path)
        mask_moving = torch.ones_like(im_moving).bool()
        seg_moving, _ = self._get_mask_or_seg(seg_moving_path)

        if im_fixed_spacing != im_moving_spacing:
            raise ValueError(f'images have different spacings {im_fixed_spacing} and {im_moving_spacing}, exiting..')

        # transformation
        mu_v = self._get_mu_v(idx)
        log_var_v = self._get_log_var_v(idx)
        u_v = self._get_u_v(idx)

        fixed = {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed.short()}
        moving = {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving.short()}
        var_params_q_v = {'mu': mu_v, 'log_var': log_var_v, 'u': u_v}

        return idx, fixed, moving, var_params_q_v
