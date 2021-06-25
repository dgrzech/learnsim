import itertools
from os import path

import torch

from base import BaseImageRegistrationDataset
from utils import get_control_grid_size, rescale_im


class OasisDataset(BaseImageRegistrationDataset):
    def __init__(self, dims, im_paths, save_paths, sigma_v_init, u_v_init, cps=None, rescale_im=False, rank=0, test=False):
        self.im_paths, self.save_paths = im_paths, save_paths
        self.im_spacing = None
        self.rescale_im = rescale_im

        self.dims, self.dims_im, = dims, (1, *dims)

        if cps is None:
            self.dims_v = (3, *dims)
        else:
            control_grid_sz = get_control_grid_size(dims, cps)
            self.dims_v = (3, *control_grid_sz)

        self.sigma_v_init, self.u_v_init = sigma_v_init, u_v_init

        # image filenames
        im_mask_seg_triples = list()
        missing_IDs = [118, 154, 369, 414]  # NOTE (DG): these IDs are missing
        self.train_pairs_idxs, self.val_pairs_idxs = list(), list()

        for IDs_idx, IDs in enumerate(itertools.combinations(list(range(1, 458)), 2)):
            if IDs[0] in missing_IDs or IDs[1] in missing_IDs:
                continue

            fixed_idx, moving_idx = IDs[0], IDs[1]

            if IDs in self.val_pairs or reversed(IDs) in self.val_pairs:
                self.val_pairs_idxs.append(IDs_idx)  # FIXME
            else:
                self.train_pairs_idxs.append(IDs_idx)

            fixed_im_path, fixed_seg_path = self._get_im_filename_from_subject_ID(im_paths, fixed_idx), \
                                            self._get_seg_filename_from_subject_ID(im_paths, fixed_idx)
            moving_im_path, moving_seg_path = self._get_im_filename_from_subject_ID(im_paths, moving_idx), \
                                              self._get_seg_filename_from_subject_ID(im_paths, moving_idx)

            im_mask_seg_triples.append({'fixed': {'im': fixed_im_path, 'mask': '', 'seg': fixed_seg_path},
                                        'moving': {'im': moving_im_path, 'mask': '', 'seg': moving_seg_path}})

        # segmentation labels
        structures_dict = {'left_cerebral_white_matter': 1, 'left_cerebral_cortex': 2,
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

        super().__init__(dims, im_paths, save_paths, im_mask_seg_triples, structures_dict, sigma_v_init, u_v_init,
                         pad=False, rescale=True, resize=False, cps=cps)

    @staticmethod
    def _get_im_filename_from_subject_ID(im_paths, subject_idx):
        return path.join(path.join(im_paths, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1'), 'aligned_norm.nii.gz')

    @staticmethod
    def _get_seg_filename_from_subject_ID(im_paths, subject_idx):
        return path.join(path.join(im_paths, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1'), 'seg35.nii.gz')

    @property
    def val_pairs(self):
        return [(str(idx), str(idx + 1)) for idx in range(438, 457)]

    def __getitem__(self, idx):
        # fixed image
        im_fixed_path = self.im_mask_seg_triples[idx]['fixed']['im']
        seg_fixed_path = self.im_mask_seg_triples[idx]['fixed']['seg']

        im_fixed, im_fixed_spacing = self._get_image(im_fixed_path)
        mask_fixed = torch.ones_like(im_fixed).bool()
        seg_fixed = self._get_seg(seg_fixed_path)

        # moving image
        im_moving_path = self.im_mask_seg_triples[idx]['moving']['im']
        seg_moving_path = self.im_mask_seg_triples[idx]['moving']['seg']

        im_moving, im_moving_spacing = self._get_image(im_moving_path)
        mask_moving = torch.ones_like(im_moving).bool()
        seg_moving = self._get_seg(seg_moving_path)

        if im_fixed_spacing != im_moving_spacing:
            raise ValueError(f'images have different spacings {im_fixed_spacing} and {im_moving_spacing}, exiting..')

        # transformation
        mu_v = self._get_mu_v(idx)
        log_var_v = self._get_log_var_v(idx)
        u_v = self._get_u_v(idx)

        fixed = {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}
        moving = {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}
        var_params_q_v = {'mu': mu_v, 'log_var': log_var_v, 'u': u_v}

        return idx, fixed, moving, var_params_q_v
