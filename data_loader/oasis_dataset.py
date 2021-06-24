import itertools
import json
import os
from os import listdir, path

import SimpleITK as sitk
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
        im_filename = 'aligned_norm.nii.gz'
        seg_filename = 'seg35.nii.gz'

        subjects = listdir(self.im_paths)

        im_filenames = [path.join(path.join(self.im_paths, subject), im_filename) for subject in subjects if 'OASIS_' in subject]
        seg_filenames = [path.join(path.join(self.im_paths, subject), seg_filename) for subject in subjects if 'OASIS_' in subject]

        # tuples
        im_seg_tuples = zip(im_filenames, seg_filenames)
        self.im_seg_tuples_pairs = list(itertools.combinations(im_seg_tuples, 2))

        # validation pairs
        self.im_seg_tuples_pairs_val = list()

        val_pairs = [('438', '439'), ('439', '440'), ('440', '441'), ('441', '442'), ('442', '443'),
                     ('443', '444'), ('444', '445'), ('445', '446'), ('446', '447'), ('447', '448'),
                     ('448', '449'), ('449', '450'), ('450', '451'), ('451', '452'), ('452', '453'),
                     ('453', '454'), ('454', '455'), ('455', '456'), ('456', '457')]

        def is_val_pair(im_pair, val_pairs):
            for val_pair in val_pairs:
                idx0, idx1 = val_pair[0], val_pair[1]

                if (idx0 in im_pair[0] and idx1 in im_pair[1]) or (idx0 in im_pair[1] and idx1 in im_pair[0]):
                    return True

            return False

        for im_pair in self.im_seg_tuples_pairs:
            if is_val_pair(im_pair, val_pairs) and not test:
                self.im_seg_tuples_pairs.remove(im_pair)
            elif is_val_pair(im_pair, val_pairs) and test:
                self.im_seg_tuples_pairs_val.append(im_pair)

        if test:
            self.im_seg_tuples_pairs = self.im_seg_tuples_pairs_val.copy()

        # spacing
        im_path = self.im_seg_tuples_pairs[0][0][0]
        im = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im_spacing = im.GetSpacing()

        self.im_spacing = torch.tensor(im_spacing, dtype=torch.float32)

        # write to disk
        if rank == 0:
            txt_file_path = os.path.join(self.save_paths['dir'], 'idx_to_oasis_ID.json')

            with open(txt_file_path, 'w') as out:
                json.dump(dict(enumerate(self.im_seg_tuples_pairs)), out, indent=4, sort_keys=True)

        # segmentation labels
        self.structures_dict = {'left_cerebral_white_matter': 1, 'left_cerebral_cortex': 2,
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

    def __len__(self):
        return len(self.im_seg_tuples_pairs)

    def _get_mu_v(self):
        return torch.zeros(self.dims_v)

    def _get_log_var_v(self):
        sigma_v = self.sigma_v_init
        var_v = (sigma_v ** 2) * torch.ones(self.dims_v)

        return torch.log(var_v)

    def _get_u_v(self):
        return self.u_v_init * torch.ones(self.dims_v)

    def _get_image(self, im_path):
        im = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im_arr = sitk.GetArrayFromImage(im)
        im_spacing = im.GetSpacing()

        # pad
        im_tensor = torch.from_numpy(im_arr).unsqueeze(0)

        if self.rescale_im:
            im_tensor = rescale_im(im_tensor)

        return im_tensor, im_spacing

    def _get_seg(self, seg_path):
        seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        seg_arr = sitk.GetArrayFromImage(seg)

        # pad
        return torch.from_numpy(seg_arr).unsqueeze(0).short()

    def __getitem__(self, idx):
        # fixed image
        im_fixed_path = self.im_seg_tuples_pairs[idx][0][0]
        seg_fixed_path = self.im_seg_tuples_pairs[idx][0][1]

        im_fixed, im_fixed_spacing = self._get_image(im_fixed_path)
        mask_fixed = torch.ones_like(im_fixed).bool()
        seg_fixed = self._get_seg(seg_fixed_path)

        # moving image
        im_moving_path = self.im_seg_tuples_pairs[idx][1][0]
        seg_moving_path = self.im_seg_tuples_pairs[idx][1][1]

        im_moving, im_moving_spacing = self._get_image(im_moving_path)
        mask_moving = torch.ones_like(im_moving).bool()
        seg_moving = self._get_seg(seg_moving_path)

        if im_fixed_spacing != im_moving_spacing:
            raise ValueError(f'images have different spacings {im_fixed_spacing} and {im_moving_spacing}, exiting..')

        # transformation
        mu_v = self._get_mu_v()
        log_var_v = self._get_log_var_v()
        u_v = self._get_u_v()

        fixed = {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}
        moving = {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}
        var_params_q_v = {'mu': mu_v, 'log_var': log_var_v, 'u': u_v}

        return idx, fixed, moving, var_params_q_v
