import torch
import torch.nn.functional as F

from base import BaseImageRegistrationDataset
from utils import rescale_im
from .biobank_structures_dict import biobank_structures_dict


class BiobankDataset(BaseImageRegistrationDataset):
    def __init__(self, save_paths, im_pairs, dims, mu_v_init=0.0, sigma_v_init=1e-5, u_v_init=0.0, cps=None):
        data_path = '/vol/biodata/data/biobank/12579/brain/t0/affine_to_mni/images/'

        im_filename = 'T2_FLAIR_unbiased_brain_affine_to_mni.nii.gz'
        seg_filename = 'T1_first_all_fast_firstseg_affine_to_mni.nii.gz'
        mask_filename = 'T1_brain_mask_affine_to_mni.nii.gz'

        structures_dict = biobank_structures_dict  # segmentation IDs

        super().__init__(data_path, save_paths, im_pairs, im_filename, mask_filename, seg_filename, dims,
                         mu_v_init=mu_v_init, sigma_v_init=sigma_v_init, u_v_init=u_v_init,
                         cps=cps, structures_dict=structures_dict)

        # pre-load the fixed image, the segmentation, and the mask
        self.fixed = self._get_fixed(0)

        for k, v in self.fixed.items():
            v.unsqueeze_(0)

    def __preprocess(self, im_or_mask_or_seg):
        im_or_mask_or_seg = torch.flipud(im_or_mask_or_seg.reshape(-1)).reshape(im_or_mask_or_seg.shape)
        im_or_mask_or_seg = F.pad(im_or_mask_or_seg, self.padding, mode='constant')

        return im_or_mask_or_seg

    def _preprocess_im(self, im):
        im = self.__preprocess(im)
        im = F.interpolate(im, size=self.dims, mode='trilinear', align_corners=True)

        return rescale_im(im).squeeze(0)

    def _preprocess_mask_or_seg(self, mask_or_seg):
        mask_or_seg = self.__preprocess(mask_or_seg)
        mask_or_seg = F.interpolate(mask_or_seg, size=self.dims, mode='nearest')

        return mask_or_seg.squeeze(0)

    def __getitem__(self, idx):
        moving = self._get_moving(idx)
        var_params_q_v = self._get_var_params(idx)

        return idx, None, moving, var_params_q_v
