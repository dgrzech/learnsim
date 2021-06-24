from os import listdir, path

from .dataset import ImageRegistrationDataset


class BiobankDataset(ImageRegistrationDataset):
    def __init__(self, dims, im_paths, save_paths, sigma_v_init, u_v_init, cps=None):
        atlas_mode = True

        # image filenames
        im_filenames = self._get_filenames(im_paths)
        mask_filenames = self._get_filenames(path.join(im_paths, 'masks'))
        seg_filenames = self._get_filenames(path.join(im_paths, 'segs'))

        im_mask_seg_triples = list()  # all-to-one

        for triple in list(zip(im_filenames, mask_filenames, seg_filenames)):
            im_mask_seg_triples.append({'im': triple[0], 'mask': triple[1], 'seg': triple[2]})

        # segmentation IDs
        structures_dict = {'left_thalamus': 10, 'left_caudate': 11, 'left_putamen': 12,
                           'left_pallidum': 13, 'brain_stem': 16, 'left_hippocampus': 17,
                           'left_amygdala': 18, 'left_accumbens': 26, 'right_thalamus': 49,
                           'right_caudate': 50, 'right_putamen': 51, 'right_pallidum': 52,
                           'right_hippocampus': 53, 'right_amygdala': 54, 'right_accumbens': 58}

        super().__init__(dims, im_paths, save_paths, im_mask_seg_triples, structures_dict, sigma_v_init, u_v_init,
                         pad=True, rescale=True, resize=True, cps=cps)

        # pre-load the fixed image, the segmentation, and the mask
        fixed_triple = im_mask_seg_triples.pop(0)

        im_fixed_path = fixed_triple['im']
        mask_fixed_path = fixed_triple['mask']
        seg_fixed_path = fixed_triple['seg']

        im_fixed, _ = self._get_image(im_fixed_path)
        mask_fixed = self._get_mask(mask_fixed_path)
        seg_fixed = self._get_seg(seg_fixed_path)

        im_fixed.unsqueeze_(0), mask_fixed.unsqueeze_(0), seg_fixed.unsqueeze_(0)

        self.fixed = {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}

    @staticmethod
    def _get_filenames(p):
        if listdir(p):
            return sorted([path.join(p, f) for f in listdir(p) if path.isfile(path.join(p, f))])

        return ['' for _ in range(2)]

    def __getitem__(self, idx):
        # moving image
        im_moving_path = self.im_mask_seg_triples[idx]['im']
        mask_moving_path = self.im_mask_seg_triples[idx]['mask']
        seg_moving_path = self.im_mask_seg_triples[idx]['seg']

        im_moving, _ = self._get_image(im_moving_path)
        mask_moving = self._get_mask(mask_moving_path)
        seg_moving = self._get_seg(seg_moving_path)

        # transformation
        mu_v = self._get_mu_v(idx)
        log_var_v = self._get_log_var_v(idx)
        u_v = self._get_u_v(idx)

        moving = {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}
        var_params_q_v = {'mu': mu_v, 'log_var': log_var_v, 'u': u_v}

        return idx, moving, var_params_q_v
