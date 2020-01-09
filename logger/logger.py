from os import path

from pathlib import Path
from utils import compute_norm, read_json, save_field_to_disk, save_im_to_disk

import logging
import logging.config


def save_images(im_pair_idxs, save_dirs_dict, im_fixed, im_moving, im_moving_warped, 
                mu_v, log_var_v, u_v, log_var_f, u_f, deformation_field,
                seg_fixed=None, seg_moving=None, seg_moving_warped=None):
    """
    save the input and output images as well as norms of vectors in the vector fields to disk
    """

    im_pair_idxs = im_pair_idxs.tolist()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        save_im_to_disk(im_moving_warped[loop_idx, :, :, :, :], 
                        path.join(save_dirs_dict['images'], 'im_moving_warped_' + str(im_pair_idx) + '.nii.gz'))

        save_im_to_disk(log_var_f[loop_idx, :, :, :, :], 
                        path.join(save_dirs_dict['images'], 'log_var_f_' + str(im_pair_idx) + '.nii.gz'))
        save_im_to_disk(u_f[loop_idx, :, :, :, :], 
                        path.join(save_dirs_dict['images'], 'u_f_' + str(im_pair_idx) + '.nii.gz'))
        
        mu_v_norm = compute_norm(mu_v[loop_idx, :, :, :, :])
        log_var_v_norm = compute_norm(log_var_v[loop_idx, :, :, :, :])
        u_v_norm = compute_norm(u_v[loop_idx, :, :, :, :])

        deformation_field_norm = compute_norm(deformation_field[loop_idx, :, :, :, :])

        save_im_to_disk(mu_v_norm, 
                        path.join(save_dirs_dict['norms'], 'mu_v_norm_' + str(im_pair_idx) + '.nii.gz'))
        save_im_to_disk(log_var_v_norm, 
                        path.join(save_dirs_dict['norms'], 'log_var_v_norm_' + str(im_pair_idx) + '.nii.gz'))
        save_im_to_disk(u_v_norm, 
                        path.join(save_dirs_dict['norms'], 'u_v_norm_' + str(im_pair_idx) + '.nii.gz'))
        save_im_to_disk(deformation_field_norm, 
                        path.join(save_dirs_dict['norms'], 'deformation_field_norm_' + str(im_pair_idx) + '.nii.gz'))

        save_field_to_disk(mu_v[loop_idx, :, :, :, :], 
                           path.join(save_dirs_dict['mu_v_field'], 'mu_v_' + str(im_pair_idx) + '.nii.gz'))

        im_fixed_path = path.join(save_dirs_dict['images'], 'im_fixed_' + str(im_pair_idx) + '.nii.gz')
        if not path.exists(im_fixed_path):
            save_im_to_disk(im_fixed[loop_idx, :, :, :, :], im_fixed_path)
        
        im_moving_path = path.join(save_dirs_dict['images'], 'im_moving_' + str(im_pair_idx) + '.nii.gz')
        if not path.exists(im_moving_path):
            save_im_to_disk(im_moving[loop_idx, :, :, :, :], im_moving_path)
        
        if seg_fixed is not None:
            seg_fixed_path = path.join(save_dirs_dict['segs'], 'seg_fixed_' + str(im_pair_idx) + '.nii.gz')

            if not path.exists(seg_fixed_path):
                save_im_to_disk(seg_fixed[loop_idx, :, :, :, :], seg_fixed_path)

        if seg_moving is not None:
            seg_moving_path = path.join(save_dirs_dict['segs'], 'seg_moving_' + str(im_pair_idx) + '.nii.gz')

            if not path.exists(seg_moving_path):
                save_im_to_disk(seg_moving[loop_idx, :, :, :, :], seg_moving_path)
                                
        if seg_moving_warped is not None:
            save_im_to_disk(seg_moving_warped[loop_idx, :, :, :, :],
                            path.join(save_dirs_dict['segs'], 'seg_moving_warped_' + str(im_pair_idx) + '.nii.gz'))


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    setup logging configuration
    """

    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
