from os import listdir, path
from skimage import transform
from torch.utils.data import Dataset

from base import BaseDataLoader
from utils.util import init_identity_grid_2d, init_identity_grid_3d, rescale_im, standardise_im

import numpy as np
import os
import SimpleITK as sitk
import torch


class RGBDDataset(Dataset):
    def __init__(self, scene_paths, save_paths, no_consecutive_frames):
        self.scene_paths = scene_paths
        self.save_paths = save_paths
        self.no_consecutive_frames = no_consecutive_frames
        
        self.img_pairs = []
        self.identity_grid = None

        img_paths = {scene_paths: [path.join(scene_paths, f)
                                   for f in listdir(scene_paths) if path.isfile(path.join(scene_paths, f))]}
        temp = img_paths[scene_paths]

        # for img_path in temp:
        #     sublist = [other_img_path for other_img_path in temp if other_img_path != img_path]

        #     for other_img_path in sublist:
        #         self.img_pairs.append((img_path, other_img_path))

        img_path = temp[0]
        
        for other_img_path in temp:
            if other_img_path == img_path:
                continue

            self.img_pairs.append((img_path, other_img_path))

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pair = self.img_pairs[idx]

        im_fixed, im_moving = sitk.ReadImage(img_pair[0], sitk.sitkFloat32), \
                              sitk.ReadImage(img_pair[1], sitk.sitkFloat32)

        dim_x = 128
        dim_y = 128
        dim_z = 128

        im_fixed, im_moving = torch.from_numpy(transform.resize(
                                  np.transpose(sitk.GetArrayFromImage(im_fixed), (2, 1, 0)), (dim_x, dim_y, dim_z))), \
                              torch.from_numpy(transform.resize(
                                  np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0)), (dim_x, dim_y, dim_z)))
        
        # standardise images
        im_fixed, im_moving = standardise_im(im_fixed), standardise_im(im_moving)

        # rescale to range (-1, 1)
        im_fixed, im_moving = rescale_im(im_fixed), rescale_im(im_moving)

        assert im_fixed.shape == im_moving.shape, "images don't have the same dimensions"

        """
        if already exist, then load them from a file; otherwise create new ones
        """

        # velocity field
        if path.exists(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt')):  # if exists, then load
            mu_v = torch.load(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt'))
        else:  # otherwise initialise
            mu_v = torch.zeros_like(im_fixed)
            mu_v = torch.stack([mu_v, mu_v, mu_v], dim=0)

        # sigma_v
        if path.exists(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt')):
            log_var_v = torch.load(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt'))
        else:
            var_v = float(dim_x ** (-2)) * torch.ones_like(mu_v)
            log_var_v = torch.log(var_v)
        # u_v
        if path.exists(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt')):
            u_v = torch.load(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt'))
        else:
            u_v = torch.zeros_like(mu_v)

        im_fixed.unsqueeze_(0)
        im_moving.unsqueeze_(0)

        # sigma_f
        if path.exists(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt')):
            log_var_f = torch.load(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt'))
        else:
            var_f = (0.1 ** 2) * torch.ones_like(im_fixed)
            log_var_f = torch.log(var_f)
        # u_f
        if path.exists(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt')):
            u_f = torch.load(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt'))
        else:
            u_f = torch.zeros_like(im_fixed)

        # identity grid
        if self.identity_grid is None and len(im_fixed.shape) == 3:
            self.identity_grid = torch.squeeze(init_identity_grid_2d(dim_x, dim_y))
        elif self.identity_grid is None and len(im_fixed.shape) == 4:
            self.identity_grid = torch.squeeze(init_identity_grid_3d(dim_x, dim_y, dim_z))

        return idx, im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f, self.identity_grid


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, save_dirs=None, shuffle=True,
                 validation_split=0.0, num_workers=1, no_consecutive_frames=1):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        self.dataset = RGBDDataset(data_dir, save_dirs, no_consecutive_frames)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
