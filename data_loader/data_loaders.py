from os import listdir, path
from skimage.transform import resize
from torch.utils.data import Dataset

from base import BaseDataLoader
from utils.util import init_identity_grid_2d, init_identity_grid_3d

import nibabel as nib
import numpy as np
import os
import torch


class RGBDDataset(Dataset):
    def __init__(self, scene_paths, save_paths, no_consecutive_frames):
        self.scene_paths = scene_paths
        self.save_paths = save_paths

        img_paths = {scene_paths: [path.join(scene_paths, f)
                                   for f in listdir(scene_paths) if path.isfile(path.join(scene_paths, f))]}

        self.no_consecutive_frames = no_consecutive_frames
        self.img_pairs = []

        temp = img_paths[scene_paths]

        for idx, img_path in enumerate(temp):
            sublist = temp[idx+1:]

            for next_img_path in sublist[:self.no_consecutive_frames]:
                self.img_pairs.append((img_path, next_img_path))

        self.identity_grid = None

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_pair = self.img_pairs[idx]
        im1, im2 = np.array(nib.load(img_pair[0]).dataobj), np.array(nib.load(img_pair[1]).dataobj)

        dim_x = 128
        dim_y = 128
        dim_z = 128

        im1, im2 = torch.from_numpy(np.array(resize(im1, (dim_x, dim_y, dim_z)))),\
                   torch.from_numpy(np.array(resize(im2, (dim_x, dim_y, dim_z))))

        im1_min, im1_max, im2_min, im2_max = torch.min(im1), torch.max(im1), torch.min(im2), torch.max(im2)
        im1, im2 = 2.0 * (im1 - im1_min) / (im1_max - im1_min) - 1.0, 2.0 * (im2 - im2_min) / (im2_max - im2_min) - 1.0

        assert im1.shape == im2.shape, "images don't have the same dimensions"

        """
        if already exist, then load them from a file; otherwise create new ones
        """

        # velocity field
        if path.exists(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt')):  # if exists, then load
            mu_v = torch.load(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt'))
        else:  # otherwise initialise
            mu_v = torch.zeros_like(im1)
            mu_v = torch.stack([mu_v, mu_v, mu_v], dim=0)

        # sigma_v
        if path.exists(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt')):
            log_var_v = torch.load(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt'))
        else:
            var_v = ((1.0 / float(mu_v.shape[-1])) ** 2) * torch.ones_like(mu_v)
            log_var_v = torch.log(var_v)
        # u_v
        if path.exists(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt')):
            u_v = torch.load(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt'))
        else:
            u_v = torch.zeros_like(mu_v)

        im1.unsqueeze_(0)
        im2.unsqueeze_(0)

        # sigma_f
        if path.exists(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt')):
            log_var_f = torch.load(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt'))
        else:
            var_f = (0.1 ** 2) * torch.ones_like(im1)
            log_var_f = torch.log(var_f)
        # u_f
        if path.exists(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt')):
            u_f = torch.load(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt'))
        else:
            u_f = torch.zeros_like(im1)

        # identity grid
        if self.identity_grid is None and len(im1.shape) == 3:
            self.identity_grid = torch.squeeze(init_identity_grid_2d(dim_x, dim_y))
        elif self.identity_grid is None and len(im1.shape) == 4:
            self.identity_grid = torch.squeeze(init_identity_grid_3d(dim_x, dim_y, dim_z))

        return idx, im1, im2, mu_v, log_var_v, u_v, log_var_f, u_f, self.identity_grid


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, save_dirs=None, shuffle=True,
                 validation_split=0.0, num_workers=1, no_consecutive_frames=1):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        self.dataset = RGBDDataset(data_dir, save_dirs, no_consecutive_frames)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
