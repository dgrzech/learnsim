from os import listdir, path
# from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset

from base import BaseDataLoader
from utils import init_identity_grid

import nibabel as nib
import numpy as np
import os
import torch


class RGBDDataset(Dataset):
    def __init__(self, scene_paths, save_paths, no_consecutive_frames):
        self.identity_grid = None

        self.scene_paths = scene_paths
        self.save_paths = save_paths

        img_paths = {scene_paths: [path.join(scene_paths, f) for f in listdir(scene_paths) if
                                   path.isfile(path.join(scene_paths, f))]}  # for scene_path in scene_paths}

        self.no_consecutive_frames = no_consecutive_frames
        self.img_pairs = []

        temp = img_paths[scene_paths]

        for idx, img_path in enumerate(temp):
            sublist = temp[idx+1:]
            for next_img_path in sublist[:self.no_consecutive_frames]:
                self.img_pairs.append((img_path, next_img_path))

            # sublist = temp[:idx]
            # for next_img_path in sublist[-self.no_consecutive_frames:]:
            #     self.img_pairs.append([next_img_path, img_path])

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_pair = self.img_pairs[idx]

        # im1 = torch.from_numpy(np.array(io.imread(img_pair[0])))
        # im2 = torch.from_numpy(np.array(io.imread(img_pair[1])))

        im1 = np.array(nib.load(img_pair[0]).dataobj)
        im1 = torch.from_numpy(np.array(resize(im1, (128, 128, 128))))

        im1_min = torch.min(im1)
        im1_max = torch.max(im1)
        im1 = 2.0 * (im1 - im1_min) / (im1_max - im1_min) - 1.0

        im2 = np.array(nib.load(img_pair[1]).dataobj)
        im2 = torch.from_numpy(np.array(resize(im2, (128, 128, 128))))

        im2_min = torch.min(im2)
        im2_max = torch.max(im2)
        im2 = 2.0 * (im2 - im2_min) / (im2_max - im2_min) - 1.0

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
            dim = 1.0 / (float(mu_v.shape[-1]) / 2.0)
            log_var_v = torch.log(dim * torch.ones_like(mu_v))
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
            log_var_f = torch.log(0.1 * torch.ones_like(im1))
        # u_f
        if path.exists(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt')):
            u_f = torch.load(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt'))
        else:
            u_f = torch.zeros_like(im1)

        # identity grid
        if self.identity_grid is None:
            self.identity_grid = torch.squeeze(init_identity_grid(im1.shape))

        return idx, im1, im2, mu_v, log_var_v, u_v, log_var_f, u_f, self.identity_grid


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, save_dirs=None, shuffle=True,
                 validation_split=0.0, num_workers=1, no_consecutive_frames=1):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        self.dataset = RGBDDataset(data_dir, save_dirs, no_consecutive_frames)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
