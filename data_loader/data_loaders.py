from os import listdir, path
from skimage import io
from torch.utils.data import Dataset

from base import BaseDataLoader
from utils import init_identity_grid

import numpy as np
import torch


class RGBDDataset(Dataset):
    def __init__(self, scene_paths, no_consecutive_frames):
        self.scene_paths = scene_paths
        img_paths = {scene_paths: [path.join(scene_paths, f) for f in listdir(scene_paths) if path.isfile(path.join(scene_paths, f))]}  # for scene_path in scene_paths}

        self.no_consecutive_frames = no_consecutive_frames
        self.img_pairs = []

        temp = img_paths[scene_paths]

        for idx, img_path in enumerate(temp):
            sublist = temp[idx+1:]
            for next_img_path in sublist[:self.no_consecutive_frames]:
                self.img_pairs.append((img_path, next_img_path))

            sublist = temp[:idx]
            for next_img_path in sublist[-self.no_consecutive_frames:]:
                self.img_pairs.append([next_img_path, img_path])

        self.identity_grid = None

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_pair = self.img_pairs[idx]

        im1 = torch.from_numpy(np.array(io.imread(img_pair[0])))
        im2 = torch.from_numpy(np.array(io.imread(img_pair[1])))

        assert im1.shape == im2.shape, "images don't have the same dimensions"

        """
        if already exist, then load them from a file; otherwise create new ones
        """

        # velocity field
        if path.exists('./temp/mu_v/mu_v_' + str(idx) + '.pt'):
            mu_v = torch.load('./temp/mu_v/mu_v_' + str(idx) + '.pt')
        else:
            mu_v = torch.zeros_like(im1)  # otherwise initialise
            mu_v = torch.stack([mu_v, mu_v, mu_v], dim=0)

        # sigma_v
        if path.exists('./temp/log_var_v/log_var_v_' + str(idx) + '.pt'):
            log_var_v = torch.load('./temp/log_var_v/log_var_v_' + str(idx) + '.pt')
        else:
            dim = 1.0 / (float(mu_v.shape[-1]) / 2.0)
            log_var_v = torch.log(dim * torch.ones_like(mu_v))
        # u_v
        if path.exists('./temp/modes_of_variation_v/u_' + str(idx) + '.pt'):
            u_v = torch.load('./temp/modes_of_variation_v/u_' + str(idx) + '.pt')
        else:
            u_v = torch.zeros_like(mu_v)

        im1.unsqueeze_(0)
        im2.unsqueeze_(0)

        # sigma_f
        if path.exists('./temp/log_var_f/log_var_f_' + str(idx) + '.pt'):
            log_var_f = torch.load('./temp/log_var_f/log_var_f_' + str(idx) + '.pt')
        else:
            log_var_f = torch.log(0.1 * torch.ones_like(im1))
        # u_f
        if path.exists('./temp/modes_of_variation_f/u_' + str(idx) + '.pt'):
            u_f = torch.load('./temp/modes_of_variation_f/u_' + str(idx) + '.pt')
        else:
            u_f = torch.zeros_like(im1)

        # identity grid
        if self.identity_grid is None:
            self.identity_grid = torch.squeeze(init_identity_grid(im1.shape))

        return idx, im1, im2, mu_v, log_var_v, u_v, log_var_f, u_f, self.identity_grid


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, no_consecutive_frames=1):
        self.data_dir = data_dir
        self.dataset = RGBDDataset(data_dir, no_consecutive_frames)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
