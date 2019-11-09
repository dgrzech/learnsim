import numpy as np
import os
import torch

from base import BaseDataLoader

from itertools import combinations
from skimage import io
from torch.utils.data import Dataset


class RGBDDataset(Dataset):
    def __init__(self, scene_paths):
        self.scene_paths = scene_paths
        img_paths = {scene_paths: [os.path.join(scene_paths, f) for f in os.listdir(scene_paths) if os.path.isfile(os.path.join(scene_paths, f))]}  # for scene_path in scene_paths}

        temp = img_paths[scene_paths]
        self.img_pairs = [comb for comb in combinations(temp, 2)]

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pair = self.img_pairs[idx]

        im1 = np.array(io.imread(img_pair[0]))
        im2 = np.array(io.imread(img_pair[1]))

        im1 = np.expand_dims(im1, axis=0)
        im2 = np.expand_dims(im2, axis=0)

        """
        if already exist, then load them from a file; otherwise create new ones
        """

        # velocity field
        if os.path.exists('./temp/vels/v_' + str(idx) + '.pt'):
            v = torch.load('./temp/vels/v_' + str(idx) + '.pt')
        else:
            v = torch.zeros([128, 128, 128, 3], dtype=torch.float32)  # otherwise initialise
            torch.save(v, './temp/vels/v_' + str(idx) + '.pt')

        # voxel uncertainty of v
        if os.path.exists('./temp/sigmas_v/sigma_' + str(idx) + '.pt'):
            sigma_voxel_v = torch.load('./temp/sigmas_v/sigma_' + str(idx) + '.pt')
        else:
            sigma_voxel_v = torch.ones([128, 128, 128], dtype=torch.float32)
            torch.save(sigma_voxel_v, './temp/sigmas_v/sigma_' + str(idx) + '.pt')

        # voxel uncertainty of f
        if os.path.exists('./temp/sigmas_f/sigma_' + str(idx) + '.pt'):
            sigma_voxel_f = torch.load('./temp/sigmas_f/sigma_' + str(idx) + '.pt')
        else:
            sigma_voxel_f = torch.ones([128, 128, 128], dtype=torch.float32)
            torch.save(sigma_voxel_f, './temp/sigmas_f/sigma_' + str(idx) + '.pt')

        # mode of variation of v
        if os.path.exists('./temp/modes_of_variation_v/u_' + str(idx) + '.pt'):
            u_v = torch.load('./temp/modes_of_variation_v/u_' + str(idx) + '.pt')
        else:
            u_v = torch.zeros([128, 128, 128, 3], dtype=torch.float32)
            torch.save(u_v, './temp/modes_of_variation_v/u_' + str(idx) + '.pt')

        # mode of variation of f
        if os.path.exists('./temp/modes_of_variation_f/u_' + str(idx) + '.pt'):
            u_f = torch.load('./temp/modes_of_variation_f/u_' + str(idx) + '.pt')
        else:
            u_f = torch.zeros([1, 128, 128, 128], dtype=torch.float32)
            torch.save(u_f, './temp/modes_of_variation_f/u_' + str(idx) + '.pt')

        return torch.from_numpy(im1), torch.from_numpy(im2), v, sigma_voxel_v, u_v, sigma_voxel_f, u_f


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = RGBDDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

