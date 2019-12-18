from os import listdir, path
from skimage import io, transform
from torch.utils.data import Dataset

from utils.util import init_identity_grid_2d, init_identity_grid_3d, rescale_im, standardise_im

import numpy as np
import os
import SimpleITK as sitk
import torch


def init_mu_v(dims):
    return torch.zeros(dims)


def init_log_var_v(dims):
    dim_x = dims[1]
    var_v = float(dim_x ** (-2)) * torch.ones(dims)

    return torch.log(var_v)


def init_u_v(dims):
    return torch.zeros(dims)


def init_log_var_f(dims):
    var_f = (0.1 ** 2) * torch.ones(dims)
    return torch.log(var_f)


def init_u_f(dims):
    return torch.zeros(dims)


class BiobankDataset(Dataset):
    def __init__(self, scene_paths, save_paths):
        self.scene_paths = scene_paths
        self.save_paths = save_paths
        self.img_pairs = []

        self.dim_x = 128
        self.dim_y = 128
        self.dim_z = 128

        self.identity_grid = None

        img_paths = {scene_paths: [path.join(scene_paths, f)
                                   for f in listdir(scene_paths) if path.isfile(path.join(scene_paths, f))]}

        # all-to-one
        temp = img_paths[scene_paths]
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

        im_fixed, im_moving = torch.from_numpy(
            transform.resize(
                np.transpose(sitk.GetArrayFromImage(im_fixed), (2, 1, 0)), (self.dim_x, self.dim_y, self.dim_z))), \
                              torch.from_numpy(
            transform.resize(
                np.transpose(sitk.GetArrayFromImage(im_moving), (2, 1, 0)), (self.dim_x, self.dim_y, self.dim_z)))

        # standardise images
        im_fixed, im_moving = standardise_im(im_fixed), standardise_im(im_moving)
        # rescale to range (-1, 1)
        im_fixed, im_moving = rescale_im(im_fixed), rescale_im(im_moving)

        assert im_fixed.shape == im_moving.shape, "images don't have the same dimensions"

        im_fixed.unsqueeze_(0)
        im_moving.unsqueeze_(0)

        dims_im = (1, self.dim_x, self.dim_y, self.dim_z)
        dims_v = (3, self.dim_x, self.dim_y, self.dim_z)

        """
        if already exist then load them from a file, otherwise initialise new ones
        """

        # velocity field
        if path.exists(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt')):
            mu_v = torch.load(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt'))
        else:
            mu_v = init_mu_v(dims_v)

        # sigma_v
        if path.exists(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt')):
            log_var_v = torch.load(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt'))
        else:
            log_var_v = init_log_var_v(dims_v)
        # u_v
        if path.exists(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt')):
            u_v = torch.load(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt'))
        else:
            u_v = init_u_v(dims_v)

        # sigma_f
        if path.exists(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt')):
            log_var_f = torch.load(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt'))
        else:
            log_var_f = init_log_var_f(dims_im)
        # u_f
        if path.exists(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt')):
            u_f = torch.load(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt'))
        else:
            u_f = init_u_f(dims_im)

        # identity grid
        if self.identity_grid is None and len(im_fixed.shape) == 3:
            self.identity_grid = torch.squeeze(init_identity_grid_2d(self.dim_x, self.dim_y))
        elif self.identity_grid is None and len(im_fixed.shape) == 4:
            self.identity_grid = torch.squeeze(init_identity_grid_3d(self.dim_x, self.dim_y, self.dim_z))

        return idx, im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f, self.identity_grid


class RGBDDataset(Dataset):
    def __init__(self, scene_paths, save_paths, no_consecutive_frames):
        self.scene_paths = scene_paths
        self.save_paths = save_paths
        self.no_consecutive_frames = no_consecutive_frames

        self.img_pairs = []
        self.identity_grid = None

        img_paths = {scene_paths: [path.join(scene_paths, f)
                                   for f in listdir(scene_paths) if path.isfile(path.join(scene_paths, f))]}

        # all-to-all
        temp = img_paths[scene_paths]

        for idx, img_path in enumerate(temp):
            sublist = temp[idx+1:]
            for next_img_path in sublist[:self.no_consecutive_frames]:
                self.img_pairs.append((img_path, next_img_path))

            sublist = temp[:idx]
            for next_img_path in sublist[-self.no_consecutive_frames:]:
                self.img_pairs.append([next_img_path, img_path])

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pair = self.img_pairs[idx]

        im_fixed, im_moving = torch.from_numpy(np.array(io.imread(img_pair[0]))), \
                              torch.from_numpy(np.array(io.imread(img_pair[1])))

        assert im_fixed.shape == im_moving.shape, "images don't have the same dimensions"

        im_fixed.unsqueeze_(0)
        im_moving.unsqueeze_(0)

        dim_x = im_fixed.shape[3]
        dim_y = im_fixed.shape[2]
        dim_z = im_fixed.shape[1]

        dims_im = (1, dim_x, dim_y, dim_z)
        dims_v = (3, dim_x, dim_y, dim_z)

        """
        if already exist then load them from a file, otherwise initialise new ones
        """

        # velocity field
        if path.exists(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt')):
            mu_v = torch.load(os.path.join(self.save_paths['mu_v'], 'mu_v_' + str(idx) + '.pt'))
        else:
            mu_v = init_mu_v(dims_v)

        # sigma_v
        if path.exists(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt')):
            log_var_v = torch.load(os.path.join(self.save_paths['log_var_v'], 'log_var_v_' + str(idx) + '.pt'))
        else:
            log_var_v = init_log_var_v(dims_v)

        # u_v
        if path.exists(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt')):
            u_v = torch.load(os.path.join(self.save_paths['u_v'], 'u_v_' + str(idx) + '.pt'))
        else:
            u_v = init_u_v(dims_v)

        # sigma_f
        if path.exists(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt')):
            log_var_f = torch.load(os.path.join(self.save_paths['log_var_f'], 'log_var_f_' + str(idx) + '.pt'))
        else:
            log_var_f = init_log_var_f(dims_im)
        # u_f
        if path.exists(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt')):
            u_f = torch.load(os.path.join(self.save_paths['u_f'], 'u_f_' + str(idx) + '.pt'))
        else:
            u_f = init_u_f(dims_im)

        # identity grid
        if self.identity_grid is None:
            self.identity_grid = torch.squeeze(init_identity_grid_3d(dim_x, dim_y, dim_z))

        return idx, im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f, self.identity_grid
