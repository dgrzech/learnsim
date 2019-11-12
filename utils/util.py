from abc import ABC, abstractmethod

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from scipy.linalg import toeplitz
from torch import nn
from torch.distributions import Normal


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """
    wrapper function for endless data loader.
    """
    for loader in repeat(data_loader):
        yield from loader


def generate_id_grid(sz):
    nz = sz[0]
    ny = sz[1]
    nx = sz[2]

    x = torch.linspace(-1, 1, steps=nx).to(dtype=torch.float32)
    y = torch.linspace(-1, 1, steps=ny).to(dtype=torch.float32)
    z = torch.linspace(-1, 1, steps=nz).to(dtype=torch.float32)

    x = x.expand(ny, -1).expand(nz, -1, -1)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(4)
    y.unsqueeze_(0).unsqueeze_(4)
    z.unsqueeze_(0).unsqueeze_(4)

    grid = torch.cat((x, y, z), 4).to(dtype=torch.float32, device='cuda:0')
    return grid


def forward(n):
    r = np.zeros(n)
    c = np.zeros(n)

    r[0] = -6.0
    r[1] = 1.0
    r[2] = 1.0
    r[4] = 1.0

    c[0] = -6.0
    c[1] = 1.0
    c[2] = 1.0
    c[4] = 1.0

    return toeplitz(r, c)


def integrate_v(v, identity_grid, no_steps=12):
    assert no_steps >= 0, 'nb_steps should be >=0, found: %d' % no_steps
    out = v / (2 ** no_steps)

    for _ in range(no_steps):
        w = identity_grid + out.permute([0, 2, 3, 4, 1])
        out = out + F.grid_sample(out, w, padding_mode='zeros')

    return out


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


class Sampler:
    def __init__(self, device):
        self.device = device
        self.normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def sample_qv(self, v, sigma_voxel_v, u_v):
        epsilon = sigma_voxel_v * self.normal_dist.sample(sample_shape=sigma_voxel_v.shape).squeeze(5).to(self.device)
        x = self.normal_dist.sample(sample_shape=u_v.shape).squeeze(5).to(self.device)
        return v + epsilon + x * u_v

    def sample_qf(self, f, sigma_voxel_f, u_f):
        epsilon = sigma_voxel_f * self.normal_dist.sample(sample_shape=f.shape).squeeze(5).to(self.device)
        x = self.normal_dist.sample(sample_shape=u_f.shape).squeeze(5).to(self.device)
        return f + epsilon + x * u_f


class DifferentialOperator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, v):
        return


class GradientOperator(DifferentialOperator):
    def __init__(self):
        self.p1d1 = (0, 0, 0, 0, 0, 0, 1, 0)
        self.p1d2 = (0, 0, 0, 0, 0, 0, 0, 1)

        self.p2d1 = (0, 0, 0, 0, 1, 0, 0, 0)
        self.p2d2 = (0, 0, 0, 0, 0, 1, 0, 0)

        self.p3d1 = (0, 0, 1, 0, 0, 0, 0, 0)
        self.p3d2 = (0, 0, 0, 1, 0, 0, 0, 0)

    def apply(self, v):
        dv_dx = F.pad(v[:, 1:, :, :, :], self.p1d1, 'constant', 0) - F.pad(v[:, :-1, :, :, :], self.p1d2, 'constant', 0)
        dv_dy = F.pad(v[:, :, 1:, :, :], self.p2d1, 'constant', 0) - F.pad(v[:, :, :-1, :, :], self.p2d2, 'constant', 0)
        dv_dz = F.pad(v[:, :, :, 1:, :], self.p3d1, 'constant', 0) - F.pad(v[:, :, :, :-1, :], self.p3d2, 'constant', 0)

        return dv_dx, dv_dy, dv_dz


class DataLoss(nn.Module, ABC):
    def __init__(self):
        pass

    def forward(self, im1, im2):
        z = self.map(im1, im2)
        return self.reduce(z)

    @abstractmethod
    def map(self, im1, im2):
        pass

    @abstractmethod
    def reduce(self, z):
        pass


class SSD(DataLoss):
    def map(self, im1, im2):
        return im1 - im2

    def reduce(self, z):
        return torch.sum(z ** 2)

