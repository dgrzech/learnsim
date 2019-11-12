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


def integrate_vect(v, no_steps=10):
    assert no_steps >= 0, 'nb_steps should be >=0, found: %d' % no_steps

    v_out = v.permute(0, 4, 1, 2, 3)
    v_out /= (2 ** no_steps)

    for _ in range(no_steps):
        v_next = v_out + F.grid_sample(v_out, v)
        v_out = v_next
        v = v_out.permute(0, 2, 3, 4, 1)

    return v_out.permute(0, 2, 3, 4, 1)


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


class Sampler:
    def __init__(self, device):
        self.device = device
        self.normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def sample_qv(self, v, sigma_voxel_v, u_v):
        dim = v.shape
        epsilon_dim = [dim[4], dim[1], dim[2], dim[3]]
        x_dim = [dim[1], dim[2], dim[3]]

        epsilon = sigma_voxel_v * self.normal_dist.sample(epsilon_dim).squeeze().to(self.device)
        epsilon = epsilon.unsqueeze(0).permute(0, 2, 3, 4, 1)
        x = self.normal_dist.sample(x_dim).to(self.device)

        v_sample = v + epsilon + x * u_v
        return v_sample

    def sample_qf(self, f, sigma_voxel_f, u_f):
        dim = f.shape[1:]

        epsilon = sigma_voxel_f * self.normal_dist.sample(dim).squeeze().to(self.device)
        x = self.normal_dist.sample(dim).squeeze().to(self.device)

        f_sample = f + epsilon + x * u_f
        return f_sample


def generate_grid(sz):
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

    return torch.cat((x, y, z), 4).to(dtype=torch.float32, device='cuda:0')


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
