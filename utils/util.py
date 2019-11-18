import json
import numpy as np
import pandas as pd
import torch

from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from scipy.linalg import toeplitz


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


def init_identity_grid(im_dim):
    """

    :param sz: size of the grid
    :return: normalised 3D grid
    """

    sz = im_dim[1:]

    nz = sz[0]
    ny = sz[1]
    nx = sz[2]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)
    z = torch.linspace(-1, 1, steps=nz)

    x = x.expand(ny, -1).expand(nz, -1, -1)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

    x.unsqueeze_(0).unsqueeze_(4)
    y.unsqueeze_(0).unsqueeze_(4)
    z.unsqueeze_(0).unsqueeze_(4)

    return torch.cat((x, y, z), 4)


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
