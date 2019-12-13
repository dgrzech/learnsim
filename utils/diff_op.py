from abc import ABC, abstractmethod
from torch import nn

import torch.nn.functional as F


class DifferentialOperator(nn.Module, ABC):
    def __init__(self):
        super(DifferentialOperator, self).__init__()

    @abstractmethod
    def apply(self, v):
        return


class GradientOperator(DifferentialOperator):
    def __init__(self):
        super(GradientOperator, self).__init__()

        self.px = (1, 1, 0, 0, 0, 0)
        self.py = (0, 0, 1, 1, 0, 0)
        self.pz = (0, 0, 0, 0, 1, 1)

    def apply(self, v):
        # central differences
        dv_dx = 0.5 * F.pad(v[:, :, :, :, 2:] - v[:, :, :, :, :-2], self.px, 'replicate')
        dv_dy = 0.5 * F.pad(v[:, :, :, 2:, :] - v[:, :, :, :-2, :], self.py, 'replicate')
        dv_dz = 0.5 * F.pad(v[:, :, 2:, :, :] - v[:, :, :-2, :, :], self.pz, 'replicate')

        return dv_dx, dv_dy, dv_dz
