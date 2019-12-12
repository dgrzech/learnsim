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

        self.pzd1 = [0, 0, 0, 0, 1, 0]
        self.pzd2 = [0, 0, 0, 0, 0, 1]

        self.pyd1 = [0, 0, 1, 0, 0, 0]
        self.pyd2 = [0, 0, 0, 1, 0, 0]

        self.pxd1 = [1, 0, 0, 0, 0, 0]
        self.pxd2 = [0, 1, 0, 0, 0, 0]

    def apply(self, v):
        dv_dz = F.pad(v[:, :, 1:, :, :], self.pzd1, 'constant', 0) - F.pad(v[:, :, :-1, :, :], self.pzd2, 'constant', 0)
        dv_dy = F.pad(v[:, :, :, 1:, :], self.pyd1, 'constant', 0) - F.pad(v[:, :, :, :-1, :], self.pyd2, 'constant', 0)
        dv_dx = F.pad(v[:, :, :, :, 1:], self.pxd1, 'constant', 0) - F.pad(v[:, :, :, :, :-1], self.pxd2, 'constant', 0)

        return dv_dx, dv_dy, dv_dz
