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

        self.pxd2 = [0, 1, 0, 0, 0, 0]
        self.pyd2 = [0, 0, 0, 1, 0, 0]
        self.pzd2 = [0, 0, 0, 0, 0, 1]

    def apply(self, v):
        dv_dx = F.pad(v[:, :, :, :, 1:] - v[:, :, :, :, :-1], self.pxd2, 'constant', 0)
        dv_dy = F.pad(v[:, :, :, 1:, :] - v[:, :, :, :-1, :], self.pyd2, 'constant', 0)
        dv_dz = F.pad(v[:, :, 1:, :, :] - v[:, :, :-1, :, :], self.pzd2, 'constant', 0)

        return dv_dx, dv_dy, dv_dz
