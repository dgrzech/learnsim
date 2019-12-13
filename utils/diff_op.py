from abc import ABC, abstractmethod
from torch import nn

import torch


class DifferentialOperator(nn.Module, ABC):
    def __init__(self):
        super(DifferentialOperator, self).__init__()

    @abstractmethod
    def apply(self, v):
        return


class GradientOperator(DifferentialOperator):
    def __init__(self):
        super(GradientOperator, self).__init__()

    def apply(self, v):
        dv_dx, dv_dy, dv_dz = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)

        # forward difference in the first entry
        dv_dx[:, :, :, :, 0] = v[:, :, :, :, 1] - v[:, :, :, :, 0]
        dv_dy[:, :, :, 0, :] = v[:, :, :, 1, :] - v[:, :, :, 0, :]
        dv_dz[:, :, 0, :, :] = v[:, :, 1, :, :] - v[:, :, 0, :, :]

        # backward difference in the last entry
        dv_dx[:, :, :, :, -1] = v[:, :, :, :, -1] - v[:, :, :, :, -2]
        dv_dy[:, :, :, -1, :] = v[:, :, :, -1, :] - v[:, :, :, -2, :]
        dv_dz[:, :, -1, :, :] = v[:, :, -1, :, :] - v[:, :, -2, :, :]

        # central differences elsewhere
        dv_dx[:, :, :, :, 1:-1] = 0.5 * (v[:, :, :, :, 2:] - v[:, :, :, :, :-2])
        dv_dy[:, :, :, 1:-1, :] = 0.5 * (v[:, :, :, 2:, :] - v[:, :, :, :-2, :])
        dv_dz[:, :, 1:-1, :, :] = 0.5 * (v[:, :, 2:, :, :] - v[:, :, :-2, :, :])

        return dv_dx, dv_dy, dv_dz
