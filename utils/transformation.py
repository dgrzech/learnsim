from abc import ABC, abstractmethod
from torch import nn

import torch.nn.functional as F


class TransformationModel(nn.Module, ABC):
    """
    abstract class for a transformation model, e.g. B-splines or a stationary velocity field
    """

    def __init__(self):
        super(TransformationModel, self).__init__()

    @abstractmethod
    def forward(self):
        pass


class SVF(TransformationModel):
    """
    stationary velocity field transformation model
    """

    def __init__(self):
        super(SVF, self).__init__()
        self.no_steps = 12

    def forward(self, identity_grid, v):
        """
        integrate an SVF using the scaling and squaring method

        :param identity_grid:
        :param v: velocity field to integrate

        :return: dense warp field
        """

        warp_field = v / (2 ** self.no_steps)

        for _ in range(self.no_steps):
            w = identity_grid + warp_field.permute([0, 2, 3, 4, 1])
            warp_field = warp_field + F.grid_sample(warp_field, w, padding_mode='zeros')

        return warp_field
