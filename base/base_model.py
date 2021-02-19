from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    base class for all models
    """

    def __init__(self, learnable):
        super(BaseModel, self).__init__()
        self.learnable = learnable

    @abstractmethod
    def encode(self, im_fixed, im_moving):
        pass

    @abstractmethod
    def print_weights(self):
        pass

    def forward(self, im_fixed, im_moving, mask):
        """
        forward pass logic

        :return: Model output
        """

        z = self.encode(im_fixed, im_moving)
        return z[mask]

    def __str__(self):
        """
        model prints with number of trainable parameters
        """

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
