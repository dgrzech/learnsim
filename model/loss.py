from abc import ABC, abstractmethod

import torch
from torch import nn

from utils import DifferentialOperator


"""
data loss
"""


class DataLoss(nn.Module, ABC):
    """
    base class for all data losses
    """

    def __init__(self):
        super(DataLoss, self).__init__()

    @abstractmethod
    def forward(self, z):
        pass


class LCC(DataLoss):
    """
    local cross-correlation
    """

    def __init__(self):
        super(LCC, self).__init__()

    def forward(self, z):
        return -1.0 * z.sum(dim=0, keepdim=True)


class SSD(DataLoss):
    """
    sum of squared differences
    """

    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, z):
        return 0.5 * z.sum(dim=0, keepdim=True)


"""
regularisation loss
"""


class RegLoss(nn.Module, ABC):
    """
    base class for all regularisation losses.
    
    diff_op: takes an input tensor and outputs a tensor, the result of the action of diff_op on the input.
    Or strings that can be parsed to a diff_op
    """

    def __init__(self, diff_op=None):
        super(RegLoss, self).__init__()

        if diff_op is not None:
            if isinstance(diff_op, str):
                # create from string, e.g. diff_op='GradientOperator'
                self.diff_op = DifferentialOperator.from_string(diff_op)
            else:
                if isinstance(diff_op, DifferentialOperator):
                    # point to passed instance, e.g. diff_op=GradientOperator()
                    self.diff_op = diff_op
                else:
                    # create instance from class name, e.g. diff_op=GradientOperator
                    self.diff_op = diff_op()
        else:
            # identity operator
            self.diff_op = DifferentialOperator()

    def forward(self, input, *args, **kwargs):
        """
        All RegLosses operate as functions of the energy defined by the differential operator.
        This is not meant to be overriden, just define the proper diff_op and _loss.
        """

        D_input = self.diff_op(input)
        y = D_input ** 2
        y = y.mean(dim=(1, 2, 3, 4, 5))  # "chi-square" variable / energy

        return self._loss(y)

    @abstractmethod
    def _loss(self, y):
        """
        Override this in children classes. For Bayesian RegLosses, this _loss is of the form 
            - log_pdf(...) and potentially includes hyperpriors for tunable parameters.
        """
        pass


class RegLoss_L2(RegLoss):
    """
    This implements all spatial and Fourier logGaussian losses, by passing
    1) a suitable differential operator
    2) a suitable input to forward (implemented in parent class)
    
    Example 1. Penalty on the squared Froebenius norm of the Jacobian of the velocity, implemented in spatial domain:
        diff_op = GradientOperator
        input = v.
    Example 2. Smooth Sobolev space equivalent of the above.
        diff_op = GradientOperator
        v = K^{1/2} alpha is the (Sobolev-smooth) velocity field obtained from its L2-preimage alpha
        input = alpha
    Example 3. Equivalent of example 2, but implemented in frequency domain.
        diff_op = Fourier1stDerivativeOperator
        alpha is an L2 preimage of the velocity, it lives in the frequency domain.
        v = fft(half_kernel_hat * alpha * 1 / sqrt(|omega|**2 + 1)) is the (Sobolev-smooth) velocity field
    """

    def __init__(self, w_reg, diff_op=None):
        super(RegLoss_L2, self).__init__(diff_op=diff_op)
        self.w_reg = w_reg

    def _loss(self, y):
        return 0.5 * self.w_reg * y.sum(dim=0, keepdim=True)


"""
entropy
"""


class Entropy(nn.Module, ABC):
    """
    base class for the entropy of a probability distribution
    """

    def __init__(self):
        super(Entropy, self).__init__()

    @abstractmethod
    def forward(self, **kwargs):
        pass


class EntropyMultivariateNormal(Entropy):
    """
    multivariate normal distribution
    """

    def __init__(self):
        super(EntropyMultivariateNormal, self).__init__()

    def forward(self, **kwargs):
        if len(kwargs) == 2:
            log_var, u = kwargs['log_var'], kwargs['u']
            sigma = torch.exp(0.5 * log_var)
            no_voxels = torch.numel(sigma[1:])

            val = 0.5 * (torch.log1p(torch.sum(torch.pow(u / sigma, 2), dim=(1, 2, 3, 4))) + torch.sum(log_var, dim=(1, 2, 3, 4)))
            val_mean = val / no_voxels

            return val_mean.sum(dim=0, keepdim=True)
        elif len(kwargs) == 4:
            sample, mu, log_var, u = kwargs['sample'], kwargs['mu'], kwargs['log_var'], kwargs['u']
            sigma = torch.exp(0.5 * log_var)
            no_voxels = torch.numel(sigma[1:])

            sample_n, u_n = (sample - mu) / sigma, u / sigma

            t1 = torch.sum(torch.pow(sample_n, 2), dim=(1, 2, 3, 4))
            t2 = torch.pow(torch.sum(sample_n * u_n, dim=(1, 2, 3, 4)), 2) / (1.0 + torch.sum(torch.pow(u_n, 2), dim=(1, 2, 3, 4)))

            val = 0.5 * (t1 - t2)
            val_mean = val / no_voxels

            return val_mean.sum(dim=0, keepdim=True)

        raise NotImplementedError
