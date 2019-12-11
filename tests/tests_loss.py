from model.loss import EntropyMultivariateNormal

import math
import numpy as np
import pytest
import torch
import unittest


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


class LossTestMethods(unittest.TestCase):
    def setUp(self):
        n = 2
        self.dims = (n, n, n)

        print(self._testMethodName)

    def test_entropy(self):
        n = 2  # no. of voxels in each dimension
        entropy = EntropyMultivariateNormal()

        # initialise sigma_v
        log_var_v = torch.log(torch.abs(torch.randn(self.dims)))
        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
        var_v = sigma_v ** 2

        # initialise the first mode of variation
        u_v = torch.zeros((n, n, n))

        # calculate the entropy
        val = entropy.forward(log_var_v, u_v).item()
        val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.data.numpy().flatten())))

        assert pytest.approx(val, 0.01) == val_true
