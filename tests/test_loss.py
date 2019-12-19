from model.loss import EntropyMultivariateNormal, LCC

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
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
        print(self._testMethodName)

        n = 4

        self.dim_x = n
        self.dim_y = n
        self.dim_z = n

        self.s = 4
        self.padding = (self.s, self.s, self.s, self.s, self.s, self.s)

    def test_entropy(self):
        # initialise the loss object
        entropy = EntropyMultivariateNormal()

        # initialise sigma_v
        log_var_v = torch.log(torch.abs(torch.randn(self.dim_x, self.dim_y, self.dim_z)))
        sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
        var_v = sigma_v ** 2

        # initialise the first mode of variation
        u_v = torch.zeros((self.dim_x, self.dim_y, self.dim_z))

        # calculate the entropy
        val = entropy.forward(log_var_v, u_v).item()
        val_true = -0.5 * math.log(np.linalg.det(np.diag(var_v.cpu().numpy().flatten())))

        assert pytest.approx(val, 0.01) == val_true

    def test_lcc_uniform(self):
        # initialise the images
        im_fixed = torch.zeros(1, 1, self.dim_x, self.dim_y, self.dim_z).to('cuda:0')
        im_moving = 2.0 * torch.ones(1, 1, self.dim_x, self.dim_y, self.dim_z).to('cuda:0')

        # initialise the loss object
        lcc = LCC(self.s)

        # calculate the local means
        im_fixed_padded = F.pad(im_fixed, self.padding, mode='replicate')
        im_moving_padded = F.pad(im_moving, self.padding, mode='replicate')

        u_F, u_M = lcc.compute_local_means(im_fixed_padded, im_moving_padded)

        assert torch.all(torch.eq(u_F, im_fixed))
        assert torch.all(torch.eq(u_M, im_moving))

        # calculate the local sums
        cross, var_F, var_M = lcc.compute_local_sums(im_fixed_padded, im_moving_padded)

        zero_tensor = torch.zeros(1, 1, self.dim_x, self.dim_y, self.dim_z).to('cuda:0')
        assert torch.all(torch.eq(cross, zero_tensor))
        assert torch.all(torch.eq(var_F, zero_tensor))
        assert torch.all(torch.eq(var_M, zero_tensor))

        # calculate the value of loss
        val = lcc(im_fixed, im_moving).item()
        val_true = 0.0

        assert pytest.approx(val, 0.01) == val_true

    def test_lcc(self):
        # initialise the images
        im_fixed = torch.randn(1, 1, self.dim_x, self.dim_y, self.dim_z).to('cuda:0')
        im_moving = 4.0 * im_fixed

        # initialise the loss object
        lcc = LCC(self.s)

        # calculate the local means
        im_fixed_padded = F.pad(im_fixed, self.padding, mode='replicate')
        im_moving_padded = F.pad(im_moving, self.padding, mode='replicate')

        u_F, u_M = lcc.compute_local_means(im_fixed_padded, im_moving_padded)
        assert torch.all(torch.eq(4.0 * u_F, u_M))

        # calculate the local sums
        cross, var_F, var_M = lcc.compute_local_sums(im_fixed_padded, im_moving_padded)
        assert torch.all(torch.eq(16.0 * var_F, var_M))

        # calculate the value of loss
        val = lcc(im_fixed, im_moving).item()
        no_voxels = float(self.dim_x * self.dim_y * self.dim_z)

        assert pytest.approx(val, 0.01) == -1.0 * no_voxels
