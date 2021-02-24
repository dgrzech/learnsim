import math
import unittest

import numpy as np
import pytest
import torch
from model.loss import EntropyMultivariateNormal, LCC, SSD
from model.model import CNN_LCC, CNN_SSD

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


class LossTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

        n = 8
        self.dim_x = self.dim_y = self.dim_z = n
        self.dims = (n, n, n)

        # entropy
        self.entropy = EntropyMultivariateNormal().to('cuda:0')

        # LCC
        self.s_LCC = 2
        self.no_feature_maps_LCC = 8

        self.encoder_LCC = CNN_LCC(learnable=True, s=self.s_LCC, no_feature_maps=self.no_feature_maps_LCC).to('cuda:0')
        self.loss_LCC = LCC().to('cuda:0')

        # SSD
        self.s_SSD = 2
        self.no_feature_maps_SSD = 12

        self.encoder_SSD = CNN_SSD(learnable=True, s=self.s_SSD, no_feature_maps=self.no_feature_maps_SSD).to('cuda:0')
        self.loss_SSD = SSD().to('cuda:0')

    def tearDown(self):
        del self.encoder_LCC, self.encoder_SSD

    def test_entropy(self):
        # initialise variance to 1 and the first mode of variation to zero
        log_var_v = torch.log(torch.ones(1, 1, *self.dims)).to('cuda:0')
        u_v = torch.zeros(1, 1, *self.dims).to('cuda:0')

        # calculate the entropy
        val = self.entropy(log_var=log_var_v, u=u_v).item()
        val_true = 0.0
        assert pytest.approx(val, 0.01) == val_true

        # initialise variance randomly
        log_var_v = torch.log(torch.abs(torch.randn(1, 1, *self.dims))).to('cuda:0')
        sigma_v = torch.exp(0.5 * log_var_v)

        # calculate the entropy
        val = self.entropy(log_var=log_var_v, u=u_v).item()

        var_v = sigma_v ** 2
        val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.cpu().numpy().flatten())))
        assert pytest.approx(val, 0.01) == val_true

    def test_LCC(self):
        # initialise the images
        im_fixed = torch.randn(1, 1, *self.dims).to('cuda:0')
        im_moving = 4.0 * im_fixed

        mask = torch.ones_like(im_fixed).bool()

        # calculate the loss value
        z = self.encoder_LCC(im_fixed, im_moving, mask)
        loss = self.loss_LCC(z).item()
        loss_true = -1.0 * self.dim_x * self.dim_y * self.dim_z

        assert pytest.approx(loss, 0.01) == loss_true

    def test_SSD(self):
        # initialise the images
        im_fixed = torch.randn(1, 1, *self.dims).to('cuda:0')
        im_moving = torch.randn_like(im_fixed)

        mask = torch.ones_like(im_fixed).bool()

        # get the residual
        z = self.encoder_SSD(im_fixed, im_moving, mask)

        res_sq = (im_fixed - im_moving) ** 2
        res_sq_masked = res_sq[mask]

        assert torch.allclose(res_sq_masked, z, atol=0.01)

        # get the loss value
        loss = self.loss_SSD(z)
        loss_true = self.loss_SSD(res_sq_masked)

        assert torch.allclose(loss_true, loss, atol=0.01)
