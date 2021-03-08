import math
import unittest

import numpy as np
import pytest
import torch

from model.loss import EntropyMultivariateNormal, LCC, SSD
from model.model import CNN_LCC, CNN_SSD

# fix random seeds for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LossTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')
        self.atol = 1e-2
        self.rtol = 1e-4
        self.device = 'cuda:0'

        n = 64
        self.dim_x = self.dim_y = self.dim_z = n
        self.dims = (n, ) * 3

        # entropy
        self.entropy = EntropyMultivariateNormal().to(self.device)

        # LCC
        self.s_LCC, self.no_features_LCC = 2, [8, 16, 16]
        self.encoder_LCC = CNN_LCC(learnable=True, s=self.s_LCC, no_features=self.no_features_LCC).to(self.device)
        self.loss_LCC = LCC().to(self.device)

        self.encoder_LCC_true = CNN_LCC(learnable=False, s=self.s_LCC).to(self.device)

        # SSD
        self.no_features_SSD = [8, 16, 16]
        self.encoder_SSD = CNN_SSD(learnable=True, no_features=self.no_features_SSD).to(self.device)
        self.loss_SSD = SSD().to(self.device)

    def tearDown(self):
        del self.encoder_LCC, self.encoder_SSD

    def test_entropy(self):
        dims = (4, 4, 4)

        # initialise variance to 1 and the first mode of variation to zero
        log_var_v = torch.log(torch.ones((1, 1, *dims), device=self.device))
        u_v = torch.zeros((1, 1, *dims), device=self.device)

        # calculate the entropy
        val = self.entropy(log_var=log_var_v, u=u_v).item()
        val_true = 0.0
        assert pytest.approx(val, self.atol) == val_true

        # initialise variance randomly
        log_var_v = torch.log(torch.abs(torch.randn((1, 1, *dims), device=self.device)))
        var_v = torch.exp(log_var_v)
        sigma_v = torch.sqrt(var_v)

        # calculate the entropy
        val = self.entropy(log_var=log_var_v, u=u_v).item()
        val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.cpu().numpy().flatten())))

        assert pytest.approx(val, self.atol) == val_true

    def test_LCC(self):
        # initialise the images (perfect correlation)
        im_fixed = torch.rand((1, 1, *self.dims), device=self.device)
        im_moving = 4.0 * im_fixed

        mask = torch.ones_like(im_fixed).bool()

        # calculate the loss value
        z = self.encoder_LCC(im_fixed, im_moving, mask)
        loss = self.loss_LCC(z).item()
        loss_true = -1.0 * self.dim_x * self.dim_y * self.dim_z
        
        assert pytest.approx(loss, self.atol) == loss_true

        self.encoder_LCC.disable_grads()
        self.encoder_LCC.enable_grads()

        # initialise the images (correlated)
        im_fixed = torch.rand((1, 1, *self.dims), device=self.device)
        im_moving = im_fixed + torch.rand_like(im_fixed) * 0.1

        # calculate the loss value
        z = self.encoder_LCC(im_fixed, im_moving, mask)
        loss = self.loss_LCC(z).item()
        
        z_true = self.encoder_LCC_true(im_fixed, im_moving, mask)
        loss_true = self.loss_LCC(z_true).item()
        
        assert pytest.approx(loss, self.atol) == loss_true

    def test_SSD(self):
        # initialise the images
        im_fixed = torch.rand((1, 1, *self.dims), device=self.device)
        im_moving = torch.rand_like(im_fixed)

        mask = torch.ones_like(im_fixed).bool()

        # get the residual
        z = self.encoder_SSD(im_fixed, im_moving, mask)

        res_sq = (im_fixed - im_moving) ** 2
        res_sq_masked = res_sq[mask]

        assert torch.allclose(res_sq_masked, z, atol=1e-1)

        # get the loss value
        loss = self.loss_SSD(z)
        loss_true = self.loss_SSD(res_sq_masked)
        assert torch.allclose(loss_true, loss, rtol=self.rtol)
