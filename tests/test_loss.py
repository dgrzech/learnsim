import pytest
from torch import nn

from model.loss import EntropyMultivariateNormal
from model.model import CNN_LCC, CNN_MI, CNN_SSD
from .test_setup import *


class LossTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

        # network parameters
        self.s_LCC = 2
        self.activations = [nn.Identity(), nn.LeakyReLU(negative_slope=0.2), nn.ELU()]
        self.no_features = [[4, 8], [4, 8, 8], [8, 16, 16], [16, 32, 32, 32]]

        self.encoder_LCC = CNN_LCC(learnable=False, s=self.s_LCC).to(device)
        self.encoder_MI = CNN_MI(learnable=False).to(device)

    def test_entropy(self):
        entropy = EntropyMultivariateNormal().to(device)

        # initialise variance to 1 and the first mode of variation to zero
        log_var_v = torch.log(torch.ones((1, 1, *dims_small), device=device))
        u_v = torch.zeros((1, 1, *dims_small), device=device)

        # calculate the entropy
        val = entropy(log_var=log_var_v, u=u_v).item()
        val_true = 0.0
        assert pytest.approx(val, atol) == val_true

        # initialise variance randomly
        log_var_v = torch.log(torch.abs(torch.randn((1, 1, *dims_small), device=device)))
        var_v = torch.exp(log_var_v)

        # calculate the entropy
        val = entropy(log_var=log_var_v, u=u_v).item()
        val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.cpu().numpy().flatten())))
        assert pytest.approx(val, atol) == val_true

    def test_LCC(self):
        # initialise perfectly correlated images
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving1 = 4.0 * im_fixed
        mask = torch.ones_like(im_fixed).bool()
        loss_true1 = -1.0 * dim_x * dim_y * dim_z * torch.ones(1, device=device)

        # initialise correlated images
        im_moving2 = im_fixed + torch.rand_like(im_fixed) * 0.1
        z_true2 = self.encoder_LCC(im_fixed, im_moving2, mask)
        loss_true2 = loss_LCC(z_true2)

        # test the loss value for different networks
        for no_features in self.no_features:
            for activation in self.activations:
                encoder_LCC = CNN_LCC(learnable=True, s=self.s_LCC,
                                      no_features=no_features, activation=activation).to(device)

                # calculate the loss value for im_moving1
                z = encoder_LCC(im_fixed, im_moving1, mask)
                loss = loss_LCC(z)
                assert torch.allclose(loss, loss_true1, atol=atol)

                # calculate the loss value for im_moving2
                z = encoder_LCC(im_fixed, im_moving2, mask)
                loss = loss_LCC(z)
                assert torch.allclose(loss, loss_true2, atol=atol)

    def test_MI(self):
        # initialise the images
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving_identical = im_fixed.clone()
        mask = torch.ones_like(im_fixed).bool()

        # test that I(F, M) == I(M, F)
        z1 = self.encoder_MI(im_fixed, im_moving_identical, mask)
        loss1_identical = loss_MI(z1)
        z2 = self.encoder_MI(im_moving_identical, im_fixed, mask)
        loss2_identical = loss_MI(z2)
        # NOTE (DG): 0.01 bc we sample different locations in each forward pass
        assert torch.allclose(loss1_identical, loss2_identical, atol=1e-2)

        # initialise another image
        im_moving = torch.rand_like(im_fixed)

        # test that I(F, M) == I(M, F)
        z1 = self.encoder_MI(im_fixed, im_moving, mask)
        loss1 = loss_MI(z1)
        z2 = self.encoder_MI(im_moving, im_fixed, mask)
        loss2 = loss_MI(z2)

        assert torch.allclose(loss1, loss2, atol=1e-2)
        assert loss1_identical < loss1

        # test for different networks
        for no_features in self.no_features:
            for activation in self.activations:
                encoder_MI = CNN_MI(learnable=True,
                                    no_features=no_features, activation=activation).to(device)

                # calculate the loss values
                z1 = encoder_MI(im_fixed, im_moving_identical, mask)
                loss1_identical = loss_MI(z1)
                z2 = encoder_MI(im_moving_identical, im_fixed, mask)
                loss2_identical = loss_MI(z2)
                assert torch.allclose(loss1_identical, loss2_identical, atol=1e-2)

                z1 = encoder_MI(im_fixed, im_moving, mask)
                loss1 = loss_MI(z1)
                z2 = encoder_MI(im_moving, im_fixed, mask)
                loss2 = loss_MI(z2)

                assert torch.allclose(loss1, loss2, atol=1e-2)
                assert loss1_identical < loss1

    def test_SSD(self):
        # initialise the images
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving = torch.rand_like(im_fixed)
        mask = torch.ones_like(im_fixed).bool()

        # get the residual
        res_sq = (im_fixed - im_moving) ** 2
        res_sq_masked = res_sq[mask]
        loss_true = loss_SSD(res_sq_masked)

        for no_features in self.no_features:
            for activation in self.activations:
                encoder_SSD = CNN_SSD(learnable=True, no_features=no_features, activation=activation).to(device)

                # calculate the residual
                z = encoder_SSD(im_fixed, im_moving, mask)
                assert torch.allclose(z, res_sq_masked, atol=1e-1)

                # calculate the loss value
                loss = loss_SSD(z)
                assert torch.allclose(loss, loss_true, rtol=rtol)
