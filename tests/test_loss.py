import pytest
from torch import nn

from model.loss import EntropyMultivariateNormal, RegLoss_L2
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

    def test_reg_loss_L2(self):
        reg_loss = RegLoss_L2(diff_op='GradientOperator', w_reg=1.0).to(device)

        # init. a batch of uniform 3D velocity fields
        v = torch.zeros(dims_v_batch, device=device)
        v[0, 0, ...] = 5.0
        v[0, 1, ...] = 4.0
        v[0, 2, ...] = 2.0

        # test the loss value
        reg_loss_value = reg_loss(v)
        
        assert len(reg_loss_value.shape) == 1
        assert reg_loss_value.shape[0] == 2
        assert torch.allclose(reg_loss_value, torch.zeros_like(reg_loss_value), atol=atol)

        # init. another batch of velocity fields
        v[1, ...] = torch.rand_like(v[1, ...])

        # test the loss value
        reg_loss_value = reg_loss(v)

        assert len(reg_loss_value.shape) == 1
        assert reg_loss_value.shape[0] == 2
        assert not torch.allclose(reg_loss_value, torch.zeros_like(reg_loss_value), atol=atol)

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

        # initialise a batch
        im_moving_batch = torch.cat((im_moving1, im_moving2), dim=0)
        im_fixed_batch = im_fixed.expand_as(im_moving_batch)
        mask_batch = mask.expand_as(im_moving_batch)
        loss_true_batch = torch.cat((loss_true1, loss_true2), dim=0)

        # test the loss value for different networks
        for no_features in self.no_features:
            for activation in self.activations:
                encoder_LCC = CNN_LCC(learnable=True, s=self.s_LCC,
                                      no_features=no_features, activation=activation).to(device)

                # calculate the loss value for im_moving1
                z = encoder_LCC(im_fixed, im_moving1, mask)
                loss = loss_LCC(z)

                assert len(loss.shape) == 1
                assert loss.shape[0] == 1
                assert torch.allclose(loss, loss_true1, atol=atol)

                # calculate the loss value for im_moving2
                z = encoder_LCC(im_fixed, im_moving2, mask)
                loss = loss_LCC(z)

                assert len(loss.shape) == 1
                assert loss.shape[0] == 1
                assert torch.allclose(loss, loss_true2, atol=atol)

                # calculate the loss value for a batch of size 2
                z = encoder_LCC(im_fixed_batch, im_moving_batch, mask_batch)
                loss = loss_LCC(z)

                assert len(loss.shape) == 1
                assert loss.shape[0] == 2
                assert torch.allclose(loss, loss_true_batch, atol=atol)

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

        # the tolerance is set to 0.01 bc we sample different locations in each forward pass
        assert len(loss1_identical.shape) == 1
        assert len(loss2_identical.shape) == 1
        assert loss1_identical.shape[0] == 1
        assert loss2_identical.shape[0] == 1
        assert torch.allclose(loss1_identical, loss2_identical, atol=1e-2)

        # initialise another image
        im_moving = torch.rand_like(im_fixed)

        # test that I(F, M) == I(M, F)
        z1 = self.encoder_MI(im_fixed, im_moving, mask)
        loss1 = loss_MI(z1)
        z2 = self.encoder_MI(im_moving, im_fixed, mask)
        loss2 = loss_MI(z2)

        assert len(loss1.shape) == 1
        assert len(loss2.shape) == 1
        assert loss1.shape[0] == 1
        assert loss2.shape[0] == 1
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

                assert len(loss1_identical.shape) == 1
                assert len(loss2_identical.shape) == 1
                assert loss1_identical.shape[0] == 1
                assert loss2_identical.shape[0] == 1
                assert torch.allclose(loss1_identical, loss2_identical, atol=1e-2)

                z1 = encoder_MI(im_fixed, im_moving, mask)
                loss1 = loss_MI(z1)
                z2 = encoder_MI(im_moving, im_fixed, mask)
                loss2 = loss_MI(z2)

                assert len(loss1.shape) == 1
                assert len(loss2.shape) == 1
                assert loss1.shape[0] == 1
                assert loss2.shape[0] == 1
                assert torch.allclose(loss1, loss2, atol=1e-2)
                assert loss1_identical < loss1

    def test_SSD(self):
        # initialise the images
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving = torch.rand_like(im_fixed)
        mask = torch.ones_like(im_fixed).bool()

        # get the residual
        res_sq = (im_fixed - im_moving) ** 2
        res_sq_masked = res_sq[mask].unsqueeze(0)
        loss_true = loss_SSD(res_sq_masked)

        assert len(loss_true.shape) == 1
        assert loss_true.shape[0] == 1

        for no_features in self.no_features:
            for activation in self.activations:
                encoder_SSD = CNN_SSD(learnable=True, no_features=no_features, activation=activation).to(device)

                # calculate the residual
                z = encoder_SSD(im_fixed, im_moving, mask)

                assert len(z.shape) == 2
                assert z.shape[0] == 1
                assert z.shape[1] == np.prod(dims)
                assert torch.allclose(z, res_sq_masked, atol=1e-1)

                # calculate the loss value
                loss = loss_SSD(z)

                assert len(loss.shape) == 1
                assert loss.shape[0] == 1
                assert torch.allclose(loss, loss_true, rtol=rtol)
