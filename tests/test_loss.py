from model.loss import EntropyMultivariateNormal
from model.model import CNN_LCC, CNN_SSD
from .test_setup import *

import pytest


class LossTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

        # encoders
        self.s_LCC, self.no_features_LCC = 2, [4, 8]
        self.no_features_SSD = [4, 8]

        self.encoder_LCC, self.encoder_LCC_learnable = CNN_LCC(learnable=False, s=self.s_LCC).to(device), \
                                                       CNN_LCC(learnable=True, s=self.s_LCC, no_features=self.no_features_LCC).to(device)
        self.encoder_SSD = CNN_SSD(learnable=True, no_features=self.no_features_SSD).to(device)

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
        # initialise the images (perfect correlation)
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving = 4.0 * im_fixed

        mask = torch.ones_like(im_fixed).bool()

        # calculate the loss value
        z = self.encoder_LCC_learnable(im_fixed, im_moving, mask)
        loss = loss_LCC(z).item()
        loss_true = -1.0 * dim_x * dim_y * dim_z
        
        assert pytest.approx(loss, atol) == loss_true
        delattr(self.encoder_LCC_learnable, 'im_fixed')

        # initialise the images (correlated)
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving = im_fixed + torch.rand_like(im_fixed) * 0.1

        # calculate the loss value
        z = self.encoder_LCC_learnable(im_fixed, im_moving, mask)
        loss = loss_LCC(z).item()
        
        z_true = self.encoder_LCC(im_fixed, im_moving, mask)
        loss_true = loss_LCC(z_true).item()
        
        assert pytest.approx(loss, atol) == loss_true

    def test_SSD(self):
        # initialise the images
        im_fixed = torch.rand((1, 1, *dims), device=device)
        im_moving = torch.rand_like(im_fixed)

        mask = torch.ones_like(im_fixed).bool()

        # get the residual
        z = self.encoder_SSD(im_fixed, im_moving, mask)

        res_sq = (im_fixed - im_moving) ** 2
        res_sq_masked = res_sq[mask]

        assert torch.allclose(res_sq_masked, z, atol=atol)

        # get the loss value
        loss = loss_SSD(z)
        loss_true = loss_SSD(res_sq_masked)

        assert torch.allclose(loss_true, loss, rtol=rtol)
