from model.loss import LCC, SSD
from model.model import CNN_LCC, CNN_SSD

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


class EncInitTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

        n = 64
        self.dim_x, self.dim_y, self.dim_z = n, n, n
        self.dims_im = (1, 1, self.dim_x, self.dim_y, self.dim_z)

        """
        modules
        """

        self.s = 4
        self.no_feature_maps = 6

        self.CNN_SSD = CNN_SSD(self.s, self.no_feature_maps).to('cuda:0')

        """
        losses
        """

        self.loss_SSD = SSD().to('cuda:0')
        self.loss_LCC = LCC(self.s).to('cuda:0')

    def tearDown(self):
        del self.CNN_SSD, self.loss_SSD

    def test_init_SSD(self):
        im_fixed = torch.randn(self.dims_im).to('cuda:0')
        im_moving_warped = torch.randn(self.dims_im).to('cuda:0')

        z = self.CNN_SSD(im_fixed, im_moving_warped)

        val = self.loss_SSD(None, None, z).item()
        val_true = self.loss_SSD(im_fixed, im_moving_warped).item()

        assert pytest.approx(val, 0.01) == val_true

    def test_init_LCC(self):
        im_fixed = torch.randn(self.dims_im).to('cuda:0')
        im_moving_warped = torch.randn(self.dims_im).to('cuda:0')

        # single feature map
        CNN_LCC_one_feature_map = CNN_LCC(self.s, 1).to('cuda:0')
        loss_LCC = LCC(self.s).to('cuda:0')

        z = CNN_LCC_one_feature_map(im_fixed, im_moving_warped)

        val = loss_LCC(None, None, z).item()
        val_true = self.loss_LCC(im_fixed, im_moving_warped).item()

        assert -1.0 * val > 0.0
        assert pytest.approx(val, 0.01) == val_true

        # multiple feature maps
        CNN_LCC_multiple_feature_maps = CNN_LCC(self.s, self.no_feature_maps).to('cuda:0')
        z = CNN_LCC_multiple_feature_maps(im_fixed, im_moving_warped)

        val = self.loss_LCC(None, None, z).item()
        val_true = self.loss_LCC(im_fixed, im_moving_warped).item()

        assert -1.0 * val > 0.0
        assert pytest.approx(val, 0.01) == val_true
