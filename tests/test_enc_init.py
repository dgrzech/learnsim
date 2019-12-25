from model.loss import LCC, SSD
from model.model import CNN_LCC, CNN_SSD
from utils import compute_lcc

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


class EncInitTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName)

        """
        utils
        """

        n = 64
        self.dim_x = n
        self.dim_y = n
        self.dim_z = n

        self.dims_im = (1, 1, self.dim_x, self.dim_y, self.dim_z)

        """
        modules
        """

        self.s = 5
        self.padding = (self.s, self.s, self.s, self.s, self.s, self.s)

        self.CNN_SSD = CNN_SSD(self.s).to('cuda:0')
        self.CNN_LCC = CNN_LCC(self.s).to('cuda:0')

        """
        losses
        """

        self.loss_SSD = SSD()
        self.loss_LCC = LCC()

    def test_init_SSD(self):
        im_fixed = torch.randn(self.dims_im).to('cuda:0')
        im_moving_warped = torch.randn(self.dims_im).to('cuda:0')

        z = self.CNN_SSD(im_fixed, im_moving_warped)

        val = self.loss_SSD(z).item()
        val_true = 0.5 * torch.sum(torch.pow(im_fixed - im_moving_warped, 2.0)).item()
        
        assert pytest.approx(val, 0.01) == val_true

    def test_init_LCC(self):
        im_fixed = torch.randn(self.dims_im).to('cuda:0')
        im_moving_warped = torch.randn(self.dims_im).to('cuda:0')

        z = self.CNN_LCC(im_fixed, im_moving_warped)
 
        val = self.loss_LCC(z).item()
        val_true = compute_lcc(im_fixed, im_moving_warped).item()

        assert -1.0 * val > 0.0
        assert pytest.approx(val, 0.01) == val_true
