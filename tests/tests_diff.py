from utils.diff_op import GradientOperator

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


class DiffTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName)

        """
        utils
        """

        n = 8

        self.dim_x = n
        self.dim_y = n
        self.dim_z = n

        self.dims_v = (1, 3, self.dim_x, self.dim_y, self.dim_z)

        """
        differential operator
        """

        self.diff_op = GradientOperator()

    def test_diff_v_uniform(self):
        """
        initialise a uniform velocity field
        """

        v = torch.zeros(self.dims_v).to('cuda:0')

        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    v_x = 5.0
                    v_y = 4.0
                    v_z = 2.0

                    v[0, 0, idx_z, idx_y, idx_x] = v_x
                    v[0, 1, idx_z, idx_y, idx_x] = v_y
                    v[0, 2, idx_z, idx_y, idx_x] = v_z

        """
        calculate its derivative
        """

        dv_dx, dv_dy, dv_dz = self.diff_op.apply(v)

        """
        test the derivative values are correct
        """

        dv_x_dx = dv_dx[0, 0, :, :, :]
        dv_y_dx = dv_dx[0, 1, :, :, :]
        dv_z_dx = dv_dx[0, 2, :, :, :]

        dv_x_dy = dv_dy[0, 0, :, :, :]
        dv_y_dy = dv_dy[0, 1, :, :, :]
        dv_z_dy = dv_dy[0, 2, :, :, :]

        dv_x_dz = dv_dz[0, 0, :, :, :]
        dv_y_dz = dv_dz[0, 1, :, :, :]
        dv_z_dz = dv_dz[0, 2, :, :, :]

        for idx_z in range(self.dim_z - 1):
            for idx_y in range(self.dim_y - 1):
                for idx_x in range(self.dim_x - 1):
                    dv_x_dx_val = dv_x_dx[idx_z, idx_y, idx_x].item()
                    dv_y_dx_val = dv_y_dx[idx_z, idx_y, idx_x].item()
                    dv_z_dx_val = dv_z_dx[idx_z, idx_y, idx_x].item()

                    dv_x_dy_val = dv_x_dy[idx_z, idx_y, idx_x].item()
                    dv_y_dy_val = dv_y_dy[idx_z, idx_y, idx_x].item()
                    dv_z_dy_val = dv_z_dy[idx_z, idx_y, idx_x].item()

                    dv_x_dz_val = dv_x_dz[idx_z, idx_y, idx_x].item()
                    dv_y_dz_val = dv_y_dz[idx_z, idx_y, idx_x].item()
                    dv_z_dz_val = dv_z_dz[idx_z, idx_y, idx_x].item()

                    assert pytest.approx(dv_x_dx_val, 1e-5) == 0.0
                    assert pytest.approx(dv_y_dx_val, 1e-5) == 0.0
                    assert pytest.approx(dv_z_dx_val, 1e-5) == 0.0

                    assert pytest.approx(dv_x_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dv_y_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dv_z_dy_val, 1e-5) == 0.0

                    assert pytest.approx(dv_x_dz_val, 1e-5) == 0.0
                    assert pytest.approx(dv_y_dz_val, 1e-5) == 0.0
                    assert pytest.approx(dv_z_dz_val, 1e-5) == 0.0

    def test_diff_v(self):
        """
        initialise a velocity field
        """

        v = torch.zeros(self.dims_v).to('cuda:0')

        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    v_x = idx_x
                    v_y = idx_z ** 2
                    v_z = 0.0

                    v[0, 0, idx_z, idx_y, idx_x] = v_x
                    v[0, 1, idx_z, idx_y, idx_x] = v_y
                    v[0, 2, idx_z, idx_y, idx_x] = v_z

        """
        calculate its derivative
        """

        dv_dx, dv_dy, dv_dz = self.diff_op.apply(v)

        """
        test the derivative values are correct 
        """

        dv_x_dx = dv_dx[0, 0, :, :, :]
        dv_y_dx = dv_dx[0, 1, :, :, :]
        dv_z_dx = dv_dx[0, 2, :, :, :]

        dv_x_dy = dv_dy[0, 0, :, :, :]
        dv_y_dy = dv_dy[0, 1, :, :, :]
        dv_z_dy = dv_dy[0, 2, :, :, :]

        dv_x_dz = dv_dz[0, 0, :, :, :]
        dv_y_dz = dv_dz[0, 1, :, :, :]
        dv_z_dz = dv_dz[0, 2, :, :, :]

        for idx_z in range(self.dim_z):
            for idx_y in range(self.dim_y):
                for idx_x in range(self.dim_z):
                    dv_x_dx_val = dv_x_dx[idx_z, idx_y, idx_x].item()
                    dv_y_dx_val = dv_y_dx[idx_z, idx_y, idx_x].item()
                    dv_z_dx_val = dv_z_dx[idx_z, idx_y, idx_x].item()

                    dv_x_dy_val = dv_x_dy[idx_z, idx_y, idx_x].item()
                    dv_y_dy_val = dv_y_dy[idx_z, idx_y, idx_x].item()
                    dv_z_dy_val = dv_z_dy[idx_z, idx_y, idx_x].item()

                    dv_x_dz_val = dv_x_dz[idx_z, idx_y, idx_x].item()
                    dv_y_dz_val = dv_y_dz[idx_z, idx_y, idx_x].item()
                    dv_z_dz_val = dv_z_dz[idx_z, idx_y, idx_x].item()

                    # print(idx_x, idx_y, idx_z)
                    #
                    # print('v: ', v[0, :, idx_z, idx_y, idx_x])
                    # print('dv_dx: ', dv_x_dx_val, dv_y_dx_val, dv_z_dx_val,
                    #       ', dv_dy: ', dv_x_dy_val, dv_y_dy_val, dv_z_dy_val,
                    #       ', dv_dz: ', dv_x_dz_val, dv_y_dz_val, dv_z_dz_val)

                    if idx_x == 0 or idx_x == self.dim_x - 1 \
                            or idx_y == 0 or idx_y == self.dim_y - 1 \
                            or idx_z == 0 or idx_z == self.dim_z - 1:
                        continue

                    assert pytest.approx(dv_x_dx_val, 1e-5) == 1.0
                    assert pytest.approx(dv_y_dx_val, 1e-5) == 0.0
                    assert pytest.approx(dv_z_dx_val, 1e-5) == 0.0

                    assert pytest.approx(dv_x_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dv_y_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dv_z_dy_val, 1e-5) == 0.0

                    assert pytest.approx(dv_x_dz_val, 1e-5) == 0.0
                    assert pytest.approx(dv_y_dz_val, 1e-5) == 2.0 * idx_z
                    assert pytest.approx(dv_z_dz_val, 1e-5) == 0.0
