from utils import calc_det_J, init_identity_grid_3d, GradientOperator

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
        print(self._testMethodName + '\n')

        n = 8
        self.dim_x = self.dim_y = self.dim_z = n
        self.dims_v = (1, 3, self.dim_x, self.dim_y, self.dim_z)

        """
        differential operator
        """

        self.diff_op = GradientOperator()

    def tearDown(self):
        del self.diff_op

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

        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)

        """
        test the derivative values are correct
        """

        for idx_z in range(self.dim_z - 1):
            for idx_y in range(self.dim_y - 1):
                for idx_x in range(self.dim_x - 1):
                    dvx_dx_val = nabla_vx[0, 0, idx_z, idx_y, idx_x].item()
                    dvy_dx_val = nabla_vy[0, 0, idx_z, idx_y, idx_x].item()
                    dvz_dx_val = nabla_vz[0, 0, idx_z, idx_y, idx_x].item()

                    dvx_dy_val = nabla_vx[0, 1, idx_z, idx_y, idx_x].item()
                    dvy_dy_val = nabla_vy[0, 1, idx_z, idx_y, idx_x].item()
                    dvz_dy_val = nabla_vz[0, 1, idx_z, idx_y, idx_x].item()

                    dvx_dz_val = nabla_vx[0, 2, idx_z, idx_y, idx_x].item()
                    dvy_dz_val = nabla_vy[0, 2, idx_z, idx_y, idx_x].item()
                    dvz_dz_val = nabla_vz[0, 2, idx_z, idx_y, idx_x].item()

                    assert pytest.approx(dvx_dx_val, 1e-5) == 0.0
                    assert pytest.approx(dvy_dx_val, 1e-5) == 0.0
                    assert pytest.approx(dvz_dx_val, 1e-5) == 0.0

                    assert pytest.approx(dvx_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dvy_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dvz_dy_val, 1e-5) == 0.0

                    assert pytest.approx(dvx_dz_val, 1e-5) == 0.0
                    assert pytest.approx(dvy_dz_val, 1e-5) == 0.0
                    assert pytest.approx(dvz_dz_val, 1e-5) == 0.0

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

        nabla_vx, nabla_vy, nabla_vz = self.diff_op(v)

        """
        test the derivative values are correct
        """

        for idx_z in range(self.dim_z):
            for idx_y in range(self.dim_y):
                for idx_x in range(self.dim_x):
                    dvx_dx_val = nabla_vx[0, 0, idx_z, idx_y, idx_x].item()
                    dvy_dx_val = nabla_vy[0, 0, idx_z, idx_y, idx_x].item()
                    dvz_dx_val = nabla_vz[0, 0, idx_z, idx_y, idx_x].item()

                    dvx_dy_val = nabla_vx[0, 1, idx_z, idx_y, idx_x].item()
                    dvy_dy_val = nabla_vy[0, 1, idx_z, idx_y, idx_x].item()
                    dvz_dy_val = nabla_vz[0, 1, idx_z, idx_y, idx_x].item()

                    dvx_dz_val = nabla_vx[0, 2, idx_z, idx_y, idx_x].item()
                    dvy_dz_val = nabla_vy[0, 2, idx_z, idx_y, idx_x].item()
                    dvz_dz_val = nabla_vz[0, 2, idx_z, idx_y, idx_x].item()

                    if idx_x == self.dim_x - 1 \
                            or idx_y == self.dim_y - 1 \
                            or idx_z == self.dim_z - 1:
                        continue

                    assert pytest.approx(dvx_dx_val, 1e-5) == 1.0
                    assert pytest.approx(dvy_dx_val, 1e-5) == 0.0
                    assert pytest.approx(dvz_dx_val, 1e-5) == 0.0

                    assert pytest.approx(dvx_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dvy_dy_val, 1e-5) == 0.0
                    assert pytest.approx(dvz_dy_val, 1e-5) == 0.0

                    assert pytest.approx(dvx_dz_val, 1e-5) == 0.0
                    assert pytest.approx(dvy_dz_val, 1e-5) == 2.0 * (float(idx_z) + 0.5)
                    assert pytest.approx(dvz_dz_val, 1e-5) == 0.0

    def test_det_J(self):
        """
        initialise a transformation
        """

        nabla_x = torch.zeros(self.dims_v).to('cuda:0')
        nabla_y = torch.zeros(self.dims_v).to('cuda:0')
        nabla_z = torch.zeros(self.dims_v).to('cuda:0')

        det_J_true = torch.zeros((1, 1, self.dim_z, self.dim_y, self.dim_x)).to('cuda:0')

        for idx_z in range(nabla_x.shape[2]):
            for idx_y in range(nabla_x.shape[3]):
                for idx_x in range(nabla_x.shape[4]):
                    dvx_dx = idx_x
                    dvx_dy = idx_z ** 2
                    dvx_dz = idx_y

                    dvy_dx = idx_y
                    dvy_dy = idx_x ** 2
                    dvy_dz = idx_z

                    dvz_dx = idx_y ** 2
                    dvz_dy = idx_x
                    dvz_dz = idx_x

                    nabla_x[0, 0, idx_z, idx_y, idx_x] = dvx_dx
                    nabla_x[0, 1, idx_z, idx_y, idx_x] = dvx_dy
                    nabla_x[0, 2, idx_z, idx_y, idx_x] = dvx_dz

                    nabla_y[0, 0, idx_z, idx_y, idx_x] = dvy_dx
                    nabla_y[0, 1, idx_z, idx_y, idx_x] = dvy_dy
                    nabla_y[0, 2, idx_z, idx_y, idx_x] = dvy_dz

                    nabla_z[0, 0, idx_z, idx_y, idx_x] = dvz_dx
                    nabla_z[0, 1, idx_z, idx_y, idx_x] = dvz_dy
                    nabla_z[0, 2, idx_z, idx_y, idx_x] = dvz_dz

                    # x^4 - x^2 * y^3 - x^2 * z + x * y^2 - x * y * z^2 + y^2 * z^3
                    det_J_true[0, 0, idx_z, idx_y, idx_x] = idx_x ** 4 \
                                                            - idx_x ** 2 * idx_y ** 3 \
                                                            - idx_x ** 2 * idx_z \
                                                            + idx_x * idx_y ** 2 \
                                                            - idx_x * idx_y * idx_z ** 2 \
                                                            + idx_y ** 2 * idx_z ** 3

        """
        calculate its Jacobian
        """

        det_J = calc_det_J(nabla_x, nabla_y, nabla_z)

        """
        test the Jacobian values are correct
        """

        assert torch.all(torch.eq(det_J, det_J_true))

    def test_log_det_J_identity(self):
        """
        initialise the identity transformation
        """

        identity_transformation = init_identity_grid_3d(self.dim_x, self.dim_y, self.dim_z).permute([0, 4, 1, 2, 3])

        """
        calculate its Jacobian
        """

        nabla_x, nabla_y, nabla_z = self.diff_op(identity_transformation)

        nabla_x *= (self.dim_x - 1.0) / 2.0
        nabla_y *= (self.dim_y - 1.0) / 2.0
        nabla_z *= (self.dim_z - 1.0) / 2.0

        """
        calculate the log determinant of the Jacobian
        """

        det_J = calc_det_J(nabla_x, nabla_y, nabla_z) + 1e-5
        log_det_J = torch.log10(det_J)

        """
        test that values of the log determinant of the Jacobian are correct
        """

        for idx_z in range(log_det_J.shape[1]):
            for idx_y in range(log_det_J.shape[2]):
                for idx_x in range(log_det_J.shape[3]):
                    log_det_J_val = log_det_J[0, idx_z, idx_y, idx_x].item()
                    assert pytest.approx(log_det_J_val, abs=1e-5) == 0.0

    def test_log_det_J_stretching(self):
        """
        initialise the transformation
        """

        identity_transformation = init_identity_grid_3d(self.dim_x, self.dim_y, self.dim_z).permute([0, 4, 1, 2, 3])
        transformation = torch.zeros_like(identity_transformation)

        for idx_z in range(transformation.shape[2]):
            for idx_y in range(transformation.shape[3]):
                for idx_x in range(transformation.shape[4]):
                    displacement_x, displacement_y, displacement_z = 2.0 * idx_x / (self.dim_x - 1.0), \
                                                                     2.0 * idx_y / (self.dim_y - 1.0), \
                                                                     2.0 * idx_z / (self.dim_z - 1.0)

                    transformation[0, 0, idx_z, idx_y, idx_x] = identity_transformation[0, 0, idx_z, idx_y, idx_x] + \
                                                                displacement_x
                    transformation[0, 1, idx_z, idx_y, idx_x] = identity_transformation[0, 1, idx_z, idx_y, idx_x] + \
                                                                displacement_y
                    transformation[0, 2, idx_z, idx_y, idx_x] = identity_transformation[0, 2, idx_z, idx_y, idx_x] + \
                                                                displacement_z

        """
        calculate its Jacobian
        """

        nabla_x, nabla_y, nabla_z = self.diff_op(transformation)

        nabla_x *= (self.dim_x - 1.0) / 2.0
        nabla_y *= (self.dim_y - 1.0) / 2.0
        nabla_z *= (self.dim_z - 1.0) / 2.0

        det_J = calc_det_J(nabla_x, nabla_y, nabla_z)

        log_det_J = torch.log10(det_J)
        log_det_J_true = torch.log10(8.0 * torch.ones_like(log_det_J))

        for idx_z in range(log_det_J.shape[1]):
            for idx_y in range(log_det_J.shape[2]):
                for idx_x in range(log_det_J.shape[3]):
                    log_det_J_val = log_det_J[0, idx_z, idx_y, idx_x].item()
                    log_det_J_true_val = log_det_J_true[0, idx_z, idx_y, idx_x].item()

                    assert pytest.approx(log_det_J_val) == log_det_J_true_val
