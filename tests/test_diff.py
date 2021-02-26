import unittest

import numpy as np
import torch

from utils import GradientOperator, calc_det_J, init_identity_grid_3D

# fix random seeds for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DiffTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')
        self.atol = 1e-4
        self.device = 'cuda:0'

        n = 64
        self.dim_x = self.dim_y = self.dim_z = n
        self.dims = (n, ) * 3
        self.dims_v = (1, 3, *self.dims)

        self.diff_op = GradientOperator()  # differential operator

    def tearDown(self):
        del self.diff_op

    def test_diff_v_uniform(self):
        # init. a uniform 3D velocity field
        v = torch.zeros(self.dims_v, device=self.device)
        v[0, 0, ...] = 5.0
        v[0, 1, ...] = 4.0
        v[0, 2, ...] = 2.0

        # calculate its derivatives
        nabla_v = self.diff_op(v)
        nabla_vx, nabla_vy, nabla_vz = nabla_v[..., 0], nabla_v[..., 1], nabla_v[..., 2]

        # test that they're correct
        assert torch.allclose(nabla_vx, torch.zeros_like(v), atol=self.atol)
        assert torch.allclose(nabla_vy, torch.zeros_like(v), atol=self.atol)
        assert torch.allclose(nabla_vz, torch.zeros_like(v), atol=self.atol)

    def test_diff_v(self):
        # init. a uniform 3D velocity field
        v = torch.zeros(self.dims_v, device=self.device)

        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    v[0, 0, idx_z, idx_y, idx_x] = idx_x
                    v[0, 1, idx_z, idx_y, idx_x] = 1.5 * idx_y + 3.0 * idx_z + 1.0
                    v[0, 2, idx_z, idx_y, idx_x] = 0.0

        # calculate its derivatives
        nabla_v = self.diff_op(v)
        nabla_vx, nabla_vy, nabla_vz = nabla_v[..., 0], nabla_v[..., 1], nabla_v[..., 2]

        # test that they're correct
        nabla_vx_true, nabla_vy_true, nabla_vz_true = torch.zeros_like(nabla_vx), torch.zeros_like(nabla_vy), torch.zeros_like(nabla_vz)

        nabla_vx_true[:, 0] = 1.0
        nabla_vy_true[:, 1] = 1.5
        nabla_vy_true[:, 2] = 3.0

        assert torch.allclose(nabla_vx, nabla_vx_true, atol=self.atol)
        assert torch.allclose(nabla_vy, nabla_vy_true, atol=self.atol)
        assert torch.allclose(nabla_vz, nabla_vz_true, atol=self.atol)

    def test_log_det_J_identity(self):
        # initialise an identity transformation
        identity_transformation = init_identity_grid_3D(self.dims).permute([0, 4, 1, 2, 3]).to(self.device)

        # calculate its Jacobian
        nabla = self.diff_op(identity_transformation, transformation=True)
        log_det_J = torch.log(calc_det_J(nabla) + 1e-5)

        # test that the values are correct
        assert torch.allclose(log_det_J, torch.zeros_like(log_det_J), atol=self.atol)

    def test_det_J(self):
        # initialise a transformation
        nabla_x = torch.zeros(self.dims_v, device=self.device)
        nabla_y, nabla_z = torch.zeros_like(nabla_x), torch.zeros_like(nabla_x)

        for idx_z in range(nabla_x.shape[2]):
            for idx_y in range(nabla_x.shape[3]):
                for idx_x in range(nabla_x.shape[4]):
                    dvx_dx, dvx_dy, dvx_dz = idx_x, idx_z ** 2, idx_y
                    dvy_dx, dvy_dy, dvy_dz = idx_y, idx_x ** 2, idx_z
                    dvz_dx, dvz_dy, dvz_dz = idx_y ** 2, idx_x, idx_x

                    nabla_x[0, :, idx_z, idx_y, idx_x] = torch.tensor([dvx_dx, dvx_dy, dvx_dz], device=self.device)
                    nabla_y[0, :, idx_z, idx_y, idx_x] = torch.tensor([dvy_dx, dvy_dy, dvy_dz], device=self.device)
                    nabla_z[0, :, idx_z, idx_y, idx_x] = torch.tensor([dvz_dx, dvz_dy, dvz_dz], device=self.device)

        # calculate its Jacobian
        nabla = torch.stack([nabla_x, nabla_y, nabla_z], dim=-1)
        det_J = calc_det_J(nabla)

        det_J_true = torch.zeros((1, 1, *self.dims), device=self.device)

        for idx_z in range(nabla_x.shape[2]):
            for idx_y in range(nabla_x.shape[3]):
                for idx_x in range(nabla_x.shape[4]):
                    # x^4 - x^2 * y^3 - x^2 * z + x * y^2 - x * y * z^2 + y^2 * z^3
                    det_J_true[..., idx_z, idx_y, idx_x] = idx_x ** 4 - idx_x ** 2 * idx_y ** 3 - idx_x ** 2 * idx_z \
                                                           + idx_x * idx_y ** 2 - idx_x * idx_y * idx_z ** 2 \
                                                           + idx_y ** 2 * idx_z ** 3

        # test that the values are correct
        assert torch.allclose(det_J, det_J_true, atol=self.atol)

    def test_log_det_J_stretching(self):
        # initialise a transformation
        identity_transformation = init_identity_grid_3D(self.dims).permute([0, 4, 1, 2, 3]).to(self.device)
        displacement = torch.zeros_like(identity_transformation)

        for idx_z in range(displacement.shape[2]):
            for idx_y in range(displacement.shape[3]):
                for idx_x in range(displacement.shape[4]):
                    d = torch.tensor([2.0 * idx_x / (self.dim_x - 1.0), 2.0 * idx_y / (self.dim_y - 1.0),
                                      2.0 * idx_z / (self.dim_z - 1.0)], device=self.device)
                    displacement[0, :, idx_z, idx_y, idx_x] = d

        transformation = identity_transformation + displacement

        # calcualte its Jacobian
        nabla = self.diff_op(transformation, transformation=True)

        log_det_J = torch.log(calc_det_J(nabla))
        log_det_J_true = torch.log(8.0 * torch.ones_like(log_det_J))

        # test that the values are correct
        assert torch.allclose(log_det_J, log_det_J_true, atol=self.atol)
