from .test_setup import *
from utils import calc_det_J


class DiffTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

    def test_diff_v_uniform(self):
        # init. a uniform 3D velocity field
        v = torch.zeros(dims_v, device=device)
        v[0, 0, ...] = 5.0
        v[0, 1, ...] = 4.0
        v[0, 2, ...] = 2.0

        # calculate its derivatives
        nabla_v = diff_op(v)
        nabla_vx, nabla_vy, nabla_vz = nabla_v[..., 0], nabla_v[..., 1], nabla_v[..., 2]

        # test that they're correct
        assert torch.allclose(nabla_vx, torch.zeros_like(v), atol=atol)
        assert torch.allclose(nabla_vy, torch.zeros_like(v), atol=atol)
        assert torch.allclose(nabla_vz, torch.zeros_like(v), atol=atol)

    def test_diff_v(self):
        # init. a uniform 3D velocity field
        v = torch.zeros(dims_v, device=device)

        for idx_z in range(v.shape[2]):
            for idx_y in range(v.shape[3]):
                for idx_x in range(v.shape[4]):
                    v[0, 0, idx_z, idx_y, idx_x] = idx_x
                    v[0, 1, idx_z, idx_y, idx_x] = 1.5 * idx_y + 3.0 * idx_z + 1.0
                    v[0, 2, idx_z, idx_y, idx_x] = 0.0

        # calculate its derivatives
        nabla_v = diff_op(v)
        nabla_vx, nabla_vy, nabla_vz = nabla_v[..., 0], nabla_v[..., 1], nabla_v[..., 2]

        # test that they're correct
        nabla_vx_true, nabla_vy_true, nabla_vz_true = torch.zeros_like(nabla_vx), torch.zeros_like(nabla_vy), torch.zeros_like(nabla_vz)

        nabla_vx_true[:, 0] = 1.0
        nabla_vy_true[:, 1] = 1.5
        nabla_vy_true[:, 2] = 3.0

        assert torch.allclose(nabla_vx, nabla_vx_true, atol=atol)
        assert torch.allclose(nabla_vy, nabla_vy_true, atol=atol)
        assert torch.allclose(nabla_vz, nabla_vz_true, atol=atol)

    def test_log_det_J_identity(self):
        # initialise an identity transformation
        identity_transformation = identity_grid.permute([0, 4, 1, 2, 3])

        # calculate its Jacobian
        nabla = diff_op(identity_transformation, transformation=True)
        log_det_J = torch.log(calc_det_J(nabla) + 1e-5)

        # test that the values are correct
        assert torch.allclose(log_det_J, torch.zeros_like(log_det_J), atol=atol)

    def test_det_J(self):
        # initialise a transformation
        nabla_x = torch.zeros(dims_v_small, device=device)
        nabla_y, nabla_z = torch.zeros_like(nabla_x), torch.zeros_like(nabla_x)

        for idx_z in range(nabla_x.shape[2]):
            for idx_y in range(nabla_x.shape[3]):
                for idx_x in range(nabla_x.shape[4]):
                    dvx_dx, dvx_dy, dvx_dz = idx_x, idx_z ** 2, idx_y
                    dvy_dx, dvy_dy, dvy_dz = idx_y, idx_x ** 2, idx_z
                    dvz_dx, dvz_dy, dvz_dz = idx_y ** 2, idx_x, idx_x

                    nabla_x[0, :, idx_z, idx_y, idx_x] = torch.tensor([dvx_dx, dvx_dy, dvx_dz], device=device)
                    nabla_y[0, :, idx_z, idx_y, idx_x] = torch.tensor([dvy_dx, dvy_dy, dvy_dz], device=device)
                    nabla_z[0, :, idx_z, idx_y, idx_x] = torch.tensor([dvz_dx, dvz_dy, dvz_dz], device=device)

        # calculate its Jacobian
        nabla = torch.stack([nabla_x, nabla_y, nabla_z], dim=-1)
        det_J = calc_det_J(nabla)

        det_J_true = torch.zeros((1, 1, *dims_small), device=device)

        for idx_z in range(nabla_x.shape[2]):
            for idx_y in range(nabla_x.shape[3]):
                for idx_x in range(nabla_x.shape[4]):
                    # x^4 - x^2 * y^3 - x^2 * z + x * y^2 - x * y * z^2 + y^2 * z^3
                    det_J_true[..., idx_z, idx_y, idx_x] = idx_x ** 4 - idx_x ** 2 * idx_y ** 3 - idx_x ** 2 * idx_z \
                                                           + idx_x * idx_y ** 2 - idx_x * idx_y * idx_z ** 2 \
                                                           + idx_y ** 2 * idx_z ** 3

        # test that the values are correct
        assert torch.allclose(det_J, det_J_true, atol=atol)

    def test_log_det_J_stretching(self):
        # initialise a transformation
        identity_transformation = identity_grid.permute([0, 4, 1, 2, 3])
        displacement = torch.zeros_like(identity_transformation)

        for idx_z in range(displacement.shape[2]):
            for idx_y in range(displacement.shape[3]):
                for idx_x in range(displacement.shape[4]):
                    d = torch.tensor(
                        [2.0 * idx_x / (dim_x - 1.0), 2.0 * idx_y / (dim_y - 1.0), 2.0 * idx_z / (dim_z - 1.0)],
                        device=device)
                    displacement[0, :, idx_z, idx_y, idx_x] = d

        transformation = identity_transformation + displacement

        # calculate its Jacobian
        nabla = diff_op(transformation, transformation=True)

        log_det_J = torch.log(calc_det_J(nabla))
        log_det_J_true = torch.log(8.0 * torch.ones_like(log_det_J))

        # test that the values are correct
        assert torch.allclose(log_det_J, log_det_J_true, atol=atol)
