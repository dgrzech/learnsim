from utils import Cubic_B_spline_FFD_3D, SVF_2D, SVF_3D, SVFFD_3D, \
    pixel_to_normalised_2D, pixel_to_normalised_3D, plot_2D, plot_3D, separable_conv_3D
from .test_setup import *


class UtilsTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

    # def test_scaling_and_squaring_2D_translation(self):
    #     transformation_module = SVF_2D(dims_2D)

    #     v = 0.2 * torch.ones(1, 2, *dims_2D)
    #     transformation, displacement = transformation_module(v)
    #     plot_2D(v, transformation)

    # def test_scaling_and_squaring_3D_translation(self):
    #     transformation_module = SVF_3D(dims)

    #     v = 0.2 * torch.ones(1, 3, *dims)
    #     transformation, displacement = transformation_module(v)
    #     plot_3D(v, transformation)

    # def test_scaling_and_squaring_2D_rotation(self):
    #     transformation_module = SVF_2D(dims_2D)

    #     v = torch.zeros(1, 2, *dims_2D)
    #     for idx_x in range(v.shape[3]):
    #         for idx_y in range(v.shape[2]):
    #             x, y = pixel_to_normalised_2D(idx_x, idx_y, *dims_2D)

    #             v[0, 0, idx_x, idx_y] = y
    #             v[0, 1, idx_x, idx_y] = -1.0 * x

    #     transformation, displacement = transformation_module(v)
    #     plot_2D(v, transformation)

    # def test_scaling_and_squaring_3D_rotation(self):
    #     transformation_module = SVF_3D(dims)

    #     v = torch.zeros(1, 3, dims)
    #     for idx_z in range(v.shape[2]):
    #         for idx_y in range(v.shape[3]):
    #             for idx_x in range(v.shape[4]):
    #                 x, y, z = pixel_to_normalised_3D(idx_x, idx_y, idx_z, *dims)

    #                 v[0, 0, idx_x, idx_y, idx_z] = y
    #                 v[0, 1, idx_x, idx_y, idx_z] = -1.0 * x

    #     transformation, displacement = transformation_module(v)
    #     plot_3D(v, transformation)

    def test_cubic_B_spline_FFD_3D(self):
        v = torch.randn(1, 3, *control_grid_sz)

        transformation_module = Cubic_B_spline_FFD_3D(dims, cps)
        v_out = transformation_module(v)
        v_out_size = v_out.shape

        assert v_out_size[0] == 1
        assert v_out_size[1] == 3
        assert v_out_size[2] == dims[0]
        assert v_out_size[3] == dims[1]
        assert v_out_size[4] == dims[2]

    def test_SVFFD_3D(self):
        v = torch.randn(1, 3, *control_grid_sz)

        transformation_module = SVFFD_3D(dims, cps)
        transformation, displacement = transformation_module(v)
        transformation_size = transformation.shape

        assert transformation_size[0] == 1
        assert transformation_size[1] == 3
        assert transformation_size[2] == dims[0]
        assert transformation_size[3] == dims[1]
        assert transformation_size[4] == dims[2]

    def test_separable_conv_3D(self):
        N = 2  # batch size
        D = H = W = 16  # no. of voxels in each dimension

        _s = 3  # Sobolev kernel size
        S = torch.ones((1, _s), device=device)
        S = torch.stack((S, S, S), 0)
        S_x, S_y, S_z = S.unsqueeze(2).unsqueeze(2), S.unsqueeze(2).unsqueeze(4), S.unsqueeze(3).unsqueeze(4)

        padding_sz = _s // 2
        padding = (padding_sz, ) * 6

        # velocity fields
        v = torch.zeros([N, 3, D, H, W], device=device)
        v[0, 1], v[1, 2] = 1.0, 1.0

        # separable convolution implemented as three 1D convolutions
        v_out = separable_conv_3D(v, S, padding_sz)
        v_out_size = v_out.shape

        assert v_out_size[0] == 2
        assert v_out_size[1] == 3
        assert v_out_size[2] == D
        assert v_out_size[3] == H
        assert v_out_size[4] == W

        assert torch.allclose(v_out[0, 0], torch.zeros_like(v_out[0, 0]), atol=atol)
        assert torch.allclose(v_out[0, 1], 27.0 * torch.ones_like(v_out[0, 1]), atol=atol)
        assert torch.allclose(v_out[0, 2], torch.zeros_like(v_out[0, 2]), atol=atol)

        assert torch.allclose(v_out[1, 0], torch.zeros_like(v_out[1, 0]), atol=atol)
        assert torch.allclose(v_out[1, 1], torch.zeros_like(v_out[1, 1]), atol=atol)
        assert torch.allclose(v_out[1, 2], 27.0 * torch.ones_like(v_out[1, 2]), atol=atol)

        # separable convolution implemented as 3D convolutions
        v_out = separable_conv_3D(v, S_x, S_y, S_z, padding)
        v_out_size = v_out.shape

        assert v_out_size[0] == 2
        assert v_out_size[1] == 3
        assert v_out_size[2] == D
        assert v_out_size[3] == H
        assert v_out_size[4] == W

        assert torch.allclose(v_out[0, 0], torch.zeros_like(v_out[0, 0]), atol=atol)
        assert torch.allclose(v_out[0, 1], 27.0 * torch.ones_like(v_out[0, 1]), atol=atol)
        assert torch.allclose(v_out[0, 2], torch.zeros_like(v_out[0, 2]), atol=atol)

        assert torch.allclose(v_out[1, 0], torch.zeros_like(v_out[1, 0]), atol=atol)
        assert torch.allclose(v_out[1, 1], torch.zeros_like(v_out[1, 1]), atol=atol)
        assert torch.allclose(v_out[1, 2], 27.0 * torch.ones_like(v_out[1, 2]), atol=atol)
