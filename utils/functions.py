from utils import separable_conv_3d

import numpy as np
import scipy.sparse as sp
import torch


def laplacian_1D(N):
    diag = np.ones(N)
    return sp.spdiags([diag, -2.0 * diag, diag], [-1, 0, 1], N, N)


def laplacian_3D(N):
    L_1D = laplacian_1D(N)
    I = sp.eye(N)

    return sp.kron(L_1D, sp.kron(I, I)) + sp.kron(I, sp.kron(L_1D, I)) + sp.kron(I, sp.kron(I, L_1D))


def sobolev_kernel_1D(_s, _lambda):
    # we do the eigendecomposition anyway for the sqrt, so might as well compute the smoothing kernel while at it
    L = np.asarray(laplacian_1D(_s).todense())
    w, v = np.linalg.eigh(L)
    w = 1.0 - _lambda * w

    mask = np.abs(w) > 1e-10
    inv_sqrt_w = np.zeros(_s)
    inv_sqrt_w[mask] = 1 / np.sqrt(w[mask])

    half = v * inv_sqrt_w

    smoothing_kernel = half.dot(half[_s // 2])  # not very pretty but I only need the middle column of half half^t
    smoothing_kernel_sqrt = half.dot(v[_s // 2])  # not very pretty because it breaks the symmetry

    return smoothing_kernel / np.sum(smoothing_kernel), smoothing_kernel_sqrt / np.sum(smoothing_kernel_sqrt)


def sobolev_kernel_3D(_s, _lambda):
    I = np.eye(_s ** 3)  # identity matrix
    L = laplacian_3D(_s).todense()  # Laplacian matrix discretised via a 7-point finite-difference stencil

    v = np.zeros(_s ** 3)  # one-hot vector that corresponds to a discretised Dirac impulse of size s ** 3 voxels
    v[(_s ** 3) // 2] = 1.0

    S = np.linalg.solve(I - _lambda * L, v)
    S /= np.sum(S)

    return S.reshape((_s, _s, _s))  # 3D Sobolev filter


class SobolevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, S_x, S_y, S_z, padding):
        ctx.save_for_backward(input, S_x, S_y, S_z)
        ctx.padding = padding

        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, S_x, S_y, S_z = ctx.saved_tensors
        grad_input = separable_conv_3d(grad_output, S_x, S_y, S_z, ctx.padding)

        return grad_input, None, None, None, None
