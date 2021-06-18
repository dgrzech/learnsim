import numpy as np
import scipy.sparse as sp
import torch

from utils import add_noise_Langevin, separable_conv_3D


def Laplacian_1D(N):
    """
    calculate the Laplacian matrix of size N
    """

    diag = np.ones(N)
    return sp.spdiags([diag, -2.0 * diag, diag], [-1, 0, 1], N, N)


def Laplacian_3D(N):
    L_1D = Laplacian_1D(N)
    I = sp.eye(N)

    return sp.kron(L_1D, sp.kron(I, I)) + sp.kron(I, sp.kron(L_1D, I)) + sp.kron(I, sp.kron(I, L_1D))


def Sobolev_kernel_1D(_s, _lambda):
    """
    approximate the Sobolev kernel

    :param _s: half the kernel width to use
    :param _lambda: smoothing parameter
    :return:
    """
    
    _kernel_sz = 2 * _s + 1

    # we do the eigendecomposition anyway for the sqrt, so might as well compute the smoothing kernel while at it
    L = np.asarray(Laplacian_1D(_kernel_sz).todense())
    w, v = np.linalg.eigh(L)
    w = 1.0 - _lambda * w

    mask = np.abs(w) > 1e-10
    inv_sqrt_w = np.zeros(_kernel_sz)
    inv_sqrt_w[mask] = 1.0 / np.sqrt(w[mask])

    half = v * inv_sqrt_w

    smoothing_kernel = half.dot(half[_s])  # not very pretty but I only need the middle column of half half^t
    smoothing_kernel_sqrt = half.dot(v[_s])  # not very pretty because it breaks the symmetry

    return smoothing_kernel / np.sum(smoothing_kernel), smoothing_kernel_sqrt / np.sum(smoothing_kernel_sqrt)


def Sobolev_kernel_3D(_s, _lambda):
    I = np.eye(_s ** 3)  # identity matrix
    L = Laplacian_3D(_s).todense()  # Laplacian matrix discretised via a 7-point finite-difference stencil

    v = np.zeros(_s ** 3)  # one-hot vector that corresponds to a discretised Dirac impulse of size s ** 3 voxels
    v[(_s ** 3) // 2] = 1.0

    S = np.linalg.solve(I - _lambda * L, v)
    S /= np.sum(S)

    return S.reshape((_s, _s, _s))  # 3D Sobolev filter


def Gaussian_kernel_3D(_s, sigma=1.0):
    """
    approximate the Gaussian kernel
    """

    x, y, z = np.mgrid[-_s // 2 + 1:_s // 2 + 1, -_s // 2 + 1:_s // 2 + 1, -_s // 2 + 1:_s // 2 + 1]
    g = np.exp(-1.0 * (x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2))

    return g / g.sum()


class SGLD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, curr_state, sigma, tau):
        ctx.sigma = sigma
        return add_noise_Langevin(curr_state, sigma, tau)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.sigma ** 2 * grad_output, None, None


class GaussianGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _log_lambda, _gaussian_kernel_hat):
        _log_lambda_hat = torch.rfft(_log_lambda, 3, normalized=False, onesided=False)
        return torch.irfft(_gaussian_kernel_hat * _log_lambda_hat, 3, normalized=False, onesided=False)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class SobolevGrad(torch.autograd.Function):
    """
    autograd function for Sobolev gradients
    """

    @staticmethod
    def forward(ctx, input, S, padding):
        return separable_conv_3D(input, S['x'], S['y'], S['z'], padding)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
