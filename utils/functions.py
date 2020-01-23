import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def laplacian_3D(N):
    diag = np.ones([N * N])
    mat = sp.spdiags([diag, -2.0 * diag, diag], [-1, 0, 1], N, N)
    I = sp.eye(N)

    L = sp.kron(mat, sp.kron(I, I)) + sp.kron(I, sp.kron(mat, I)) + sp.kron(I, sp.kron(I, mat))
    return L.todense()


def sobolev_kernel(_s, _lambda):
    I = np.eye(_s ** 3)  # identity matrix
    L = laplacian_3D(_s)  # Laplacian matrix discretised via a 7-point finite-difference stencil

    v = np.zeros(_s ** 3)  # one-hot vector that corresponds to a discretised Dirac impulse of size s ** 3 voxels
    v[(_s ** 3) // 2] = 1.0

    S = np.linalg.solve(I - _lambda * L, v)
    S /= np.sum(S)

    return S.reshape((_s, _s, _s))  # 3D Sobolev filter


class SobolevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, S, padding):
        ctx.save_for_backward(input, S)
        ctx.padding = padding

        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, S = ctx.saved_variables
        padding = (ctx.padding, ctx.padding, ctx.padding, ctx.padding, ctx.padding, ctx.padding)

        grad_output = F.pad(grad_output, padding, 'replicate')
        grad_input = F.conv3d(grad_output, S)

        return grad_input, None, None
