import math
import torch


def loss_function(im, v=None, sigma_voxel_v=None, u_v=None, diff_op=None):
    if im is None:
        return -1.0 * KL(v, sigma_voxel_v, u_v, diff_op)

    return SSD(im)


def SSD(im):
    return -0.5 * torch.sum(im ** 2)


def KL(v, sigma_voxel_v, u_v, diff_op):
    du_v_dx, du_v_dy, du_v_dz = diff_op.apply(u_v)
    dv_dx, dv_dy, dv_dz = diff_op.apply(v)

    t1 = 36.0 * torch.sum(sigma_voxel_v) + torch.sum(du_v_dx ** 2) + torch.sum(du_v_dy ** 2) + torch.sum(du_v_dz ** 2)
    t2 = -1.0 * (torch.sum(dv_dx ** 2) + torch.sum(dv_dy ** 2) + torch.sum(dv_dz) ** 2)
    t3 = -1.0 * (torch.log(1.0 + torch.sum(u_v * 1.0 / sigma_voxel_v * u_v)) + torch.sum(torch.log(sigma_voxel_v)))

    return 0.5 * (t1 + t2 + t3)
