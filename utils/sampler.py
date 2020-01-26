from utils import separable_conv_3d

import torch


def sample_qv(mu_v, log_var_v, u_v, S_sqrt_x=None, S_sqrt_y=None, S_sqrt_z=None, padding=None, no_samples=1):
    sigma = torch.exp(0.5 * log_var_v) + 1e-5
    eps = torch.randn_like(sigma)
    x = torch.randn_like(u_v)

    if no_samples == 1:
        sample = mu_v + eps * sigma + x * u_v

        if S_sqrt_x is None:
            return sample

        return separable_conv_3d(sample, S_sqrt_x, S_sqrt_y, S_sqrt_z, padding)

    sample1, sample2 = mu_v + (eps * sigma + x * u_v), mu_v - (eps * sigma + x * u_v)

    if S_sqrt_x is None:
        return sample1, sample2

    return separable_conv_3d(sample1, S_sqrt_x, S_sqrt,_y, S_sqrt_z, padding), separable_conv_3d(sample2, S_sqrt_x, S_sqrt_y, S_sqrt_z, padding)


def sample_qf(mu_f, log_var_f, u_f, no_samples=1):
    sigma = torch.exp(0.5 * log_var_f) + 1e-5
    eps = torch.randn_like(sigma)
    x = torch.randn_like(u_f)

    if no_samples == 1:
        return mu_f + eps * sigma + x * u_f

    return mu_f + (eps * sigma + x * u_f), mu_f - (eps * sigma + x * u_f)
