from utils import separable_conv_3d

import torch


def sample_qv(mu_v, log_var_v, u_v, *args, **kwargs):
    sigma = torch.exp(0.5 * log_var_v) + 1e-5
    eps = torch.randn_like(sigma)
    x = torch.randn(1, device=u_v.device)

    no_samples = kwargs.get('no_samples', 1)

    if no_samples == 1:
        sample = mu_v + eps * sigma + x * u_v

        if len(args) == 0:
            return sample

        return separable_conv_3d(sample, *args, **kwargs)

    elif no_samples == 2:
        sample1, sample2 = mu_v + (eps * sigma + x * u_v), mu_v - (eps * sigma + x * u_v)

        if len(args) == 0:
            return sample1, sample2

        return separable_conv_3d(sample1, *args, **kwargs), separable_conv_3d(sample2, *args, **kwargs)

    raise NotImplementedError


def sample_qf(mu_f, log_var_f, u_f, no_samples=1):
    sigma = torch.exp(0.5 * log_var_f) + 1e-5
    eps = torch.randn_like(sigma)
    x = torch.randn(1, device=u_f.device)

    if no_samples == 1:
        return mu_f + eps * sigma + x * u_f

    return mu_f + (eps * sigma + x * u_f), mu_f - (eps * sigma + x * u_f)
