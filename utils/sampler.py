from utils import transform_coordinates

import torch


def sample_qv(mu_v, log_var_v, u_v, no_samples=1):
    sigma = transform_coordinates(torch.exp(0.5 * log_var_v))
    u = transform_coordinates(u_v)

    eps = torch.randn(sigma.size(), device=sigma.device)
    x = torch.randn(1, device=u_v.device)

    if no_samples == 1:
        return mu_v + eps * sigma + x * u
    elif no_samples == 2:
        return mu_v + (eps * sigma + x * u), mu_v - (eps * sigma + x * u)

    raise NotImplementedError


def sample_qf(mu_f, log_var_f, u_f, no_samples=1):
    sigma = torch.exp(0.5 * log_var_f)
    eps = torch.randn(sigma.size(), device=sigma.device)
    x = torch.randn(1, device=u_f.device)

    if no_samples == 1:
        return mu_f + eps * sigma + x * u_f
    elif no_samples == 2:
        return mu_f + (eps * sigma + x * u_f), mu_f - (eps * sigma + x * u_f)

    raise NotImplementedError
