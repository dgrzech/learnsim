from utils import transform_coordinates

import torch


def add_noise(v, sigma_scaled, u_v_scaled):
    eps = torch.randn(sigma_scaled.size(), device=sigma_scaled.device)
    x = torch.randn(1, device=u_v_scaled.device)

    return v + eps * sigma_scaled + x * u_v_scaled


def add_noise_uniform(field, log_var, alpha=0.25):
    sigma = transform_coordinates(torch.exp(0.5 * log_var))
    return field + alpha * sigma


def sample_q_v(mu_v, log_var_v, u_v, no_samples=1):
    sigma = transform_coordinates(torch.exp(0.5 * log_var_v))
    u = transform_coordinates(u_v)

    eps = torch.randn(sigma.size(), device=sigma.device)
    x = torch.randn(1, device=u_v.device)

    if no_samples == 1:
        return mu_v + eps * sigma + x * u

    return mu_v + (eps * sigma + x * u), mu_v - (eps * sigma + x * u)
