import torch


def sample_qv(mu_v, log_var_v, u_v, no_samples=1):
    sigma = torch.exp(0.5 * log_var_v) + 1e-5
    eps = torch.randn_like(sigma)
    x = torch.randn_like(u_v)

    if no_samples == 1:
        return mu_v + eps * sigma + x * u_v

    return mu_v + (eps * sigma + x * u_v), mu_v - (eps * sigma + x * u_v)


def sample_qf(mu_f, log_var_f, u_f, no_samples=1):
    sigma = torch.exp(0.5 * log_var_f) + 1e-5
    eps = torch.randn_like(sigma)
    x = torch.randn_like(u_f)

    if no_samples == 1:
        return mu_f + eps * sigma + x * u_f

    return mu_f + (eps * sigma + x * u_f), mu_f - (eps * sigma + x * u_f)
