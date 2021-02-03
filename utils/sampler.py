import torch


def sample_q_f(im_fixed, var_params_q_f, no_samples=1):
    log_var_f = var_params_q_f['log_var']
    u_f = var_params_q_f['u']

    sigma_f = torch.exp(0.5 * log_var_f)
    eps = torch.randn(sigma_f.shape, device=sigma_f.device)
    x = torch.randn(1, device=u_f.device)

    if no_samples == 1:
        return im_fixed + eps * sigma_f + x * u_f
    elif no_samples == 2:
        return im_fixed + (eps * sigma_f + x * u_f), im_fixed - (eps * sigma_f + x * u_f)

    raise NotImplementedError


def sample_q_v(var_params_q_v, no_samples=1):
    """
    sample from the posterior distribution using the reparameterisation trick
    """

    mu_v = var_params_q_v['mu']
    log_var_v = var_params_q_v['log_var']
    u_v = var_params_q_v['u']

    sigma_v = torch.exp(0.5 * log_var_v)
    eps = torch.randn(sigma_v.shape, device=sigma_v.device)
    x = torch.randn(1, device=u_v.device)

    if no_samples == 1:
        return mu_v + eps * sigma_v + x * u_v

    return mu_v + (eps * sigma_v + x * u_v), mu_v - (eps * sigma_v + x * u_v)
