import torch


def sample_q_v(var_params_q_v, no_samples=1):
    """
    sample from the posterior distribution using the reparameterisation trick
    """

    mu_v = var_params_q_v['mu']
    log_var_v = var_params_q_v['log_var']
    u_v = var_params_q_v['u']

    sigma_v = torch.exp(0.5 * log_var_v)
    eps = torch.randn_like(sigma_v)
    x = torch.randn(1, device=u_v.device)

    if no_samples == 1:
        return mu_v + eps * sigma_v + x * u_v

    return mu_v + (eps * sigma_v + x * u_v), mu_v - (eps * sigma_v + x * u_v)
