import torch


def sample_q_v(mu_v, log_var_v, u_v, no_samples=1):
    """
    sample from the posterior distribution using the reparameterisation trick
    """

    sigma_v = torch.exp(0.5 * log_var_v)

    eps = torch.randn(sigma_v.shape, device=sigma_v.device)
    x = torch.randn(1, device=u_v.device)

    if no_samples == 1:
        return mu_v + eps * sigma_v + x * u_v

    return mu_v + (eps * sigma_v + x * u_v), mu_v - (eps * sigma_v + x * u_v)
