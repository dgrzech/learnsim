from utils import transform_coordinates
import torch


def sample_q_v(mu_v, log_var_v, u_v, no_samples=1):
    """
    sample from the posterior distribution using the reparameterisation trick
    """

    sigma = transform_coordinates(torch.exp(0.5 * log_var_v))
    u = transform_coordinates(u_v)

    eps = torch.randn(sigma.shape, device=sigma.device)
    x = torch.randn(1, device=u_v.device)

    if no_samples == 1:
        return mu_v + eps * sigma + x * u

    return mu_v + (eps * sigma + x * u), mu_v - (eps * sigma + x * u)
