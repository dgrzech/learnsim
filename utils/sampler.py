import math
import torch
import torch.nn.functional as F


def sample_q(mu_hat, log_var_hat, u_hat, omega_norm_inv=None, no_samples=1):
    """
    sample from the posterior distribution using the reparameterisation trick
    """

    sigma_hat = torch.exp(0.5 * log_var_hat)

    eps = torch.randn(sigma_hat.shape, device=sigma_hat.device)
    padding = (0, 0, 1, 0, 1, 0, 1, 0)

    if no_samples == 1:
        sample = mu_hat.clone()  # + eps * sigma_hat
        # so that the LR for higher frequencies is lower
        return F.pad(sample, padding, 'constant', 0.0) * omega_norm_inv

    # sample1, sample2 = mu_hat + eps * sigma_hat, mu_hat - eps * sigma_hat
    sample1, sample2 = mu_hat.clone(), mu_hat.clone()
    return F.pad(sample1, padding, 'constant', 0.0) * omega_norm_inv, \
           F.pad(sample2, padding, 'constant', 0.0) * omega_norm_inv
