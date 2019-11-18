import torch


class Sampler:
    def __init__(self):
        pass

    @staticmethod
    def sample_qv(mu_v, log_var_v, u_v):
        sigma = torch.exp(0.5 * log_var_v) + 1e-5
        eps = torch.randn_like(sigma)
        x = torch.randn_like(u_v)

        return mu_v + sigma * eps + x * u_v

    @staticmethod
    def sample_qf(mu_f, log_var_f, u_f):
        sigma = torch.exp(0.5 * log_var_f) + 1e-5
        eps = torch.randn_like(sigma)
        x = torch.randn_like(u_f)

        return mu_f + sigma * eps + x * u_f
