from model.loss import EntropyMultivariateNormal

import math
import numpy as np
import pytest
import torch


def test_entropy():
    entropy = EntropyMultivariateNormal()

    n = 2  # no. of voxels in each dimension

    # initialise sigma_v
    log_var_v = torch.log(torch.abs(torch.randn(n, n, n)))
    sigma_v = torch.exp(0.5 * log_var_v) + 1e-5
    var_v = sigma_v ** 2

    # initialise the first mode of variation
    u_v = torch.zeros((n, n, n))

    # calculate the entropy
    val = entropy.forward(log_var_v, u_v).item()
    val_true = 0.5 * math.log(np.linalg.det(np.diag(var_v.data.numpy().flatten())))

    print(val, val_true)
    assert pytest.approx(val, 0.01) == val_true


def test_utils():
    test_entropy()


if __name__ == '__main__':
    test_utils()
