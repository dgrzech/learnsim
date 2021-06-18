import math
import numpy as np
import torch
import unittest

from model.loss import LCC, SSD
from utils import get_control_grid_size, init_identity_grid_3D, GradientOperator, RegistrationModule

# fix random seeds for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# parameters
device = 'cuda:0'
rank = 0

N_small, N_large = 4, 64

dim_x_small = dim_y_small = dim_z_small = N_small
dims_small = (N_small,) * 3
dims_v_small = (1, 3, *dims_small)

dims_2D = (N_large, ) * 2

dim_x = dim_y = dim_z = N_large
dims = (N_large, ) * 3
dims_v = (1, 3, *dims)

cps = (4, ) * len(dims)  # for use with B-splines
control_grid_sz = get_control_grid_size(dims, cps)

# transformations
identity_grid = init_identity_grid_3D(dims).to(device)
identity_transformation = identity_grid.permute([0, 4, 1, 2, 3])

# losses
loss_SSD = SSD().to(device)
loss_LCC = LCC().to(device)

# utils
diff_op = GradientOperator()
registration_module = RegistrationModule().to(device)

# error tolerances
atol = 1e-4
rtol = 1e-2
