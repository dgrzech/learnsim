import numpy as np
import torch
import unittest


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


class RegistrationTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName)
        pass

    def test_translation(self):
        pass
