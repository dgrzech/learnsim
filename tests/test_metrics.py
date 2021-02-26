import json
import shutil
import unittest
from datetime import datetime

import numpy as np
import pytest
import torch

from data_loader import BiobankDataset
from parse_config import ConfigParser
from utils import calc_metrics

# fix random seeds for reproducibility
SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


test_config_str = '{' \
                  '"name": "test", "no_GPUs": 0, "optimize_q_phi": false, "optimize_q_v": false,' \
                  '"data_dir": "/vol/bitbucket/dig15/datasets/mine/biobank/biobank_02", "dims": [64, 64, 64], ' \
                  '"trainer": {"save_dir": "./temp"}' \
                  '}'


class MetricsTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')
        self.atol = 1e-5
        self.device = 'cuda:0'
        self.rank = 0

    def tearDown(self):
        save_path = './temp/test'

        try:
            shutil.rmtree(save_path, ignore_errors=True)
        except:
            pass

    def test_DSC(self):
        test_config_json = json.loads(test_config_str)
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

        cfg = ConfigParser(test_config_json, self.rank, timestamp=timestamp)
        structures_dict = cfg.structures_dict

        im_paths = cfg['data_dir']
        save_paths = cfg.save_dirs
        dims = (64, 64, 64)

        dataset = BiobankDataset(dims, im_paths, save_paths, rescale_im=False, rank=0)
        spacing = dataset.im_spacing

        fixed = dataset.fixed
        _, moving, var_params_q_v = dataset[0]
        im_pair_idxs = [0]

        for key in fixed:
            fixed[key] = fixed[key].to(self.device, non_blocking=True)
        for key in moving:
            moving[key] = moving[key].to(self.device, non_blocking=True)
        for key in var_params_q_v:
            var_params_q_v[key] = var_params_q_v[key].to(self.device, non_blocking=True)

        ASD, DSC_CPU = calc_metrics(im_pair_idxs, fixed['seg'], moving['seg'], structures_dict, spacing, GPU=False)
        ASD, DSC_GPU = calc_metrics(im_pair_idxs, fixed['seg'], moving['seg'], structures_dict, spacing, GPU=True)

        for im_pair_idx in im_pair_idxs:
            DSC_CPU, DSC_GPU = DSC_CPU[im_pair_idx], DSC_GPU[im_pair_idx]

            for structure_idx, structure in enumerate(structures_dict):
                val_CPU, val_GPU = DSC_CPU[structure_idx].cpu(), DSC_GPU[structure_idx].cpu()
                assert pytest.approx(val_CPU, self.atol) == val_GPU
