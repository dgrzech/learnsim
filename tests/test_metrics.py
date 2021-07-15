import json
import shutil
from datetime import datetime

import pytest

from data_loader import BiobankDataset
from parse_config import ConfigParser
from utils import calc_metrics
from .test_setup import *


class MetricsTestMethods(unittest.TestCase):
    def setUp(self):
        print(self._testMethodName + '\n')

    def tearDown(self):
        save_path = './temp/test'

        try:
            shutil.rmtree(save_path, ignore_errors=True)
        except:
            pass

    def test_DSC(self):
        test_config_json = json.loads(test_config_str)
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

        cfg = ConfigParser(test_config_json, rank, timestamp=timestamp)

        save_paths, im_pairs = cfg.save_dirs, cfg['im_pairs']
        dataset = BiobankDataset(save_paths, im_pairs, dims)
        structures_dict = dataset.structures_dict
        spacing = dataset.im_spacing

        fixed = dataset.fixed
        _, moving, var_params_q_v = dataset[0]
        im_pair_idxs = [0]

        for key in fixed:
            fixed[key] = fixed[key].to(device, non_blocking=True)
        for key in moving:
            moving[key] = moving[key].to(device, non_blocking=True)
        for key in var_params_q_v:
            var_params_q_v[key] = var_params_q_v[key].to(device, non_blocking=True)

        ASD, DSC_CPU = calc_metrics(im_pair_idxs, fixed['seg'], moving['seg'], structures_dict, spacing, GPU=False)
        ASD, DSC_GPU = calc_metrics(im_pair_idxs, fixed['seg'], moving['seg'], structures_dict, spacing, GPU=True)

        for im_pair_idx in im_pair_idxs:
            DSC_CPU, DSC_GPU = DSC_CPU[im_pair_idx], DSC_GPU[im_pair_idx]

            for structure_idx, structure in enumerate(structures_dict):
                val_CPU, val_GPU = DSC_CPU[structure_idx].cpu(), DSC_GPU[structure_idx].cpu()
                assert pytest.approx(val_CPU, atol) == val_GPU
