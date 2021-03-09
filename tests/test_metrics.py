from data_loader import BiobankDataset
from datetime import datetime
from parse_config import ConfigParser
from .test_setup import *
from utils import calc_metrics

import json
import pytest
import shutil


test_config_str = '{' \
                  '"name": "test", "no_GPUs": 0,' \
                  '"data_dir": "/vol/bitbucket/dig15/datasets/mine/biobank/biobank_02", "dims": [64, 64, 64], ' \
                  '"trainer": {"save_dir": "./temp", "sigma_v_init": 0.5, "u_v_init": 0.0}' \
                  '}'


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
        structures_dict = cfg.structures_dict

        im_paths = cfg['data_dir']
        save_paths = cfg.save_dirs
        sigma_v_init, u_v_init = cfg['trainer']['sigma_v_init'], cfg['trainer']['u_v_init']

        dataset = BiobankDataset(dims, im_paths, save_paths, sigma_v_init, u_v_init, rescale_im=False, rank=0)
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
