import logging
import os
from datetime import datetime
from functools import partial, reduce
from operator import getitem
from pathlib import Path
from shutil import copy, copytree
import torch.distributed as dist

import data_loader.data_loaders as module_data
import model.loss as model_loss
import utils.registration as registration
import utils.transformation as transformation
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, local_rank, resume=None, modification=None, run_id=None, test=False):
        """
        class to parse configuration json file
        handles hyperparameters for training, initializations of modules, checkpoint saving and logging module

        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)

        self.rank = local_rank
        self.resume = resume
        self.test = test

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']

        if run_id is None:  # use timestamp as default run-id
            timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

            if config['optimize_q_phi']:
                run_id = 'learnt_' + timestamp
            else:
                run_id = 'baseline_' + timestamp

            if self.test:
                run_id = 'test_' + run_id

        dir = save_dir / exper_name / run_id

        self._dir = dir
        self._save_dir = dir / 'checkpoints'
        self._optimizers_dir = dir / 'optimizers'
        self._tensors_dir = dir / 'tensors'
        self._samples_dir = dir / 'samples'

        self._log_dir = dir / 'log'

        self._im_dir = dir / 'images'
        self._fields_dir = dir / 'fields'
        self._grids_dir = dir / 'grids'
        self._norms_dir = dir / 'norms'

        # make directory for saving checkpoints and log.
        if local_rank == 0:
            exist_ok = run_id == ''

            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.optimizers_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.tensors_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.samples_dir.mkdir(parents=True, exist_ok=exist_ok)

            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

            self.im_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.fields_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.grids_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.norms_dir.mkdir(parents=True, exist_ok=exist_ok)

            # copy values of variational parameters
            if self.resume is not None and not self.test:
                print('copying previous values of variational parameters..')
                copytree(self.resume.parent.parent / 'optimizers', self.optimizers_dir)
                copytree(self.resume.parent.parent / 'tensors', self.tensors_dir)
                print('done!\n')

            # save updated config file to the checkpoint dir
            write_json(self.config, dir / 'config.json')

        # configure logging module
        if local_rank == 0:
            setup_logging(self.log_dir)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        # segmentation IDs
        self.structures_dict = {'left_thalamus': 10, 'left_caudate': 11, 'left_putamen': 12,
                                'left_pallidum': 13, 'brain_stem': 16, 'left_hippocampus': 17,
                                'left_amygdala': 18, 'left_accumbens': 26, 'right_thalamus': 49,
                                'right_caudate': 50, 'right_putamen': 51, 'right_pallidum': 52,
                                'right_hippocampus': 53, 'right_amygdala': 54, 'right_accumbens': 58}

    @classmethod
    def from_args(cls, args, options='', test=False):
        """
        initialize this class from some cli arguments; used in train, test
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        local_rank = args.local_rank
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}

        return cls(config, local_rank, resume=resume, modification=modification, test=test)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """

        module_name = self[name]['type']

        if 'args' in dict(self[name]):
            module_args = dict(self[name]['args'])
            assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
            module_args.update(kwargs)
        else:
            module_args = dict()

        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """

        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def init_data_loader(self):
        return self.init_obj('data_loader', module_data, no_GPUs=self['no_GPUs'], rank=self.rank, save_dirs=self.save_dirs)

    def init_losses(self):
        data_loss = self.init_obj('data_loss', model_loss)
        reg_loss = self.init_obj('reg_loss', model_loss)
        entropy_loss = self.init_obj('entropy_loss', model_loss)

        return {'data': data_loss, 'regularisation': reg_loss, 'entropy': entropy_loss}

    def init_metrics(self, no_samples):
        loss_terms = ['loss/data_term', 'loss/reg_term', 'loss/entropy_term', 'loss/q_v', 'loss/q_f_q_phi']

        ASD = ['ASD/im_pair_' + str(im_pair_idx) + '/' + structure for structure in self.structures_dict for im_pair_idx in range(no_samples)]
        DSC = ['DSC/im_pair_' + str(im_pair_idx) + '/' + structure for structure in self.structures_dict for im_pair_idx in range(no_samples)]
        no_non_diffeomorphic_voxels = ['no_non_diffeomorphic_voxels/im_pair_' + str(im_pair_idx) for im_pair_idx in range(no_samples)]

        if self.test:
            ASD.extend(['test/ASD/im_pair_' + str(im_pair_idx) + '/' + structure for structure in self.structures_dict for im_pair_idx in range(no_samples)])
            DSC.extend(['test/DSC/im_pair_' + str(im_pair_idx) + '/' + structure for structure in self.structures_dict for im_pair_idx in range(no_samples)])
            no_non_diffeomorphic_voxels.extend(['test/no_non_diffeomorphic_voxels/im_pair_' + str(im_pair_idx) for im_pair_idx in range(no_samples)])

        return loss_terms + ASD + DSC + no_non_diffeomorphic_voxels

    def init_transformation_and_registration_modules(self, dims):
        return self.init_obj('transformation_module', transformation, dims), self.init_obj('registration_module', registration)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name):
        verbosity = self['trainer']['verbosity']
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity

        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def copy_var_params_to_backup_dirs(self, epoch):
        dist.barrier()

        if self.rank == 0:
            epoch_str = 'epoch_' + str(epoch).zfill(4)

            optimizers_backup_path = self.optimizers_dir / epoch_str
            tensors_backup_path = self.tensors_dir / epoch_str

            optimizers_backup_path.mkdir(parents=True, exist_ok=True)
            tensors_backup_path.mkdir(parents=True, exist_ok=True)

            for f in os.listdir(self.optimizers_dir):
                f_path = os.path.join(self.optimizers_dir, f)

                if os.path.isfile(f_path):
                    copy(f_path, optimizers_backup_path)

            for f in os.listdir(self.tensors_dir):
                f_path = os.path.join(self.tensors_dir, f)

                if os.path.isfile(f_path):
                    copy(f_path, tensors_backup_path)

        dist.barrier()

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def dir(self):
        return self._dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def optimizers_dir(self):
        return self._optimizers_dir

    @property
    def tensors_dir(self):
        return self._tensors_dir

    @property
    def samples_dir(self):
        return self._samples_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def im_dir(self):
        return self._im_dir

    @property
    def fields_dir(self):
        return self._fields_dir

    @property
    def grids_dir(self):
        return self._grids_dir

    @property
    def norms_dir(self):
        return self._norms_dir

    @property
    def save_dirs(self):
        return {'dir': self.dir, 'optimizers': self.optimizers_dir, 'tensors': self.tensors_dir, 'samples': self.samples_dir,
                'images': self.im_dir, 'fields': self.fields_dir, 'grids': self.grids_dir, 'norms': self.norms_dir}


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """
    set a value in a nested object in tree by sequence of keys
    """

    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """
    access a nested object in tree by sequence of keys
    """

    return reduce(getitem, keys, tree)
