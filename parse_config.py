import json
import logging
import os
import socket
from functools import reduce
from operator import getitem
from pathlib import Path
from shutil import copy, copytree, rmtree

from torch import nn

import data_loader.data_loaders as module_data
import model.loss as model_loss
import model.model as model
import utils.registration as registration
import utils.transformation as transformation
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, local_rank, modification=None, resume=None, test=False, timestamp=None):
        self._config = _update_config(config, modification)
        self.rank = local_rank
        self.resume = resume
        self.test = test

        # logger
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

        verbosity = self['trainer']['verbosity']
        msg_verbosity = f'verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}.'
        assert verbosity in self.log_levels, msg_verbosity

        self._logger = logging.getLogger('default')
        self._logger.setLevel(self.log_levels[verbosity])

        # set save_dir where the trained model and log will be saved
        run_id = timestamp

        if self.test:
            if resume is None:
                run_id = f'test_baseline_{run_id}'
            else:
                run_id = f'test_learnt_{run_id}'

        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']
        dir = save_dir / exper_name / run_id

        self._dir = dir
        self._log_dir = dir / 'log'
        self._save_dir = dir / 'checkpoints'
        self._tensors_dir = dir / 'tensors'
        self._samples_dir = dir / 'samples'

        self._im_dir = dir / 'images'
        self._fields_dir = dir / 'fields'
        self._grids_dir = dir / 'grids'
        self._norms_dir = dir / 'norms'

        # make directories for saving checkpoints and logs
        if self.rank == 0:
            exist_ok = run_id == ''

            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.tensors_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.samples_dir.mkdir(parents=True, exist_ok=exist_ok)

            self.im_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.fields_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.grids_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.norms_dir.mkdir(parents=True, exist_ok=exist_ok)

            # copy values of variational parameters
            if self.resume is not None and not self.test:
                self.logger.info('copying previous values of variational parameters..')
                copytree(self.resume.parent.parent / 'tensors', self.tensors_dir, dirs_exist_ok=True)
                self.logger.info('done!')

            # configure logging
            setup_logging(self.log_dir)

            # save updated config file to the checkpoint dir
            self.config_str = json.dumps(self.config, indent=4, sort_keys=False).replace('\n', '')
            write_json(self.config, dir / 'config.json')

    @classmethod
    def from_args(cls, args, options='', timestamp=None, test=False):
        # initialize this class from cli arguments
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        local_rank = args.local_rank

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent.parent / 'config.json'
        else:
            assert args.config is not None, "config file needs to be specified; add '-c config.json'"
            cfg_fname, resume = Path(args.config), None

        config = read_json(cfg_fname)
        if args.config and resume:
            config.update(read_json(args.config))

        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, local_rank, modification=modification, resume=resume, test=test, timestamp=timestamp)

    def init_data_loader(self):
        self['data_loader']['args']['save_dirs'] = self.save_dirs
        self['data_loader']['args']['no_GPUs'] = self['no_GPUs']
        self['data_loader']['args']['rank'] = self.rank

        try:
            self['data_loader']['args']['cps'] = self['transformation_module']['args']['cps']
        except:
            self['data_loader']['args']['cps'] = None


        data_loader = self.init_obj('data_loader', module_data)
        self.structures_dict = data_loader.structures_dict

        return data_loader

    def init_model(self):
        try:
            args = self['model']['args']
            self['model']['args']['activation'] = getattr(nn, args['activation']['type'])(**dict(args['activation']['args']))
        except:
            self['model']['args']['activation'] = nn.Identity()

        return self.init_obj('model', model)

    def init_losses(self):
        return {'data': self.init_obj('data_loss', model_loss), 'regularisation': self.init_obj('reg_loss', model_loss),
                'entropy': self.init_obj('entropy_loss', model_loss)}

    def init_metrics(self, no_samples):
        loss_terms = ['loss/data_term', 'loss/reg_term', 'loss/entropy_term', 'loss/q_v', 'loss/q_phi']

        ASD = [f'ASD/im_pair_{im_pair_idx}/{structure}' for structure in self.structures_dict for im_pair_idx in range(no_samples)]
        ASD.extend([f'ASD/im_pair_{im_pair_idx}/avg' for im_pair_idx in range(no_samples)])
        ASD.extend(['ASD/avg'])
        ASD.extend([f'ASD/avg/{structure}' for structure in self.structures_dict])

        DSC = [f'DSC/im_pair_{im_pair_idx}/{structure}' for structure in self.structures_dict for im_pair_idx in range(no_samples)]
        DSC.extend([f'DSC/im_pair_{im_pair_idx}/avg' for im_pair_idx in range(no_samples)])
        DSC.extend(['DSC/avg'])
        DSC.extend([f'DSC/avg/{structure}' for structure in self.structures_dict])
        
        no_non_diffeomorphic_voxels = [f'no_non_diffeomorphic_voxels/im_pair_{im_pair_idx}' for im_pair_idx in range(no_samples)]
        no_non_diffeomorphic_voxels.extend(['no_non_diffeomorphic_voxels/avg'])

        if self.test:
            ASD.extend([f'test/ASD/im_pair_{im_pair_idx}/{structure}' for structure in self.structures_dict for im_pair_idx in range(no_samples)])
            DSC.extend([f'test/DSC/im_pair_{im_pair_idx}/{structure}' for structure in self.structures_dict for im_pair_idx in range(no_samples)])
            no_non_diffeomorphic_voxels.extend([f'test/no_non_diffeomorphic_voxels/im_pair_{im_pair_idx}' for im_pair_idx in range(no_samples)])

        return loss_terms + ASD + DSC + no_non_diffeomorphic_voxels

    def init_transformation_and_registration_modules(self):
        self['transformation_module']['args']['dims'] = self['data_loader']['args']['dims']

        return self.init_obj('transformation_module', transformation), self.init_obj('registration_module', registration)

    def init_obj(self, name, module, *args, **kwargs):
        """
        find a function handle with the name given as 'type' in config, and return the
        instance initialized with corresponding arguments given;
        `object = config.init_obj('name', module, a, b=1)` is equivalent to `object = module.name(a, b=1)`
        """

        module_name = self[name]['type']

        if 'args' in dict(self[name]):
            module_args = dict(self[name]['args'])
            module_args.update(kwargs)
        else:
            module_args = dict()

        return getattr(module, module_name)(*args, **module_args)

    def copy_var_params_to_backup_dirs(self, epoch):
        epoch_str = 'epoch_' + str(epoch).zfill(4)

        tensors_backup_path = self.tensors_dir / epoch_str
        tensors_backup_path.mkdir(parents=True, exist_ok=True)

        for f in os.listdir(self.tensors_dir):
            f_path = os.path.join(self.tensors_dir, f)

            if os.path.isfile(f_path):
                copy(f_path, tensors_backup_path)

    def copy_var_params_from_backup_dirs(self, resume_epoch):
        var_params_backup_dirs = [f for f in os.listdir(self.tensors_dir)
                                  if not os.path.isfile(os.path.join(self.tensors_dir, f))]

        def find_last_backup_epoch_dirs():
            for epoch in reversed(range(1, resume_epoch + 1)):
                resume_epoch_str = 'epoch_' + str(epoch).zfill(4)

                if resume_epoch_str in var_params_backup_dirs:
                    return resume_epoch_str

            raise ValueError

        last_backup_epoch = find_last_backup_epoch_dirs()

        def copy_backup_to_current_dir():
            var_params_backup_dir = os.path.join(self.tensors_dir, last_backup_epoch)

            for f in os.listdir(var_params_backup_dir):
                f_path = os.path.join(var_params_backup_dir, f)
                copy(f_path, self.tensors_dir)

        copy_backup_to_current_dir()

    def remove_backup_dirs(self):
        var_params_backup_dirs = [f for f in os.listdir(self.tensors_dir)
                                  if not os.path.isfile(os.path.join(self.tensors_dir, f))]

        for f in var_params_backup_dirs:
            f_path = os.path.join(self.tensors_dir, f)
            rmtree(f_path)

    def __getitem__(self, name):
        # access items like in a dict
        return self.config[name]

    # setting read-only attributes
    @property
    def logger(self):
        return self._logger

    @property
    def config(self):
        return self._config

    @property
    def dir(self):
        return self._dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def tensors_dir(self):
        return self._tensors_dir

    @property
    def samples_dir(self):
        return self._samples_dir

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
        return {'dir': self.dir, 'tensors': self.tensors_dir, 'samples': self.samples_dir,
                'images': self.im_dir, 'fields': self.fields_dir, 'grids': self.grids_dir, 'norms': self.norms_dir}


def _update_config(config, modification):
    # helper function to update config dict with custom cli options
    config['hostname'] = socket.gethostname()

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
    # set a value in a nested object in tree by sequence of keys
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    # access a nested object in tree by sequence of keys
    return reduce(getitem, keys, tree)
