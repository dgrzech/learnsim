import logging
import os
import socket
from functools import reduce
from operator import getitem

import data_loader.data_loaders as module_data
import model.loss as model_loss
import model.model as model
import utils.registration as registration
import utils.transformation as transformation
from logger import Logger, setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, local_rank, modification=None, resume=None, test=False, timestamp=None):
        self._config = _update_config(config, modification)
        self.resume, self.test = resume, test
        self.rank = local_rank

        # set the directory where the trained model and log will be saved
        run_id = timestamp

        if self.test:
            run_id = f'test_baseline_{run_id}' if resume is None else f'test_learnt_{run_id}'

        exper_name = self.config['name']
        save_dir = self.config['trainer']['save_dir']

        self._run_dir = os.path.join(save_dir, exper_name, run_id)
        self._log_dir = os.path.join(self.run_dir, 'log')
        self._checkpoints_dir = os.path.join(self.run_dir, 'model', 'checkpoints')
        self._var_params_dir = os.path.join(self.run_dir, 'model', 'var_params')

        self._im_dir = os.path.join(self.run_dir, 'images')
        self._samples_dir = os.path.join(self.run_dir, 'samples')

        # make directories for saving checkpoints and logs
        if self.rank == 0:
            for k, v in self.save_dirs.items():
                os.makedirs(v)

        # logger
        logging.setLoggerClass(Logger)

        self._logger = logging.getLogger('default')
        self._logger.setLevel(logging.DEBUG)

        if self.rank == 0:
            setup_logging(self.log_dir)
            write_json(self.config, os.path.join(self.run_dir, 'config.json'))  # save updated config file to the checkpoint run_dir

    @classmethod
    def from_args(cls, args, options='', timestamp=None, test=False):
        # initialize this class from cli arguments
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        try:
            local_rank = args.local_rank
        except:
            local_rank = 0

        if args.resume is not None:
            resume = os.fspath(args.resume)
            cfg_fname = resume.parent.parent.parent / 'config.json'
        else:
            assert args.config is not None, "config file needs to be specified; add '-c config.json'"
            cfg_fname, resume = os.fspath(args.config), None

        config = read_json(cfg_fname)
        if args.config and resume:
            config.update(read_json(args.config))

        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, local_rank, modification=modification, resume=resume, test=test, timestamp=timestamp)

    def init_data_loader(self):
        cfg_data_loader = self['data_loader']['args']
        cfg_data_loader['save_dirs'] = self.save_dirs

        try:
            cfg_transformation_module = self['transformation_module']['args']
            cfg_data_loader['cps'] = cfg_transformation_module['cps']
        except:
            pass

        data_loader = self.init_obj('data_loader', module_data)
        self.no_samples, self.structures_dict = data_loader.no_samples, data_loader.structures_dict

        return data_loader

    def init_model(self):
        return self.init_obj('model', model)

    def init_losses(self):
        return {'data': self.init_obj('data_loss', model_loss), 'regularisation': self.init_obj('reg_loss', model_loss),
                'entropy': self.init_obj('entropy_loss', model_loss)}

    def init_metrics(self):
        loss_terms = ['loss/data_term', 'loss/regularisation_term', 'loss/negative_entropy_term',
                      'loss/q_v', 'loss/positive_sample_energy', 'loss/negative_sample_energy', 'loss/q_phi']

        ASD = ['ASD/avg'] + [f'ASD/avg/{structure}' for structure in self.structures_dict]
        DSC = ['DSC/avg'] + [f'DSC/avg/{structure}' for structure in self.structures_dict]
        no_non_diffeomorphic_voxels = ['non_diffeomorphic_voxels/avg']

        if self.test:
            ASD.extend([f'test/ASD/im_pair_{im_pair_idx}/{structure}' for structure in self.structures_dict for im_pair_idx in range(self.no_samples)])
            DSC.extend([f'test/DSC/im_pair_{im_pair_idx}/{structure}' for structure in self.structures_dict for im_pair_idx in range(self.no_samples)])
            no_non_diffeomorphic_voxels.extend([f'test/non_diffeomorphic_voxels/im_pair_{im_pair_idx}' for im_pair_idx in range(self.no_samples)])

        return loss_terms + ASD + DSC + no_non_diffeomorphic_voxels

    def init_transformation_and_registration_modules(self):
        cfg_transformation = self['transformation_module']['args']
        cfg_data_loader = self['data_loader']['args']

        cfg_transformation['dims'] = cfg_data_loader['dims']

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
    def run_dir(self):
        return self._run_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    @property
    def var_params_dir(self):
        return self._var_params_dir

    @property
    def samples_dir(self):
        return self._samples_dir

    @property
    def im_dir(self):
        return self._im_dir

    @property
    def save_dirs(self):
        return {'run_dir' : self.run_dir, 'log_dir': self.log_dir,
                'checkpoints_dir': self.checkpoints_dir, 'var_params_dir': self.var_params_dir,
                'images_dir': self.im_dir, 'samples_dir': self.samples_dir}


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

