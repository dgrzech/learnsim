from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

from logger import setup_logging
from utils import read_json, write_json

import os
import logging


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
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
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']

        if run_id is None:  # use timestamp as default run-id
            timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

            if config['optimize_q_phi']:
                run_id = 'learnt_' + timestamp
            else:
                run_id = 'baseline_' + timestamp

        self._save_dir = save_dir / exper_name / run_id / 'checkpoints'
        self._optimizers_dir = save_dir / exper_name / run_id / 'optimizers'
        self._tensors_dir = save_dir / exper_name / run_id / 'tensors'
        self._samples_dir = save_dir / exper_name / run_id / 'samples'

        self._log_dir = save_dir / exper_name / run_id / 'log'

        self._im_dir = save_dir / exper_name / run_id / 'images'
        self._fields_dir = save_dir / exper_name / run_id / 'fields'
        self._grids_dir = save_dir / exper_name / run_id / 'grids'
        self._norms_dir = save_dir / exper_name / run_id / 'norms'

        # make directory for saving checkpoints and log.
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

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        initialize this class from some cli arguments; used in train, test
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

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

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = \
            'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

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
        return {'optimizers': self.optimizers_dir, 'tensors': self.tensors_dir, 'samples': self.samples_dir,
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
