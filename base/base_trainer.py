from logger import TensorboardWriter

from abc import abstractmethod

import torch


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, data_loss, scale_prior, proportion_prior, reg_loss, reg_loss_prior_loc, reg_loss_prior_scale,
                 entropy_loss, transformation_model, registration_module, config):
        self.config = config
        self.checkpoint_dir = config.save_dir
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available and move the model and losses into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.transformation_model = transformation_model.to(self.device)
        self.registration_module = registration_module.to(self.device)

        self.data_loss = data_loss.to(self.device)
        self.scale_prior = scale_prior.to(self.device)
        self.proportion_prior = proportion_prior.to(self.device)

        self.reg_loss = reg_loss.to(self.device)
        self.reg_loss_prior_loc = reg_loss_prior_loc.to(self.device)
        self.reg_loss_prior_scale = reg_loss_prior_scale.to(self.device)

        self.entropy_loss = entropy_loss.to(self.device)

        if len(device_ids) > 1:
            self.transformation_model = torch.nn.DataParallel(transformation_model, device_ids=device_ids)
            self.registration_module = torch.nn.DataParallel(registration_module, device_ids=device_ids)

            self.data_loss = torch.nn.DataParallel(data_loss, device_ids=device_ids)
            self.scale_prior = torch.nn.DataParallel(scale_prior, device_ids=device_ids)
            self.proportion_prior = torch.nn.DataParallel(proportion_prior, device_ids=device_ids)

            self.reg_loss = torch.nn.DataParallel(reg_loss, device_ids=device_ids)
            self.reg_loss_prior_loc = torch.nn.DataParallel(reg_loss_prior_loc, device_ids=device_ids)
            self.reg_loss_prior_scale = torch.nn.DataParallel(reg_loss_prior_scale, device_ids=device_ids)

            self.entropy_loss = torch.nn.DataParallel(entropy_loss, device_ids=device_ids)

        # training logic
        cfg_trainer = config['trainer']

        self.no_iters_vi = int(cfg_trainer['no_iters_vi'])
        self.no_iters_burn_in = int(cfg_trainer['no_iters_burn_in'])
        self.no_samples = int(cfg_trainer['no_samples'])

        self.log_period = cfg_trainer['log_period']
        self.save_period = cfg_trainer['save_period']

        self.log_period_mcmc = cfg_trainer['log_period_mcmc']
        self.save_period_mcmc = cfg_trainer['save_period_mcmc']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

    @abstractmethod
    def _train_epoch(self):
        """
        training logic for an epoch
        """

        raise NotImplementedError

    def train(self):
        """
        full training logic
        """

        self._train_epoch()

    def _prepare_device(self, n_gpu_use):
        """
        set up GPU device if available and move model into configured device
        """

        n_gpu = torch.cuda.device_count()

        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0

        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))

        return device, list_ids
