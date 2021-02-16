from abc import abstractmethod

import torch
from torch import nn

from logger import TensorboardWriter
from model.distributions import LowRankMultivariateNormalDistribution
from utils import get_module_attr


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, config, data_loader, model, losses, transformation_module, registration_module, test):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.test = test

        self.data_loader = data_loader
        self.spacing, self.structures_dict = self.data_loader.spacing, self.config.structures_dict
        self.save_dirs = self.data_loader.save_dirs

        # setup visualization writer instance
        cfg_trainer = config['trainer']

        self.writer = TensorboardWriter(config.log_dir, cfg_trainer['tensorboard'])
        self.writer.write_graph(model)
        self.writer.write_hparams(config)

        # setup GPU device if available and move the model and losses into configured device
        self.device, device_ids = self._prepare_device(config['no_GPUs'])

        # optimisers
        self.optimize_q_v, self.optimize_q_phi = config['optimize_q_v'],  config['optimize_q_phi']

        # all-to-one registration
        self.fixed = self.fixed_batch = {k: v.to(self.device, non_blocking=True) for k, v in self.data_loader.fixed.items()}

        # model and losses
        self.model = model.to(self.device)

        # losses
        self.data_loss = losses['data'].to(self.device)
        self.reg_loss = losses['regularisation'].to(self.device)
        self.entropy_loss = losses['entropy'].to(self.device)

        # transformation and registration modules
        self.transformation_module = transformation_module.to(self.device)
        self.registration_module = registration_module.to(self.device)

        if self.optimize_q_phi and not self.test:
            log_var_f, u_f = self.data_loader.dataset.init_log_var_f(self.fixed['im'].shape), self.data_loader.dataset.init_u_f(self.fixed['im'].shape)
            q_f = LowRankMultivariateNormalDistribution(mu=self.fixed['im'], log_var=log_var_f, u=u_f, loc_learnable=False, cov_learnable=True)
            self.q_f = q_f.to(self.device)

        # if multiple GPUs in use
        if len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids=device_ids)

            self.data_loss = nn.DataParallel(losses['data'], device_ids=device_ids)
            self.reg_loss = nn.DataParallel(losses['regularisation'], device_ids=device_ids)
            self.entropy_loss = nn.DataParallel(losses['entropy'], device_ids=device_ids)

            self.transformation_module = nn.DataParallel(transformation_module, device_ids=device_ids)
            self.registration_module = nn.DataParallel(registration_module, device_ids=device_ids)

            if self.optimize_q_phi and not self.test:
                self.q_f = nn.DataParallel(q_f, device_ids=device_ids)

        # differential operator for use with the transformation Jacobian
        self.diff_op = get_module_attr(self.reg_loss, 'diff_op')

        # training logic
        self.start_epoch, self.step = 1, 0
        self.no_epochs, self.no_iters_q_v = int(cfg_trainer['no_epochs']), int(cfg_trainer['no_iters_q_v'])
        self.no_batches = len(self.data_loader)
        self.log_period = int(cfg_trainer['log_period'])

        # resuming
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        if self.test:
            self.optimize_q_phi = False

        # prints
        if self.optimize_q_phi:
            self._enable_gradients_model()

        self.logger.info(model)
        self.logger.info('')

        if self.optimize_q_phi:
            self._disable_gradients_model()

    @abstractmethod
    def _train_epoch(self):
        """
        training logic for an epoch
        """

        raise NotImplementedError

    @abstractmethod
    def _test(self):
        """
        testing logic for a dataset
        """

        raise NotImplementedError

    def train(self):
        """
        full training logic
        """

        for epoch in range(self.start_epoch, self.no_epochs+1):
            self._train_epoch(epoch)
            self.no_iters_q_v = self.no_iters_q_v / 2 if epoch == 3 else self.no_iters_q_v

    def eval(self):
        """
        full testing logic
        """

        self._train_epoch(epoch=1)
        self._test(no_samples=50)

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

    @staticmethod
    def _enable_gradients_variational_parameters(var_params):
        for param_key in var_params:
            var_params[param_key].requires_grad_(True)

    @staticmethod
    def _disable_gradients_variational_parameters(var_params):
        for param_key in var_params:
            var_params[param_key].requires_grad_(False)

    def _enable_gradients_model(self):
        if not self.test:
            get_module_attr(self.q_f, 'enable_gradients')()

        get_module_attr(self.model, 'enable_gradients')()

    def _disable_gradients_model(self):
        if not self.test:
            get_module_attr(self.q_f, 'disable_gradients')

        get_module_attr(self.model, 'disable_gradients')()

    def __init_optimizer_q_f(self):
        trainable_params_q_f = filter(lambda p: p.requires_grad, self.q_f.parameters())
        self.optimizer_q_f = self.config.init_obj('optimizer_q_f', torch.optim, trainable_params_q_f)

    def __init_optimizer_q_phi(self):
        trainable_params_q_phi = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer_q_phi = self.config.init_obj('optimizer_q_phi', torch.optim, trainable_params_q_phi)

    def _init_optimizers(self):
        if self.optimize_q_v:
            self.optimizer_q_v = None

        if self.optimize_q_phi:
            self._enable_gradients_model()
            self.__init_optimizer_q_f()
            self.__init_optimizer_q_phi()
            self._disable_gradients_model()

    def _save_checkpoint(self, epoch):
        filename = str(self.config.save_dir / f'checkpoint_{epoch}.pth')
        self.logger.info(f'\nsaving checkpoint: {filename}..')

        state = {'epoch': epoch, 'step': self.step, 'config': self.config}

        if self.optimize_q_phi:
            if isinstance(self.q_f, nn.DataParallel):
                state['q_f'] = self.q_f.module.state_dict()
            else:
                state['q_f'] = self.q_f.state_dict()

            if isinstance(self.model, nn.DataParallel):
                state['model'] = self.model.module.state_dict()
            else:
                state['model'] = self.model.state_dict()

            state['optimizer_q_f'] = self.optimizer_q_f.state_dict()
            state['optimizer_q_phi'] = self.optimizer_q_phi.state_dict()

        torch.save(state, filename)
        self.logger.info('checkpoint saved\n')

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'\nloading checkpoint: {resume_path}')

        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1 if not self.test else 1
        self.step = checkpoint['step'] + 1 if not self.test else 0

        if self.test:
            self.model.load_state_dict(checkpoint['model'])
            self._disable_gradients_model()
            self.logger.info('checkpoint loaded\n')

            return

        if self.optimize_q_phi:
            self._enable_gradients_model()

            self.__init_optimizer_q_f()
            self.optimizer_q_f.load_state_dict(checkpoint['optimizer_q_f'])
            self.q_f.load_state_dict(checkpoint['q_f'])

            self.__init_optimizer_q_phi()
            self.optimizer_q_phi.load_state_dict(checkpoint['optimizer_q_phi'])
            self.model.load_state_dict(checkpoint['model'])

            self._disable_gradients_model()
            self.logger.info('checkpoint loaded\n')

            return
