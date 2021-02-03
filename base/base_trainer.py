from abc import abstractmethod

import torch

from logger import TensorboardWriter
from utils import get_module_attr


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, config, data_loader, model, losses, transformation_model, registration_module):
        self.config = config
        self.data_loader = data_loader

        self.checkpoint_dir = config.save_dir
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available and move the model and losses into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        # model and losses
        self.model = model.to(self.device)
        self.logger.info(model)

        self.data_loss = losses['data'].to(self.device)
        self.reg_loss = losses['regularisation'].to(self.device)
        self.entropy_loss = losses['entropy'].to(self.device)

        self.transformation_model = transformation_model.to(self.device)
        self.registration_module = registration_module.to(self.device)

        # if multiple GPUs in use
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

            self.data_loss = torch.nn.DataParallel(losses['data'], device_ids=device_ids)
            self.reg_loss = torch.nn.DataParallel(losses['regularisation'], device_ids=device_ids)
            self.entropy_loss = torch.nn.DataParallel(losses['entropy'], device_ids=device_ids)

            self.transformation_model = torch.nn.DataParallel(transformation_model, device_ids=device_ids)
            self.registration_module = torch.nn.DataParallel(registration_module, device_ids=device_ids)

        # optimisers
        self.optimize_q_v, self.optimize_q_f, self.optimize_q_phi = config['optimize_q_v'], config['optimize_q_f'], config['optimize_q_phi']

        # differential operator for use with the transformation Jacobian
        self.diff_op = get_module_attr(self.reg_loss, 'diff_op')

        # training logic
        self.start_epoch = 1
        self.step_global = 0

        cfg_trainer = config['trainer']
        self.no_epochs = int(cfg_trainer['no_epochs'])
        self.no_iters_q_v = int(cfg_trainer['no_iters_q_v'])
        self.log_period = cfg_trainer['log_period']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, cfg_trainer['tensorboard'])
        self.writer.write_hparams(config)

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

        for epoch in range(self.start_epoch, self.no_epochs + 1):
            self._train_epoch(epoch)

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
        get_module_attr(self.model, 'enable_gradients')()

    def _disable_gradients_model(self):
        get_module_attr(self.model, 'disable_gradients')()

    def _save_checkpoint(self, epoch):
        filename = str(self.checkpoint_dir / f'checkpoint_{epoch}.pth')
        self.logger.info(f'\nsaving checkpoint: {filename}..')

        state = {'epoch': epoch, 'step_global': self.step_global,
                 'config': self.config}

        if self.optimize_q_f:
            for param_key in self.var_params_q_f:
                state[param_key] = self.var_params_q_f[param_key]

            state['optimizer_q_f'] = self.optimizer_q_f.state_dict()

        if self.optimize_q_phi:
            state['model'] = self.model.state_dict()
            state['optimizer_q_phi'] = self.optimizer_q_phi.state_dict()

        torch.save(state, filename)
        self.logger.info('checkpoint saved\n')

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'\nloading checkpoint: {resume_path}')

        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        self.step_global = checkpoint['step_global'] + 1

        if self.optimize_q_f:
            for param_key in self.var_params_q_f:
                self.var_params_q_f[param_key] = checkpoint[param_key]

            self._enable_gradients_variational_parameters(self.var_params_q_f)
            self.__init_optimizer_q_f()
            self.optimizer_q_f.load_state_dict(checkpoint['optimizer_q_f'])
            self._disable_gradients_variational_parameters(self.var_params_q_f)

        if self.optimize_q_phi:
            self.model.load_state_dict(checkpoint['model'])
            self.model.enable_gradients()
            self.__init_optimizer_q_phi()
            self.model.disable_gradients()

        self.logger.info('checkpoint loaded\n')
