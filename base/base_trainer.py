from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import TensorboardWriter
from utils import MetricTracker, init_grid_im


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, config, data_loader, model, losses, transformation_module, registration_module, metrics, test_only):
        self.config = config
        self.logger = config.logger
        self.test_only = test_only

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tqdm_disable = not self.rank == 0

        self.data_loader = data_loader
        self.atlas_mode = data_loader.atlas_mode

        if self.atlas_mode:
            self.fixed = data_loader.fixed

            for key in self.fixed:
                self.fixed[key] = self.fixed[key].to(self.rank, memory_format=torch.channels_last_3d)

        self.im_spacing, self.structures_dict = self.data_loader.im_spacing, self.data_loader.structures_dict
        self.grid_im = init_grid_im(data_loader.dims).to(self.rank)
        self.save_dirs = self.data_loader.save_dirs

        # model and losses
        self.model = model.to(self.rank, memory_format=torch.channels_last_3d)

        self.data_loss = losses['data'].to(self.rank)
        self.reg_loss = losses['regularisation'].to(self.rank)
        self.entropy_loss = losses['entropy'].to(self.rank)

        # transformation and registration modules
        self.transformation_module = transformation_module.to(self.rank)
        self.registration_module = registration_module.to(self.rank)

        # differential operator for use with the transformation Jacobian
        self.diff_op = self.reg_loss.diff_op

        # training logic
        cfg_trainer = config['trainer']

        self.start_epoch, self.no_epochs = 1, int(cfg_trainer['no_epochs'])
        self.step, self.no_iters_q_v = 0, int(cfg_trainer['no_iters_q_v'])
        self.no_samples_SGLD, self.tau = cfg_trainer['no_samples_SGLD'], config['optimizer_LD']['args']['lr']

        if not self.atlas_mode:
            self.no_im_pairs_per_epoch = int(cfg_trainer['no_im_pairs_per_epoch'])

        if self.atlas_mode:
            self.no_batches = len(self.data_loader)
        else:
            cfg_data_loader = config['data_loader']['args']
            batch_size = cfg_data_loader['batch_size']

            self.no_batches = self.no_im_pairs_per_epoch // (self.world_size * batch_size)

        self.log_period = int(cfg_trainer['log_period'])  # NOTE (DG): unused
        self.log_period_model_samples = int(cfg_trainer['log_period_model_samples'])
        self.log_period_var_params = int(cfg_trainer['log_period_var_params']) if 'log_period_var_params' in cfg_trainer else None  # NOTE (DG): unused

        if self.test_only:
            self.no_samples_test = cfg_trainer['no_samples_test']
            self.time_only = cfg_trainer['time_only'] if 'time_only' in cfg_trainer else False

        # resuming
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        # metrics and prints
        self.writer = TensorboardWriter(config.log_dir)
        self.metrics = MetricTracker(*metrics, writer=self.writer)

        self.writer.write_graph(self.model)
        self.writer.write_hparams(config.config_str)

        self.logger.info(self.model)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        training logic for an epoch
        """

        raise NotImplementedError

    @abstractmethod
    def _test(self, no_samples):
        """
        testing logic for a dataset
        """

        raise NotImplementedError

    def train(self):
        """
        full training logic
        """

        for epoch in range(self.start_epoch, self.no_epochs + 1):
            self._train_epoch(epoch)

            if self.log_period_var_params is not None and epoch % self.log_period_var_params == 0:
                if self.rank == 0:
                    self.config.copy_var_params_to_backup_dirs(epoch)

            dist.barrier()

    def test(self):
        """
        full testing logic
        """

        self.start_epoch = 1
        self.no_epochs = 1
        self.train()

        if self.time_only:
            exit()

        self._test(no_samples=self.no_samples_test)

    @staticmethod
    def _enable_gradients_variational_parameters(var_params):
        for param_key in var_params:
            var_params[param_key].requires_grad_(True)

    @staticmethod
    def _disable_gradients_variational_parameters(var_params):
        for param_key in var_params:
            var_params[param_key].requires_grad_(False)

    def _enable_gradients_model(self):
        assert not self.test_only  # only to be used in training

        self.model.enable_grads()
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        self.__init_optimizer_q_phi()
        dist.barrier()

    def _disable_gradients_model(self):
        assert not self.test_only  # to be used only in training
        
        self.model = self.model.module
        self.model.disable_grads()
        dist.barrier()

    def _init_optimizers(self):
        self.optimizer_q_v = None

        if not self.test_only:
            self.optimizer_LD = None
            self.optimizer_q_phi = None

    def __init_optimizer_q_phi(self):
        trainable_params_q_phi = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_q_phi = self.config.init_obj('optimizer_q_phi', torch.optim, trainable_params_q_phi)
        
        if self.optimizer_q_phi is not None:
            optimizer_q_phi.load_state_dict(self.optimizer_q_phi.state_dict())

        self.optimizer_q_phi = optimizer_q_phi

    def _save_checkpoint(self, epoch):
        if self.rank == 0:
            filename = str(self.config.save_dir / f'checkpoint_{epoch}.pt')
            self.logger.info(f'\nsaving checkpoint: {filename}..')

            state = {'epoch': epoch, 'step': self.step, 'config': self.config,
                     'model': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
                     'optimizer_q_phi': self.optimizer_q_phi.state_dict()}

            torch.save(state, filename)
            self.logger.info('checkpoint saved\n')

        dist.barrier()

    def _resume_checkpoint(self, resume_path):
        if self.rank == 0:
            self.logger.info(f'\nloading checkpoint: {resume_path}')

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path, map_location=map_location)

        if self.test_only:
            self.model.load_state_dict(checkpoint['model'])

            if self.rank == 0:
                self.logger.info('checkpoint loaded\n')
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.step = checkpoint['step'] + 1

            self.__init_optimizer_q_phi()
            self.optimizer_q_phi.load_state_dict(checkpoint['optimizer_q_phi'])
            self.model.load_state_dict(checkpoint['model'])

        dist.barrier()
