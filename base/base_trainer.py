import os
from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import MetricTracker, TensorboardWriter
from utils import init_grid_im


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, config, data_loader, model, losses, transformation_module, registration_module, metrics, is_test=False):
        self.config = config
        self.logger = config.logger
        self.is_test = is_test
        self.save_dirs = data_loader.save_dirs

        try:
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        except:
            self.rank, self.world_size = 0, 1

        self.tqdm_disable = not self.rank == 0

        self.data_loader = data_loader
        self.im_spacing, self.structures_dict = self.data_loader.im_spacing, self.data_loader.structures_dict
        self.grid_im = init_grid_im(data_loader.dims).to(self.rank)

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
        self.no_batches = len(self.data_loader)

        if not self.is_test:
            self.no_samples_SGLD, self.no_samples_SGLD_burn_in = cfg_trainer['no_samples_SGLD'], cfg_trainer['no_samples_SGLD_burn_in']
            self.tau = config['optimizer_LD']['args']['lr']
        else:
            self.no_samples_test = cfg_trainer['no_samples_test']

        # resuming
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        # metrics and prints
        self.writer = TensorboardWriter(config.log_dir)
        self.metrics = MetricTracker(*metrics, writer=self.writer)

        self.writer.write_graph(self.model)
        self.writer.write_hparams(config)

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

            try:
                dist.barrier()
            except:
                pass

    def test(self):
        """
        full testing logic
        """

        self.start_epoch, self.no_epochs = 1, 1
        self.train()
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
        assert not self.is_test  # only to be used in training

        self.model.enable_grads()
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        self.__init_optimizer_q_phi()
        dist.barrier()

    def _disable_gradients_model(self):
        assert not self.is_test  # to be used only in training
        
        self.model = self.model.module
        self.model.disable_grads()
        dist.barrier()

    def _init_optimizers(self):
        self.optimizer_q_v = None

        if not self.is_test:
            self.optimizer_LD_positive, self.optimizer_LD_negative, self.optimizer_q_phi = None, None, None

    def __init_optimizer_q_phi(self):
        trainable_params_q_phi = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_q_phi = self.config.init_obj('optimizer_q_phi', torch.optim, trainable_params_q_phi)
        
        if self.optimizer_q_phi is not None:
            optimizer_q_phi.load_state_dict(self.optimizer_q_phi.state_dict())

        self.optimizer_q_phi = optimizer_q_phi

    def _save_checkpoint(self, epoch):
        if self.rank == 0:
            filename = os.path.join(self.config.checkpoints_dir, f'checkpoint_{epoch}.pt')
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

        if self.is_test:
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
