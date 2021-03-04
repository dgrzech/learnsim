from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import TensorboardWriter
from utils import MetricTracker


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, config, data_loader, model, losses, transformation_module, registration_module, metrics, test_only):
        self.config = config
        self.logger = config.get_logger('train')
        self.test_only = test_only

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.data_loader = data_loader
        self.im_spacing, self.structures_dict = self.data_loader.im_spacing, self.config.structures_dict
        self.save_dirs = self.data_loader.save_dirs

        # setup visualization writer instance
        cfg_trainer = config['trainer']

        if self.rank == 0:
            self.writer = TensorboardWriter(config.log_dir, cfg_trainer['tensorboard'])
            self.writer.write_graph(model)
            self.writer.write_hparams(config)

        # optimisers
        self.optimize_q_v, self.optimize_q_phi = config['optimize_q_v'], config['optimize_q_phi']

        # all-to-one registration
        self.fixed = self.fixed_batch = {k: v.to(self.rank) for k, v in self.data_loader.fixed.items()}

        # model and losses
        self.model = model.to(self.rank)

        if not self.test_only:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        if self.optimize_q_phi and not self.test_only:
            import model.distributions as distr

            self.q_f = self.config.init_obj('q_f', distr, self.fixed['im']).to(self.rank)
            self.q_f = DDP(self.q_f, device_ids=[self.rank], find_unused_parameters=True)

        # training logic
        self.start_epoch, self.no_epochs = 1, int(cfg_trainer['no_epochs'])
        self.step, self.no_iters_q_v = 0, int(cfg_trainer['no_iters_q_v'])
        self.no_batches = len(self.data_loader)

        self.log_period = int(cfg_trainer['log_period'])
        self.var_params_backup_period = int(cfg_trainer['var_params_backup_period'])

        # resuming
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.data_loss = losses['data'].to(self.rank)
        self.reg_loss = losses['regularisation'].to(self.rank)
        self.entropy_loss = losses['entropy'].to(self.rank)

        # transformation and registration modules
        self.transformation_module = transformation_module.to(self.rank)
        self.registration_module = registration_module.to(self.rank)

        # differential operator for use with the transformation Jacobian
        self.diff_op = self.reg_loss.diff_op

        # metrics
        if self.rank == 0:
            self.metrics = MetricTracker(*[m for m in metrics], writer=self.writer)

        if self.test_only:
            self.no_samples_test = cfg_trainer['no_samples_test']
            self.optimize_q_phi = False

        # prints
        if self.rank == 0:
            print(model)
            print('')

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

        for epoch in range(self.start_epoch, self.no_epochs + 1):
            self._no_iters_q_v_scheduler(epoch)
            self._train_epoch(epoch)

            if epoch % self.var_params_backup_period == 0:
                self.config.copy_var_params_to_backup_dirs(epoch)

    def test(self):
        """
        full testing logic
        """

        self._train_epoch(epoch=1)
        self._test(no_samples=self.no_samples_test)

    @staticmethod
    def _enable_gradients_variational_parameters(var_params):
        for param_key in var_params:
            var_params[param_key].requires_grad_(True)

    @staticmethod
    def _disable_gradients_variational_parameters(var_params):
        for param_key in var_params:
            var_params[param_key].requires_grad_(False)

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
            self.__init_optimizer_q_f()
            self.__init_optimizer_q_phi()

    def _no_iters_q_v_scheduler(self, epoch):
        """
        scheduler for the no. of iterations in an epoch
        """

        cfg_trainer = self.config['trainer']

        if epoch % 2 == 1:
            self.no_iters_q_v = int(cfg_trainer['no_iters_q_v'])
        elif epoch % 2 == 0:
            self.no_iters_q_v = int(cfg_trainer['no_iters_q_v']) // 2

    def _save_checkpoint(self, epoch):
        if self.rank == 0:
            filename = str(self.config.save_dir / f'checkpoint_{epoch}.pth')
            print(f'\nsaving checkpoint: {filename}..')

            state = {'epoch': epoch, 'step': self.step, 'config': self.config}

            if self.optimize_q_phi:
                state['q_f'] = self.q_f.module.state_dict() if isinstance(self.q_f, DDP) else self.q_f.state_dict()
                state['optimizer_q_f'] = self.optimizer_q_f.state_dict()

                state['model'] = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
                state['optimizer_q_phi'] = self.optimizer_q_phi.state_dict()

            torch.save(state, filename)
            print('checkpoint saved\n')

        # dist.barrier()  # NOTE (DG): not really needed cause other processes don't read the file

    def _resume_checkpoint(self, resume_path):
        if self.rank == 0:
            print(f'\nloading checkpoint: {resume_path}')

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path, map_location=map_location)

        resume_epoch = checkpoint['epoch']
        self.start_epoch = resume_epoch + 1 if not self.test_only else 1
        self.step = checkpoint['step'] + 1 if not self.test_only else 0

        if self.test_only:
            self.model.load_state_dict(checkpoint['model'])

            if self.rank == 0:
                print('checkpoint loaded\n')

            dist.barrier()
            return

        if self.optimize_q_phi:
            self.__init_optimizer_q_f()
            self.optimizer_q_f.load_state_dict(checkpoint['optimizer_q_f'])
            self.q_f.load_state_dict(checkpoint['q_f'])

            self.__init_optimizer_q_phi()
            self.optimizer_q_phi.load_state_dict(checkpoint['optimizer_q_phi'])
            self.model.load_state_dict(checkpoint['model'])

            if self.rank == 0:
                self.config.copy_var_params_from_backup_dirs(resume_epoch)
                self.config.remove_backup_dirs()
                print('checkpoint loaded\n')

            dist.barrier()
            return
