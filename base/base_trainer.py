from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

import torch


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, enc, data_loss, reg_loss, entropy, transformation_model, registration_module,
                 metric_ftns, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available and move the model and losses into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.enc = enc.to(self.device)
        self.transformation_model = transformation_model.to(self.device)
        self.registration_module = registration_module.to(self.device)

        self.data_loss = data_loss.to(self.device)
        self.reg_loss = reg_loss.to(self.device)
        self.entropy = entropy.to(self.device)

        if len(device_ids) > 1:
            self.enc = torch.nn.DataParallel(enc, device_ids=device_ids)
            self.transformation_model = torch.nn.DataParallel(transformation_model, device_ids=device_ids)
            self.registration_module = torch.nn.DataParallel(registration_module, device_ids=device_ids)

            self.data_loss = torch.nn.DataParallel(data_loss, device_ids=device_ids)
            self.reg_loss = torch.nn.DataParallel(reg_loss, device_ids=device_ids)
            self.entropy = torch.nn.DataParallel(entropy, device_ids=device_ids)

        # encoder optimiser
        trainable_params = filter(lambda p: p.requires_grad, enc.parameters())
        self.optimizer_phi = config.init_obj('optimizer_phi', torch.optim, trainable_params)

        # metrics
        self.metric_ftns = metric_ftns

        # training logic
        cfg_trainer = config['trainer']

        self.epochs = int(cfg_trainer['epochs'])
        self.no_samples = int(cfg_trainer['no_samples'])
        self.no_steps_v = int(cfg_trainer['no_steps_v'])

        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: Current epoch number
        """

        raise NotImplementedError

    def train(self):
        """
        full training logic
        """

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                if isinstance(value, int):
                    self.logger.info(f'    {str(key):15s}: {value}')
                else:
                    self.logger.info(f'    {str(key):15s}: {value:.5f}')

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

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

    def _save_checkpoint(self, epoch, save_best=False):
        """
        saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        arch = type(self.enc).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.enc.state_dict(),
            'optimizer_phi': self.optimizer_phi.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {} ...".format(resume_path))
        
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("warning: architecture configuration given in config file is different from that of "
                                "checkpoint; this may yield an exception while state_dict is being loaded.")
        self.enc.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_phi']['type'] != self.config['optimizer_phi']['type']:
            self.logger.warning("warning: optimiser type given in config file is different from that of checkpoint; "
                                "optimizer parameters not being resumed")
        else:
            self.optimizer_phi.load_state_dict(checkpoint['optimizer_phi'])

        self.logger.info("checkpoint loaded, resuming training from epoch {}\n".format(self.start_epoch))
