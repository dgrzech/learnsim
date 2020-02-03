from abc import abstractmethod
from logger import TensorboardWriter

import torch


class BaseTrainer:
    """
    base class for all trainers
    """

    def __init__(self, data_loss, reg_loss, entropy_loss,
                 transformation_model, registration_module, metric_ftns, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available and move the model and losses into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.transformation_model = transformation_model.to(self.device)
        self.registration_module = registration_module.to(self.device)

        self.data_loss = data_loss.to(self.device)
        self.reg_loss = reg_loss.to(self.device)
        self.entropy_loss = entropy_loss.to(self.device)

        if len(device_ids) > 1:
            self.transformation_model = torch.nn.DataParallel(transformation_model, device_ids=device_ids)
            self.registration_module = torch.nn.DataParallel(registration_module, device_ids=device_ids)

            self.data_loss = torch.nn.DataParallel(data_loss, device_ids=device_ids)
            self.reg_loss = torch.nn.DataParallel(reg_loss, device_ids=device_ids)
            self.entropy_loss = torch.nn.DataParallel(entropy_loss, device_ids=device_ids)

        # metrics
        self.metric_ftns = metric_ftns

        # training logic
        cfg_trainer = config['trainer']

        self.start_epoch = 1
        self.epochs = 1

        self.no_steps_v = int(cfg_trainer['no_steps_v'])
        self.no_samples = int(cfg_trainer['no_samples'])
        self.no_steps_burn_in = int(cfg_trainer['no_steps_burn_in'])

        # setup visualization writer instance
        self.log_step = cfg_trainer['log_step']
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

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
