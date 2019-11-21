import numpy as np
import torch

from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.sampler import sample_qv, sample_qf


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, model, data_loss, reg_loss, entropy, transformation_model, registration_module,
                 metric_ftns, config, data_loader, valid_data_loader=None, len_epoch=None):
        super().__init__(model, data_loss, reg_loss, entropy, transformation_model, registration_module, metric_ftns, config)

        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _enable_encoder_gradients(self, mode):
        if self.config['n_gpu'] == 1:
            for p in self.model.parameters():
                p.requires_grad_(mode)
        else:
            for p in self.model.module.parameters():
                p.requires_grad_(mode)

    @staticmethod
    def _save_tensors(im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f):
        mu_v = mu_v.cpu()
        log_var_v = log_var_v.cpu()
        u_v = u_v.cpu()

        log_var_f = log_var_f.cpu()
        u_f = u_f.cpu()

        im_pair_idxs = im_pair_idxs.tolist()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            torch.save(mu_v[loop_idx, :, :, :, :], './temp/mu_v/mu_v_' + str(im_pair_idx) + '.pt')
            torch.save(log_var_v[loop_idx, :, :, :, :], './temp/log_var_v/log_var_v_' + str(im_pair_idx) + '.pt')
            torch.save(u_v[loop_idx, :, :, :, :], './temp/modes_of_variation_v/u_' + str(im_pair_idx) + '.pt')

            torch.save(log_var_f[loop_idx, :, :, :, :], './temp/log_var_f/log_var_f_' + str(im_pair_idx) + '.pt')
            torch.save(u_f[loop_idx, :, :, :, :], './temp/modes_of_variation_f/u_' + str(im_pair_idx) + '.pt')

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: integer, current training epoch
        :return: log that contains average loss and metric in this epoch
        """

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (im_pair_idxs, im1, im2, mu_v, log_var_v, u_v, log_var_f, u_f, identity_grid) \
                in enumerate(self.data_loader):
            im1 = im1.to(self.device, non_blocking=True)
            im2 = im2.to(self.device, non_blocking=True)

            mu_v = mu_v.to(self.device, non_blocking=True).requires_grad_(True)
            log_var_v = log_var_v.to(self.device, non_blocking=True).requires_grad_(True)
            u_v = u_v.to(self.device, non_blocking=True).requires_grad_(True)

            log_var_f = log_var_f.to(self.device, non_blocking=True).requires_grad_(True)
            u_f = u_f.to(self.device, non_blocking=True).requires_grad_(True)

            identity_grid = identity_grid.to(self.device, non_blocking=True)

            """
            optimizers
            """

            # v
            if self.optimizer_v is None:
                self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v, log_var_v, u_v])

            # f
            if self.optimizer_f is None:
                self.optimizer_f = self.config.init_obj('optimizer_f', torch.optim, [log_var_f, u_f])

            total_loss = 0.0

            """
            q_v
            """

            self._enable_encoder_gradients(False)

            for iter_no in range(self.no_steps_v):
                self.optimizer_v.zero_grad()
                loss = 0.0

                for _ in range(self.no_samples):
                    v_sample = sample_qv(mu_v, log_var_v, u_v)
                    warp_field = self.transformation_model.forward(identity_grid, v_sample)

                    im2_warped = self.registration_module.forward(im2, identity_grid, warp_field)
                    im_out = self.model(im1, im2_warped)
                    loss += self.data_loss.forward(im_out)

                loss = loss.sum() / float(self.no_samples)

                loss += self.reg_loss(mu_v).sum()
                loss -= self.entropy(log_var_v, u_v).sum()

                if iter_no == 0 or iter_no % 16 == 0 or iter_no == self.no_steps_v - 1:
                    print('iter ' + str(iter_no) + '/' + str(self.no_steps_v - 1) + ', cost: ' + str(loss.item()))

                loss.backward()
                self.optimizer_v.step()

            total_loss += loss.item()

            mu_v.requires_grad_(False)
            log_var_v.requires_grad_(False)
            u_v.requires_grad_(False)

            """
            q_phi
            """

            self._enable_encoder_gradients(True)

            self.optimizer_f.zero_grad()
            self.optimizer_phi.zero_grad()

            loss = 0.0

            for _ in range(self.no_samples):
                # first term
                v_sample = sample_qv(mu_v, log_var_v, u_v)
                warp_field = self.transformation_model.forward(identity_grid, v_sample)

                im2_warped = self.registration_module.forward(im2, identity_grid, warp_field)
                im_out = self.model(im1, im2_warped)
                loss += self.data_loss.forward(im_out)

                # second term
                for _ in range(self.no_samples):
                    f_sample = sample_qf(im1, log_var_f, u_f)
                    im_out = self.model(f_sample, im2_warped)
                    loss += (self.data_loss.forward(im_out) / float(self.no_samples))

            loss = loss.sum() / (2.0 * float(self.no_samples))
            total_loss += loss.item()

            loss.backward()

            self.optimizer_f.step()
            self.optimizer_phi.step()

            log_var_f.requires_grad_(False)
            u_f.requires_grad_(False)

            """
            logging
            """

            self.train_metrics.update('loss', total_loss)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self._save_tensors(im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f)  # save updated tensors

            with torch.no_grad():
                warp_field = self.transformation_model.forward(identity_grid, mu_v)
                im2_warped = self.registration_module.forward(im2, identity_grid, warp_field)
                im_out = self.model(im1, im2_warped)

                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(im_out))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'

        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch

        return base.format(current, total, 100.0 * current / total)
