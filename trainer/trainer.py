from base import BaseTrainer
from logger import save_images
from utils import grid_to_deformation_field, inf_loop, MetricTracker, sample_qv, sample_qf

from torch import nn

import numpy as np
import os
import torch


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, enc, data_loss, reg_loss, entropy, transformation_model, registration_module,
                 metric_ftns, config, data_loader, valid_data_loader=None, len_epoch=None):
        super().__init__(enc, data_loss, reg_loss, entropy, transformation_model, registration_module,
                         metric_ftns, config)

        self.config = config
        self.data_loader = data_loader

        self.learn_var_params = config['trainer']['learn_var_params']  # whether to learn the variational parameters
        self.learn_sim_metric = config['trainer']['learn_sim_metric']  # whether to learn the similarity metric

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
        self.train_metrics = MetricTracker('loss', *[m for m in self.metric_ftns], writer=self.writer)

    def _save_tensors(self, im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f):
        """
        save the variational parameters to disk in order to load at the next epoch
        """

        mu_v = mu_v.cpu()

        log_var_v, u_v = log_var_v.cpu(), u_v.cpu()
        log_var_f, u_f = log_var_f.cpu(), u_f.cpu()

        im_pair_idxs = im_pair_idxs.tolist()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            torch.save(mu_v[loop_idx, :, :, :, :], os.path.join(self.data_loader.save_dirs['mu_v'],
                                                                'mu_v_' + str(im_pair_idx) + '.pt'))

            torch.save(log_var_v[loop_idx, :, :, :, :], os.path.join(self.data_loader.save_dirs['log_var_v'],
                                                                     'log_var_v_' + str(im_pair_idx) + '.pt'))
            torch.save(log_var_f[loop_idx, :, :, :, :], os.path.join(self.data_loader.save_dirs['log_var_f'],
                                                                     'log_var_f_' + str(im_pair_idx) + '.pt'))

            torch.save(u_v[loop_idx, :, :, :, :], os.path.join(self.data_loader.save_dirs['u_v'],
                                                               'u_v_' + str(im_pair_idx) + '.pt'))
            torch.save(u_f[loop_idx, :, :, :, :], os.path.join(self.data_loader.save_dirs['u_f'],
                                                               'u_f_' + str(im_pair_idx) + '.pt'))

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: integer, current training epoch
        :return: log that contains average loss and metric in this epoch
        """

        self.train_metrics.reset()

        for batch_idx, (im_pair_idxs, im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f, identity_grid) \
                in enumerate(self.data_loader):
            im_fixed, im_moving = im_fixed.to(self.device, non_blocking=True), \
                                  im_moving.to(self.device, non_blocking=True)  # images to register

            mu_v = mu_v.to(self.device, non_blocking=True).requires_grad_(True)  # mean velocity field

            log_var_v, u_v = log_var_v.to(self.device, non_blocking=True).requires_grad_(True), \
                             u_v.to(self.device, non_blocking=True).requires_grad_(True)
            log_var_f, u_f = log_var_f.to(self.device, non_blocking=True).requires_grad_(True), \
                             u_f.to(self.device, non_blocking=True).requires_grad_(True)  # variational parameters

            identity_grid = identity_grid.to(self.device, non_blocking=True).requires_grad_(False)

            total_loss = 0.0

            # print value of the data term before registration
            with torch.no_grad():
                transformation = self.transformation_model(identity_grid, mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)
                im_out_unwarped = self.enc(im_fixed, im_moving)
                im_out = self.enc(im_fixed, im_moving_warped)

                print(f'\nPRE-REGISTRATION: ' +
                      f'unwarped: {self.data_loss(im_out_unwarped).mean().item():.5f}' +
                      f', warped w/ the current estimate: {self.data_loss(im_out).mean().item():.5f}\n'
                      )

            # initialise the optimisers
            if self.learn_var_params:
                optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v, log_var_v, u_v])
                optimizer_f = self.config.init_obj('optimizer_f', torch.optim, [log_var_f, u_f])
            else:
                optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v])

            """
            optimise q_v
            """

            self.enc.eval()
            
            if isinstance(self.enc, nn.DataParallel):
                self.enc.module.set_grad_enabled(False)
            else:
                self.enc.set_grad_enabled(False)

            for iter_no in range(self.no_steps_v):
                optimizer_v.zero_grad()
                data_term = 0.0
                
                if self.learn_var_params:
                    for _ in range(self.no_samples):  # draw samples from q_v
                        v_sample = sample_qv(mu_v, log_var_v, u_v)
                        transformation = self.transformation_model(identity_grid, v_sample)

                        im_moving_warped = self.registration_module(im_moving, transformation)
                        im_out = self.enc(im_fixed, im_moving_warped)

                        data_term_sample = self.data_loss(im_out).sum() / float(self.no_samples)
                        data_term += data_term_sample
                else:
                    transformation = self.transformation_model(identity_grid, mu_v)

                    im_moving_warped = self.registration_module(im_moving, transformation)
                    im_out = self.enc(im_fixed, im_moving_warped)

                    data_term_sample = self.data_loss(im_out).sum()
                    data_term += data_term_sample

                reg_term = self.reg_loss(mu_v).sum()
                entropy_term = self.entropy(log_var_v, u_v).sum()

                loss_qv = data_term + reg_term + entropy_term
                loss_qv.backward()
                optimizer_v.step()

                if iter_no == 0 or iter_no % 16 == 0 or iter_no == self.no_steps_v - 1:
                    print(f'ITERATION ' + str(iter_no) + '/' + str(self.no_steps_v - 1) +
                          f', TOTAL ENERGY: {loss_qv.item():.5f}' +
                          f'\ndata: {data_term.item():.5f}' +
                          f', regularisation: {reg_term.item():.5f}' +
                          f', entropy: {entropy_term.item():.5f}'
                          )

                total_loss += (loss_qv.item() / float(self.no_steps_v))

            mu_v.requires_grad_(False)
            log_var_v.requires_grad_(False)
            u_v.requires_grad_(False)

            """
            optimise q_phi
            """
            
            if self.learn_sim_metric:
                self.enc.train()
                
                if isinstance(self.enc, nn.DataParallel):
                    self.enc.module.set_grad_enabled(True)
                else:
                    self.enc.set_grad_enabled(True)

                optimizer_f.zero_grad()
                loss_qphi = 0.0

                for _ in range(self.no_samples):  # draw samples from q_v
                    v_sample = sample_qv(mu_v, log_var_v, u_v)
                    transformation = self.transformation_model(identity_grid, v_sample)

                    im_moving_warped = self.registration_module(im_moving, transformation)
                    im_out = self.enc(im_fixed, im_moving_warped)

                    data_term_sample = self.data_loss(im_out).sum() / float(self.no_samples)
                    loss_qphi += data_term_sample

                    for _ in range(self.no_samples):  # draw samples from q_f
                        f_sample = sample_qf(im_fixed, log_var_f, u_f)
                        im_out = self.enc(f_sample, im_moving_warped)

                        data_term_sample = self.data_loss(im_out).sum() / float(self.no_samples ** 2)
                        loss_qphi -= data_term_sample

                loss_qphi.backward()
                optimizer_f.step()

                total_loss += loss_qphi.item()

            log_var_f.requires_grad_(False)
            u_f.requires_grad_(False)

            self._save_tensors(im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f)  # save the updated tensors

            # metrics
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', total_loss)

            with torch.no_grad():
                transformation = self.transformation_model(identity_grid, mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)
                im_out = self.enc(im_fixed, im_moving_warped)

                data_term = self.data_loss(im_out).mean()
                reg_term = self.reg_loss(mu_v).mean()
                entropy_term = self.entropy(log_var_v, u_v).mean()

                self.train_metrics.update('data_term', data_term.item())
                self.train_metrics.update('reg_term', reg_term.item())
                self.train_metrics.update('entropy_term', entropy_term.item())

                # save the images
                warp_field = grid_to_deformation_field(identity_grid, transformation)

                save_images(im_pair_idxs, self.data_loader.save_dirs, im_fixed, im_moving, im_moving_warped, 
                            mu_v, log_var_v, u_v, log_var_f, u_f, warp_field)
                print('\nsaved the output images and vector fields to disk\n')

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), total_loss / self.data_loader.batch_size))

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
