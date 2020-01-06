from base import BaseTrainer
from model.metric import dice
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

        self.learn_q_v = config['trainer']['learn_q_v']  # whether to learn the variational parameters of q_v
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
        
        # optimisers
        if isinstance(self.enc, nn.DataParallel):
            self.enc.module.set_grad_enabled(True)
        else:
            self.enc.set_grad_enabled(True)

        self.optimizer_q_f = None
        self.optimizer_q_phi = self.config.init_obj('optimizer_phi', torch.optim, self.enc.parameters())
        self.optimizer_q_v = None
        
        if isinstance(self.enc, nn.DataParallel):
            self.enc.module.set_grad_enabled(False)
        else:
            self.enc.set_grad_enabled(False)

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

    def _reg_print(self, iter_no, no_steps_v, curr_batch_size, loss_q_v, data_term, reg_term, entropy_term):
        """
        print value of the energy function at a given step of registration
        """

        loss_q_v /= curr_batch_size

        data_term /= curr_batch_size
        reg_term /= curr_batch_size
        entropy_term /= curr_batch_size

        self.logger.info(f'ITERATION ' + str(iter_no) + '/' + str(no_steps_v - 1) +
                         f', TOTAL ENERGY: {loss_q_v:.5f}' +
                         f'\ndata: {data_term:.5f}' +
                         f', regularisation: {reg_term:.5f}' +
                         f', entropy: {entropy_term:.5f}'
                         )

    def _step_q_v(self, curr_batch_size, im_fixed, im_moving, mu_v, log_var_v, u_v, identity_grid):
        """
        update parameters of q_v
        """

        loss_q_v = 0.0

        if self.learn_q_v:
            # enable gradients
            mu_v.requires_grad_(True)
            log_var_v.requires_grad_(True)
            u_v.requires_grad_(True)

            # initialise the optimiser
            self.optimizer_q_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v, log_var_v, u_v])

            # optimise q_v
            for iter_no in range(self.no_steps_v):
                self.optimizer_q_v.zero_grad()
                data_term = 0.0

                for _ in range(self.no_samples):
                    v_sample = sample_qv(mu_v, log_var_v, u_v)
                    transformation = self.transformation_model(identity_grid, v_sample)

                    im_moving_warped = self.registration_module(im_moving, transformation)
                    im_out = self.enc(im_fixed, im_moving_warped)

                    data_term_sample = self.data_loss(im_out).sum() / float(self.no_samples)
                    data_term += data_term_sample

                reg_term = self.reg_loss(mu_v).sum()
                entropy_term = self.entropy(log_var_v, u_v).sum()

                loss_q_v = data_term + reg_term + entropy_term
                loss_q_v.backward()
                self.optimizer_q_v.step()  # backprop

                if iter_no == 0 or iter_no % 16 == 0 or iter_no == self.no_steps_v - 1:
                    self._reg_print(iter_no, self.no_steps_v, curr_batch_size,
                                    loss_q_v.item(), data_term.item(), reg_term.item(), entropy_term.item())
            
            # disable gradients
            mu_v.requires_grad_(False)
            log_var_v.requires_grad_(False)
            u_v.requires_grad_(False)
        else:
            mu_v.requires_grad_(True)
            self.optimizer_q_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v])

            for iter_no in range(self.no_steps_v):
                self.optimizer_q_v.zero_grad()
                transformation = self.transformation_model(identity_grid, mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)
                im_out = self.enc(im_fixed, im_moving_warped)

                data_term = self.data_loss(im_out).sum()
                reg_term = self.reg_loss(mu_v).sum()

                loss_q_v = data_term + reg_term
                loss_q_v.backward()
                self.optimizer_q_v.step()

                if iter_no == 0 or iter_no % 16 == 0 or iter_no == self.no_steps_v - 1:
                    entropy_term = self.entropy(log_var_v, u_v)
                    self._reg_print(iter_no, self.no_steps_v, curr_batch_size,
                                    loss_q_v.item(), data_term.item(), reg_term.item(), entropy_term.item())
            
            mu_v.requires_grad_(False)

        return loss_q_v.item() / curr_batch_size

    def _step_q_f_q_phi(self, curr_batch_size, im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f, identity_grid):
        """
        update parameters of q_f and q_phi
        """

        # enable gradients
        log_var_f.requires_grad_(True)
        u_f.requires_grad_(True)

        self.enc.train()
        if isinstance(self.enc, nn.DataParallel):
            self.enc.module.set_grad_enabled(True)
        else:
            self.enc.set_grad_enabled(True)
        
        # initialise the optimiser
        self.optimizer_q_f = self.config.init_obj('optimizer_f', torch.optim, [log_var_f, u_f])

        # optimise the encoding function
        self.optimizer_q_f.zero_grad()
        self.optimizer_q_phi.zero_grad()

        loss_q_f_q_phi = 0.0
        
        for _ in range(self.no_samples):
            v_sample = sample_qv(mu_v, log_var_v, u_v)  # draw a sample from q_v
            transformation = self.transformation_model(identity_grid, v_sample)

            im_moving_warped = self.registration_module(im_moving, transformation)
            im_out = self.enc(im_fixed, im_moving_warped)

            data_term_sample = self.data_loss(im_out).sum() / float(self.no_samples)
            loss_q_f_q_phi += data_term_sample
            
            for _ in range(self.no_samples):
                im_fixed_sample = sample_qf(im_fixed, log_var_f, u_f)  # draw a sample from q_f
                im_out = self.enc(im_fixed_sample, im_moving_warped)

                data_term_sample = self.data_loss(im_out).sum() / float(self.no_samples ** 2)
                loss_q_f_q_phi -= data_term_sample

        loss_q_f_q_phi /= curr_batch_size
        loss_q_f_q_phi.backward()

        self.optimizer_q_f.step()  # backprop
        self.optimizer_q_phi.step()  
        
        # disable gradients
        log_var_f.requires_grad_(False)
        u_f.requires_grad_(False)

        self.enc.eval()
        if isinstance(self.enc, nn.DataParallel):
            self.enc.module.set_grad_enabled(False)
        else:
            self.enc.set_grad_enabled(False)

        return loss_q_f_q_phi.item()

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: integer, current training epoch
        :return: log that contains average loss and metric in this epoch
        """

        self.train_metrics.reset()

        for batch_idx, (im_pair_idxs, im_fixed, seg_fixed, im_moving, seg_moving,
                        mu_v, log_var_v, u_v, log_var_f, u_f, identity_grid) in enumerate(self.data_loader):
            im_fixed, im_moving = im_fixed.to(self.device, non_blocking=True), \
                                  im_moving.to(self.device, non_blocking=True)  # images to register

            seg_fixed, seg_moving = seg_fixed.to(self.device, non_blocking=True), \
                                    seg_moving.to(self.device, non_blocking=True)  # corresponding segmentations

            mu_v = mu_v.to(self.device, non_blocking=True)  # mean velocity field

            log_var_v, u_v = log_var_v.to(self.device, non_blocking=True), \
                             u_v.to(self.device, non_blocking=True)
            log_var_f, u_f = log_var_f.to(self.device, non_blocking=True), \
                             u_f.to(self.device, non_blocking=True)  # variational parameters

            identity_grid = identity_grid.to(self.device, non_blocking=True)
            curr_batch_size = float(im_pair_idxs.numel())

            # print value of the data term before registration
            with torch.no_grad():
                transformation = self.transformation_model(identity_grid, mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)
                im_out_unwarped = self.enc(im_fixed, im_moving)
                im_out = self.enc(im_fixed, im_moving_warped)
                
                loss_unwarped = self.data_loss(im_out_unwarped).sum() / curr_batch_size
                loss_warped = self.data_loss(im_out).sum() / curr_batch_size

                self.logger.info(f'\nPRE-REGISTRATION: ' +
                                 f'unwarped: {loss_unwarped:.5f}' +
                                 f', warped w/ the current estimate: {loss_warped:.5f}\n'
                                 )

            """
            training
            """

            total_loss = 0.0
            loss_q_f_q_phi = 0.0

            loss_q_v = self._step_q_v(curr_batch_size, im_fixed, im_moving,
                                      mu_v, log_var_v, u_v, identity_grid)  # q_v
            total_loss += loss_q_v

            if self.learn_sim_metric:
                self.logger.info('\noptimising q_f and q_phi..')
                loss_q_f_q_phi = self._step_q_f_q_phi(curr_batch_size, im_fixed, im_moving, 
                                                      mu_v, log_var_v, u_v, log_var_f, u_f, identity_grid)  # q_phi
                total_loss += loss_q_f_q_phi

            # save the updated tensors
            self._save_tensors(im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f) 

            # metrics
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', total_loss)

            with torch.no_grad():
                transformation = self.transformation_model(identity_grid, mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)
                im_out = self.enc(im_fixed, im_moving_warped)

                data_term = self.data_loss(im_out).sum() / curr_batch_size
                reg_term = self.reg_loss(mu_v).sum() / curr_batch_size
                entropy_term = self.entropy(log_var_v, u_v).sum() / curr_batch_size

                self.train_metrics.update('data_term', data_term.item())
                self.train_metrics.update('reg_term', reg_term.item())
                self.train_metrics.update('entropy_term', entropy_term.item())

                seg_moving_warped = self.registration_module(seg_moving, transformation, mode='nearest')
                dsc = dice(seg_fixed, seg_moving_warped)
                
                for class_idx, val in enumerate(dsc):
                    self.train_metrics.update('dice_' + str(class_idx + 1), val)

                # save the images
                warp_field = grid_to_deformation_field(identity_grid, transformation)

                save_images(im_pair_idxs, self.data_loader.save_dirs, im_fixed, im_moving, im_moving_warped, 
                            mu_v, log_var_v, u_v, log_var_f, u_f, warp_field)
                self.logger.info('\nsaved the output images and vector fields to disk\n')

            if batch_idx % self.log_step == 0:
                self.logger.info(
                        f'train epoch: {epoch} {self._progress(batch_idx)}\n' +
                        f'loss: {total_loss:.5f}\n' +
                        f'loss_q_v: {loss_q_v:.5f}, loss_q_f_q_phi: {loss_q_f_q_phi:.5f}'
                                )

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
