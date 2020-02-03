from os import path

from base import BaseTrainer
from logger import log_images, log_log_det_J_transformation, log_q_v, log_v, mixing_print, registration_print, \
    save_grids, save_images
from utils import add_noise, calc_det_J, get_module_attr, inf_loop, sample_q_v, save_optimiser_to_disk, \
    separable_conv_3d, sobolev_kernel_1d, transform_coordinates, MetricTracker, SobolevGrad

import math
import numpy as np
import torch


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, data_loss, reg_loss, entropy_loss,
                 transformation_model, registration_module, metric_ftns, config, data_loader):
        super().__init__(data_loss, reg_loss, entropy_loss,
                         transformation_model, registration_module, metric_ftns, config)

        self.config = config
        self.data_loader = data_loader
        self.train_metrics = MetricTracker('loss', *[m for m in self.metric_ftns], writer=self.writer)
        
        # Sobolev kernel
        self.sobolev_grad = config['sobolev_grad']['enabled']

        if self.sobolev_grad:
            _s = config['sobolev_grad']['s']
            _lambda = config['sobolev_grad']['lambda']

            S, S_sqrt = sobolev_kernel_1d(_s, _lambda)

            S = torch.from_numpy(S).float()
            S.unsqueeze_(0)
            S = torch.stack((S, S, S), 0)

            S_x = S.unsqueeze(2).unsqueeze(2)
            S_y = S.unsqueeze(2).unsqueeze(4)
            S_z = S.unsqueeze(3).unsqueeze(4)

            self.S_x = S_x.to(self.device, non_blocking=True)
            self.S_y = S_y.to(self.device, non_blocking=True)
            self.S_z = S_z.to(self.device, non_blocking=True)

            self.padding_sz = _s // 2
        
        self.im_fixed = None
        self.seg_fixed = None
        self.mask_fixed = None

        self.optimizer_v = None  # VI
        self.optimizer_mala = None  # MCMC

        self.tau = 0.0
        self.sqrt_tau_twice = 0.0

        self.mu_v = None
        self.sigma = None
        self.u = None

        self.no_samples_accepted = 0.0
        self.no_samples_rejected = 0.0

    def _save_v(self, im_pair_idx, v):
        torch.save(v, path.join(self.data_loader.save_dirs['v'], 'v_' + str(im_pair_idx) + '.pt'))

    def _save_tensors(self, im_pair_idxs, v):
        im_pair_idxs = im_pair_idxs.tolist()
        v = v.cpu()

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            self._save_v(im_pair_idx, v[loop_idx])

    def _step_VI(self, im_pair_idxs, im_moving, mu_v, log_var_v, u_v):
        if self.optimizer_v is None:
            mu_v.requires_grad_(True)
            log_var_v.requires_grad_(True)
            u_v.requires_grad_(True)

            self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v, log_var_v, u_v])

        for iter_no in range(self.no_steps_v):
            data_term = 0.0
            reg_term = 0.0
            entropy_term = 0.0

            v_sample1, v_sample2 = sample_q_v(mu_v, log_var_v, u_v, no_samples=2)
            if self.sobolev_grad:
                v_sample1 = SobolevGrad.apply(v_sample1, self.S_x, self.S_y, self.S_z, self.padding_sz)
                v_sample2 = SobolevGrad.apply(v_sample2, self.S_x, self.S_y, self.S_z, self.padding_sz)

            transformation1, displacement1 = self.transformation_model(v_sample1)
            transformation2, displacement2 = self.transformation_model(v_sample2)

            im_moving_warped1 = self.registration_module(im_moving, transformation1)
            im_moving_warped2 = self.registration_module(im_moving, transformation2)

            data_term += self.data_loss(self.im_fixed, im_moving_warped1, self.mask_fixed).sum() / 2.0
            data_term += self.data_loss(self.im_fixed, im_moving_warped2, self.mask_fixed).sum() / 2.0

            reg_term += self.reg_loss(v_sample1).sum() / 2.0
            reg_term += self.reg_loss(v_sample2).sum() / 2.0

            entropy_term += self.entropy_loss(v_sample=v_sample1, mu_v=mu_v, log_var_v=log_var_v, u_v=u_v).sum() / 2.0
            entropy_term += self.entropy_loss(v_sample=v_sample2, mu_v=mu_v, log_var_v=log_var_v, u_v=u_v).sum() / 2.0

            entropy_term += self.entropy_loss(log_var_v=log_var_v, u_v=u_v).sum()

            self.optimizer_v.zero_grad()
            loss_q_v = data_term + reg_term - entropy_term
            loss_q_v.backward()
            self.optimizer_v.step()

            # metrics and prints
            self.writer.set_step(iter_no)

            self.train_metrics.update('data_term', data_term.item())
            self.train_metrics.update('reg_term', reg_term.item())
            self.train_metrics.update('entropy_term', entropy_term.item())
            self.train_metrics.update('total_loss', loss_q_v.item())

            if iter_no % self.log_step == 0 or iter_no == self.no_steps_v:
                registration_print(self.logger, iter_no, self.no_steps_v,
                                   loss_q_v.item(), data_term.item(), reg_term.item(), entropy_term.item())

            step = iter_no + 1
            self.writer.set_step(step)

            if math.log2(step).is_integer():
                with torch.no_grad():
                    if self.sobolev_grad:
                        mu_v_smoothed = SobolevGrad.apply(mu_v, self.S_x, self.S_y, self.S_z, self.padding_sz)
                        transformation, displacement = self.transformation_model(mu_v_smoothed)
                    else:
                        transformation, displacement = self.transformation_model(mu_v)

                    im_moving_warped = self.registration_module(im_moving, transformation)

                    # log to tensorboard and save images, fields etc.
                    nabla_x, nabla_y, nabla_z = get_module_attr(self.reg_loss, 'diff_op')(transformation)
                    det_J_transformation = calc_det_J(nabla_x, nabla_y, nabla_z)
                    log_det_J_transformation = torch.log10(det_J_transformation)

                    log_images(self.writer, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped)
                    log_log_det_J_transformation(self.writer, im_pair_idxs, log_det_J_transformation)
                    save_grids(self.data_loader.save_dirs, im_pair_idxs, transformation)

                    if self.sobolev_grad:
                        log_var_v_smoothed = SobolevGrad.apply(log_var_v, self.S_x, self.S_y, self.S_z, self.padding_sz)
                        u_v_smoothed = SobolevGrad.apply(u_v, self.S_x, self.S_y, self.S_z, self.padding_sz)

                        log_q_v(self.writer, im_pair_idxs,
                                mu_v_smoothed, displacement, log_var_v_smoothed, u_v_smoothed)
                        save_images(self.data_loader.save_dirs, im_pair_idxs,
                                    self.im_fixed, im_moving, im_moving_warped,
                                    mu_v_smoothed, log_det_J_transformation, displacement)
                    else:
                        log_q_v(self.writer, im_pair_idxs, mu_v, displacement, log_var_v, u_v)
                        save_images(self.data_loader.save_dirs, im_pair_idxs,
                                    self.im_fixed, im_moving, im_moving_warped,
                                    mu_v, log_det_J_transformation, displacement)

        mu_v.requires_grad_(False)
        log_var_v.requires_grad_(False)
        u_v.requires_grad_(False)
        
    def _step_MCMC(self, im_pair_idxs, im_moving):
        v_curr_state = self.mu_v.clone()

        if self.optimizer_mala is None:
            v_curr_state.requires_grad_(True)
            self.optimizer_mala = self.config.init_obj('optimizer_mala', torch.optim, [v_curr_state])

        self.logger.info('\nBURNING IN THE MARKOV CHAIN\n')

        for sample_no in range(self.no_samples):
            self.optimizer_mala.zero_grad()

            if sample_no % 100 == 0:
                self.logger.info('burn-in sample no. ' + str(sample_no) + '/' + str(self.no_steps_burn_in))
            
            """
            stochastic gradient Langevin dynamics
            """
            
            v_prev_state = v_curr_state.clone()
            v_curr_state_noise = add_noise(v_curr_state, self.sigma_scaled, self.u_v_scaled)

            if self.sobolev_grad:
                v_curr_state_noise_smoothed = \
                    SobolevGrad.apply(v_curr_state_noise, self.S_x, self.S_y, self.S_z, self.padding_sz)
                transformation, displacement = self.transformation_model(v_curr_state_noise_smoothed)
                reg_term = self.reg_loss(v_curr_state_noise_smoothed).sum()
            else:
                transformation, displacement = self.transformation_model(v_curr_state_noise)
                reg_term = self.reg_loss(v_curr_state_noise).sum()

            im_moving_warped = self.registration_module(im_moving, transformation)
            data_term = self.data_loss(self.im_fixed, im_moving_warped, self.mask_fixed).sum()

            loss = data_term + reg_term
            loss.backward()
            self.optimizer_mala.step()

            """
            Metropolis-Hastings
            """

            with torch.no_grad():
                log_pi_prev = loss.item()
                log_pi_next = 0.0
                
                v_next_state = v_curr_state.clone()
                v_next_state_noise = add_noise(v_next_state, self.sigma_scaled, self.u_v_scaled)

                if self.sobolev_grad:
                    v_next_state_noise_smoothed = \
                        SobolevGrad.apply(v_next_state_noise, self.S_x, self.S_y, self.S_z, self.padding_sz)
                    transformation_next, displacement_next = self.transformation_model(v_next_state_noise_smoothed)
                    log_pi_next += self.reg_loss(v_next_state_noise_smoothed).sum().item()
                else:
                    transformation_next, displacement_next = self.transformation_model(v_next_state_noise)
                    log_pi_next += self.reg_loss(v_next_state_noise).sum().item()

                im_moving_warped_next = self.registration_module(im_moving, transformation_next)
                log_pi_next += self.data_loss(self.im_fixed, im_moving_warped_next, self.mask_fixed).sum().item()

                log_alpha = log_pi_next - log_pi_prev
                log_u = torch.log(torch.rand(1))

                if log_u > log_alpha:  # reject the sample
                    self.no_samples_rejected += 1.0
                    v_curr_state = v_prev_state
                else:
                    self.no_samples_accepted += 1.0

            # metrics and prints
            if sample_no == self.no_steps_burn_in - 1:
                self.logger.info('\nENDED BURNING IN\n')

            if sample_no % 50 == 0:
                r = self.no_samples_accepted / (self.no_samples_accepted + self.no_samples_rejected)
                self.logger.info(f'acceptance rate: {r:.5f}')

            if sample_no >= self.no_steps_burn_in:
                with torch.no_grad():
                    self.writer.set_step(sample_no)

                    self.train_metrics.update('sample_data_term', data_term.item())
                    self.train_metrics.update('sample_reg_term', reg_term.item())

                    mixing_print(self.logger, sample_no, self.no_samples,
                                 loss.item(), data_term.item(), reg_term.item())

                    # # log to tensorboard and save images, fields etc.
                    # nabla_x, nabla_y, nabla_z = get_module_attr(self.reg_loss, 'diff_op')(transformation)
                    # det_J_transformation = calc_det_J(nabla_x, nabla_y, nabla_z)
                    # log_det_J_transformation = torch.log10(det_J_transformation)

                    # log_images(self.writer, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped)
                    # log_v(self.writer, im_pair_idxs, v_curr_state, displacement)
                    # log_log_det_J_transformation(self.writer, im_pair_idxs, log_det_J_transformation)

    def _train_epoch(self, epoch):
        self.train_metrics.reset()

        for batch_idx, (im_pair_idxs, im_fixed, mask_fixed, im_moving, mu_v, log_var_v, u_v) \
                in enumerate(self.data_loader):
            if self.im_fixed is None:
                self.im_fixed = im_fixed.to(self.device, non_blocking=True)
            if self.mask_fixed is None:
                self.mask_fixed = mask_fixed.to(self.device, non_blocking=True)

            im_moving = im_moving.to(self.device, non_blocking=True)

            mu_v = mu_v.to(self.device, non_blocking=True)
            log_var_v = log_var_v.to(self.device, non_blocking=True)
            u_v = u_v.to(self.device, non_blocking=True)

            # print value of the data term before registration
            with torch.no_grad():
                if self.sobolev_grad:
                    v_smoothed = SobolevGrad.apply(mu_v, self.S_x, self.S_y, self.S_z, self.padding_sz)
                    transformation, displacement = self.transformation_model(v_smoothed)
                else:
                    transformation, displacement = self.transformation_model(mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)

                loss_unwarped = self.data_loss(self.im_fixed, im_moving, self.mask_fixed).sum()
                loss_warped = self.data_loss(self.im_fixed, im_moving_warped, self.mask_fixed).sum()

                self.logger.info(f'\nPRE-REGISTRATION: ' +
                                 f'unwarped: {loss_unwarped:.5f}' +
                                 f', warped w/ the current state: {loss_warped:.5f}\n'
                                 )

            """
            variational inference
            """

            self._step_VI(im_pair_idxs, im_moving, mu_v, log_var_v, u_v)

            self.tau = self.config['optimizer_mala']['args']['lr']
            self.sqrt_tau_twice = np.sqrt(2.0 * self.tau)

            self.mu_v = mu_v
            self.sigma_scaled = self.sqrt_tau_twice * transform_coordinates(torch.exp(0.5 * log_var_v))
            self.u_v_scaled = self.sqrt_tau_twice * transform_coordinates(u_v)
            
            """
            sampling from the posterior
            """

            self._step_MCMC(im_pair_idxs, im_moving)

        return self.train_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'

        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch

        return base.format(current, total, 100.0 * current / total)
