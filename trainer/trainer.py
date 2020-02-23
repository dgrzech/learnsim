from base import BaseTrainer
from logger import log_fields, log_hist_res, log_images, log_sample, print_log, \
    save_fields, save_grids, save_images, save_norms, save_sample
from optimizers import Adam
from utils import add_noise, add_noise_uniform, calc_det_J, get_module_attr, inf_loop, max_field_update, \
    rescale_residuals, sample_q_v, sobolev_kernel_1d, transform_coordinates, vd, MetricTracker, SobolevGrad

import math
import numpy as np
import torch


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, data_loss, scale_prior, proportion_prior, reg_loss, reg_scale_loss_prior, entropy_loss,
                 transformation_model, registration_module, metric_ftns_vi, metric_ftns_mcmc, config, data_loader):
        super().__init__(data_loss, scale_prior, proportion_prior, reg_loss, reg_scale_loss_prior, entropy_loss,
                         transformation_model, registration_module, config)

        self.config = config
        self.data_loader = data_loader
        self.im_fixed, self.seg_fixed, self.mask_fixed = None, None, None

        # variational inference
        self.start_iter = 1
        self.mu_v, self.log_var_v, self.u_v = None, None, None

        self.VI = config['trainer']['vi']
        self.optimizer_mixture_model, self.optimizer_v = None, None
        self.train_metrics_vi = MetricTracker(*[m for m in metric_ftns_vi], writer=self.writer)

        # Markov chain Monte Carlo
        self.start_sample = 1
        self.v_curr_state, self.sigma_scaled, self.u_scaled = None, None, None

        self.MCMC = config['trainer']['mcmc']
        self.optimizer_mala = None
        self.train_metrics_mcmc = MetricTracker(*[m for m in metric_ftns_mcmc], writer=self.writer)

        # Sobolev gradients
        self.sobolev_grad = config['sobolev_grad']['enabled']

        if self.sobolev_grad:
            _s = config['sobolev_grad']['s']
            _lambda = config['sobolev_grad']['lambda']
            self.padding_sz = _s // 2

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

        # virtual decimation
        self.vd = config['vd']
        self.vd_reg = config['vd_reg']

        # resuming
        if config.resume is not None and self.VI:
            self._resume_checkpoint_vi(config.resume)
        elif config.resume is not None and self.MCMC:
            self._resume_checkpoint_mcmc(config.resume)

        self.use_moving_mask = config['use_moving_mask']  # whether to use the moving mask

    def _step_GMM(self, res, alpha=1.0):
        if self.optimizer_mixture_model is None:  # initialise the optimiser
            self.optimizer_mixture_model = Adam([{'params': [self.data_loss.log_std], 'lr': 1e-1},
                                                 {'params': [self.data_loss.logits], 'lr': 1e-2}],
                                                lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)

        data_term = self.data_loss(res.detach()) * alpha
        data_term -= torch.sum(self.scale_prior(self.data_loss.log_scales()))
        data_term -= torch.sum(self.proportion_prior(self.data_loss.log_proportions()))

        self.optimizer_mixture_model.zero_grad()
        data_term.backward()
        self.optimizer_mixture_model.step()  # backprop

    def _step_VI(self, im_pair_idxs, im_moving, mask_moving):
        self.mu_v.requires_grad_(True)
        self.log_var_v.requires_grad_(True)
        self.u_v.requires_grad_(True)

        if self.optimizer_v is None:
            if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
                self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim,
                                                        [self.mu_v, self.log_var_v, self.u_v, self.reg_loss.log_w_reg])
            else:
                self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim,
                                                        [self.mu_v, self.log_var_v, self.u_v])

        for iter_no in range(self.start_iter, self.no_iters_vi + 1):
            self.train_metrics_vi.reset()

            if iter_no % self.log_period == 0 or iter_no == self.no_iters_vi:
                mu_v_old = self.mu_v.detach().clone()  # needed to calculate the maximum update in terms of the L2 norm
                log_var_v_old = self.log_var_v.detach().clone()
                u_v_old = self.u_v.detach().clone()

            v_sample1, v_sample2 = sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=2)
            if self.sobolev_grad:
                v_sample1 = SobolevGrad.apply(v_sample1, self.S_x, self.S_y, self.S_z, self.padding_sz)
                v_sample2 = SobolevGrad.apply(v_sample2, self.S_x, self.S_y, self.S_z, self.padding_sz)

            transformation1, displacement1 = self.transformation_model(v_sample1)
            transformation2, displacement2 = self.transformation_model(v_sample2)

            # add noise to account for interpolation uncertainty
            transformation1, displacement1 = add_noise_uniform(transformation1), add_noise_uniform(displacement1)
            transformation2, displacement2 = add_noise_uniform(transformation2), add_noise_uniform(displacement2)

            im_moving_warped1, im_moving_warped2 = self.registration_module(im_moving, transformation1), \
                                                   self.registration_module(im_moving, transformation2)
            
            n_F, n_M1 = self.data_loss.map(self.im_fixed, im_moving_warped1)
            n_F, n_M2 = self.data_loss.map(self.im_fixed, im_moving_warped2)

            res1, res2 = n_F - n_M1, n_F - n_M2

            alpha1, alpha2 = 1.0, 1.0
            alpha_mean = 1.0

            if self.vd:  # virtual decimation
                # rescale the residuals by the estimated voxel-wise standard deviation
                if self.use_moving_mask:
                    mask_moving_warped1, mask_moving_warped2 = self.registration_module(mask_moving.float(), transformation1, mode='nearest').bool(), \
                                                               self.registration_module(mask_moving.float(), transformation2, mode='nearest').bool()
                    mask1, mask2 = (self.mask_fixed * mask_moving_warped1), (self.mask_fixed * mask_moving_warped2)

                    res1_rescaled = rescale_residuals(res1.detach(), mask1, self.data_loss)
                    res2_rescaled = rescale_residuals(res2.detach(), mask2, self.data_loss)

                    with torch.no_grad():
                        alpha1, alpha2 = vd(res1_rescaled, mask1), vd(res2_rescaled, mask2)
                        alpha_mean = (alpha1.item() + alpha2.item()) / 2.0
                    
                    res1_masked, res2_masked = res1[mask1], res2[mask2]
                else:
                    res1_rescaled = rescale_residuals(res1.detach(), self.mask_fixed, self.data_loss)
                    res2_rescaled = rescale_residuals(res2.detach(), self.mask_fixed, self.data_loss)

                    with torch.no_grad():
                        alpha1, alpha2 = vd(res1_rescaled, self.mask_fixed), vd(res2_rescaled, self.mask_fixed)
                        alpha_mean = (alpha1.item() + alpha2.item()) / 2.0
                    
                    res1_masked, res2_masked = res1[self.mask_fixed], res2[self.mask_fixed]

            # Gaussian mixture
            self._step_GMM(res1_masked, alpha1)

            # q_v
            data_term = self.data_loss(res1_masked) / 2.0 * alpha1
            data_term += self.data_loss(res2_masked) / 2.0 * alpha2
            data_term -= torch.sum(self.scale_prior(self.data_loss.log_scales()))
            data_term -= torch.sum(self.proportion_prior(self.data_loss.log_proportions()))

            reg_term1, alpha_reg1 = self.reg_loss(v_sample1, self.vd_reg)
            reg_term2, alpha_reg2 = self.reg_loss(v_sample2, self.vd_reg)

            reg_term = reg_term1.sum() / 2.0 + reg_term2.sum() / 2.0
            alpha_reg_mean = (alpha_reg1 + alpha_reg2) / 2.0

            if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
                reg_term -= 3.0 * torch.sum(self.reg_loss_scale_prior(self.reg_loss.log_scales()))  # prior

            entropy_term = self.entropy_loss(v_sample=v_sample1,
                                             mu_v=self.mu_v, log_var_v=self.log_var_v, u_v=self.u_v).sum() / 2.0
            entropy_term += self.entropy_loss(v_sample=v_sample2,
                                              mu_v=self.mu_v, log_var_v=self.log_var_v, u_v=self.u_v).sum() / 2.0
            entropy_term += self.entropy_loss(log_var_v=self.log_var_v, u_v=self.u_v).sum()

            loss_q_v = data_term + reg_term - entropy_term

            self.optimizer_v.zero_grad()
            loss_q_v.backward()  # backprop
            self.optimizer_v.step()

            """
            metrics and prints
            """

            self.writer.set_step(iter_no)

            self.train_metrics_vi.update('VI/data_term', data_term.item())
            self.train_metrics_vi.update('VI/reg_term', reg_term.item())
            self.train_metrics_vi.update('VI/entropy_term', entropy_term.item())
            self.train_metrics_vi.update('VI/total_loss', loss_q_v.item())

            self.train_metrics_vi.update('other/alpha', alpha_mean)

            if self.vd_reg:
                self.train_metrics_vi.update('other/alpha_reg', alpha_reg_mean)

            if iter_no % self.log_period == 0 or iter_no == self.no_iters_vi:
                with torch.no_grad():
                    # FIXME: should probably do this over the masked region
                    sigmas = torch.exp(self.data_loss.log_scales())
                    proportions = torch.exp(self.data_loss.log_proportions())

                    max_update_mu_v, max_update_mu_v_idx = max_field_update(mu_v_old, self.mu_v)
                    max_update_log_var_v, max_update_log_var_v_idx = max_field_update(log_var_v_old, self.log_var_v)
                    max_update_u_v, max_update_u_v_idx = max_field_update(u_v_old, self.u_v)

                if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
                    with torch.no_grad():
                       mu_w_reg = torch.mean(torch.exp(self.reg_loss.log_scales()))
                       self.train_metrics_vi.update('other/mean_w_reg', mu_w_reg.item())

                for idx in range(self.data_loss.num_components):
                    self.train_metrics_vi.update('GM/sigma_' + str(idx), sigmas[idx])
                    self.train_metrics_vi.update('GM/proportion_' + str(idx), proportions[idx])

                self.train_metrics_vi.update('other/max_updates/mu_v', max_update_mu_v.item())
                self.train_metrics_vi.update('other/max_updates/log_var_v', max_update_log_var_v.item())
                self.train_metrics_vi.update('other/max_updates/u_v', max_update_u_v.item())

                log = {'iter_no': iter_no}
                log.update(self.train_metrics_vi.result())
                print_log(self.logger, log)

            """
            outputs
            """

            if math.log2(iter_no).is_integer() or iter_no == self.no_iters_vi:
                with torch.no_grad():
                    if self.sobolev_grad:
                        mu_v_smoothed = \
                            SobolevGrad.apply(self.mu_v, self.S_x, self.S_y, self.S_z, self.padding_sz)
                        log_var_v_smoothed = \
                            SobolevGrad.apply(self.log_var_v, self.S_x, self.S_y, self.S_z, self.padding_sz)
                        u_v_smoothed = \
                            SobolevGrad.apply(self.u_v, self.S_x, self.S_y, self.S_z, self.padding_sz)

                        var_params = {'mu_v': mu_v_smoothed, 'log_var_v': log_var_v_smoothed, 'u_v': u_v_smoothed}
                        transformation, displacement = self.transformation_model(mu_v_smoothed)
                    else:
                        var_params = {'mu_v': self.mu_v, 'log_var_v': self.log_var_v, 'u_v': self.u_v}
                        transformation, displacement = self.transformation_model(self.mu_v)

                    im_moving_warped = self.registration_module(im_moving, transformation)
                    nabla_x, nabla_y, nabla_z = get_module_attr(self.reg_loss, 'diff_op')(transformation)
                    log_det_J_transformation = torch.log(calc_det_J(nabla_x, nabla_y, nabla_z))

                    # tensorboard
                    if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
                        log_fields(self.writer, im_pair_idxs, var_params, displacement, log_det_J_transformation,
                                   self.reg_loss.log_scales())
                    else:
                        log_fields(self.writer, im_pair_idxs, var_params, displacement, log_det_J_transformation)

                    log_images(self.writer, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped)
                    log_hist_res(self.writer, im_pair_idxs, res1_masked, self.data_loss)

                    # .nii.gz/.vtk
                    save_fields(
                        self.data_loader.save_dirs, im_pair_idxs, var_params, displacement, log_det_J_transformation)
                    save_grids(self.data_loader.save_dirs, im_pair_idxs, transformation)
                    save_images(self.data_loader.save_dirs, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped)
                    save_norms(self.data_loader.save_dirs, im_pair_idxs, var_params, displacement)

            # checkpoint
            if iter_no % self.save_period == 0 or iter_no == self.no_iters_vi:
                self._save_checkpoint_vi(iter_no)

    def _step_MCMC(self, im_pair_idxs, im_moving, mask_moving):
        self.mu_v.requires_grad_(False)
        self.log_var_v.requires_grad_(False)
        self.u_v.requires_grad_(False)

        self.v_curr_state.requires_grad_(True)

        if self.optimizer_mala is None:
            self.optimizer_mala = self.config.init_obj('optimizer_mala', torch.optim, [self.v_curr_state])

        self.logger.info('\nBURNING IN THE MARKOV CHAIN\n')

        for sample_no in range(self.start_sample, self.no_samples + 1):
            self.train_metrics_mcmc.reset()

            if sample_no < self.no_iters_burn_in and sample_no % 2000 == 0:
                self.logger.info('burn-in sample no. ' + str(sample_no) + '/' + str(self.no_iters_burn_in))
            
            """
            stochastic gradient Langevin dynamics
            """

            v_curr_state_noise = add_noise(self.v_curr_state, self.sigma_scaled, self.u_v_scaled)

            if self.sobolev_grad:
                v_curr_state_noise_smoothed = \
                    SobolevGrad.apply(v_curr_state_noise, self.S_x, self.S_y, self.S_z, self.padding_sz)
                transformation, displacement = self.transformation_model(v_curr_state_noise_smoothed)
                reg_term, alpha_reg = self.reg_loss(v_curr_state_noise_smoothed, self.vd_reg)

                if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
                    reg_term -= 3.0 * torch.sum(self.reg_loss_scale_prior(self.reg_loss.log_scales()))
            else:
                transformation, displacement = self.transformation_model(v_curr_state_noise)
                reg_term, alpha_reg = self.reg_loss(v_curr_state_noise, self.vd_reg)

                if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
                    reg_term -= 3.0 * torch.sum(self.reg_loss_scale_prior(self.reg_loss.log_scales()))

            transformation, displacement = add_noise_uniform(transformation), add_noise_uniform(displacement)
            im_moving_warped = self.registration_module(im_moving, transformation)

            n_F, n_M = self.data_loss.map(self.im_fixed, im_moving_warped)
            res = n_F - n_M

            if self.use_moving_mask:
                mask_moving_warped = self.registration_module(mask_moving.float(), transformation, mode='nearest').bool()
                mask = (self.mask_fixed * mask_moving_warped)

            alpha = 1.0
            alpha_mean = alpha

            if self.vd:  # virtual decimation
                if self.use_moving_mask:
                    res_rescaled = rescale_residuals(res.detach(), mask, self.data_loss)
                    res_masked = res[mask]

                    with torch.no_grad():
                        alpha = vd(res_rescaled, mask)
                        alpha_mean = alpha.item()
                else:
                    res_rescaled = rescale_residuals(res.detach(), self.mask_fixed, self.data_loss)
                    res_masked = res[self.mask_fixed]
                    
                    with torch.no_grad():
                        alpha = vd(res_rescaled, self.mask_fixed)
                        alpha_mean = alpha.item()

            # Gaussian mixture
            self._step_GMM(res_masked, alpha)

            # MCMC
            data_term = self.data_loss(res_masked) * alpha
            data_term -= torch.sum(self.scale_prior(self.data_loss.log_scales()))
            data_term -= torch.sum(self.proportion_prior(self.data_loss.log_proportions()))

            loss = data_term + reg_term.sum()

            self.optimizer_mala.zero_grad()
            loss.backward()  # backprop
            self.optimizer_mala.step()

            """
            metrics and prints
            """

            self.writer.set_step(sample_no)

            self.train_metrics_mcmc.update('MCMC/data_term', data_term.item())
            self.train_metrics_mcmc.update('MCMC/reg_term', reg_term.item())
            self.train_metrics_mcmc.update('other/alpha', alpha_mean)

            if self.vd_reg:
                self.train_metrics_mcmc.update('other/alpha_reg', alpha_reg.item())

            if sample_no == self.no_iters_burn_in:
                self.logger.info('\nENDED BURNING IN\n')

            # tensorboard
            if sample_no > self.no_iters_burn_in and sample_no % self.log_period == 0:
                with torch.no_grad():
                    log = {'sample_no': sample_no}
                    log.update(self.train_metrics_mcmc.result())
                    print_log(self.logger, log)

                    if self.sobolev_grad:
                        log_sample(self.writer, im_pair_idxs, self.data_loss,
                                   im_moving_warped, res_masked, v_curr_state_noise_smoothed, displacement)
                    else:
                        log_sample(self.writer, im_pair_idxs, self.data_loss,
                                   im_moving_warped, res_masked, self.v_curr_state, displacement)
            
            """
            outputs
            """

            if sample_no % self.save_period == 0 or sample_no == self.no_samples:
                with torch.no_grad():
                    if self.sobolev_grad:
                        save_sample(self.data_loader.save_dirs, im_pair_idxs,
                                    sample_no, im_moving_warped, v_curr_state_noise_smoothed)
                    else:
                        save_sample(self.data_loader.save_dirs, im_pair_idxs,
                                    sample_no, im_moving_warped, v_curr_state_noise)

                    self._save_checkpoint_mcmc(sample_no)  # checkpoint

    def _train_epoch(self):
        for batch_idx, (im_pair_idxs, im_fixed, mask_fixed, im_moving, mask_moving, mu_v, log_var_v, u_v) \
                in enumerate(self.data_loader):
            if self.im_fixed is None:
                self.im_fixed = im_fixed.to(self.device, non_blocking=True)
            if self.mask_fixed is None:
                self.mask_fixed = mask_fixed.to(self.device, non_blocking=True)

            im_moving = im_moving.to(self.device, non_blocking=True)

            if mask_moving is not None:
                mask_moving = mask_moving.to(self.device, non_blocking=True)

            if self.mu_v is None:
                self.mu_v = mu_v.to(self.device, non_blocking=True)
            if self.log_var_v is None:
                self.log_var_v = log_var_v.to(self.device, non_blocking=True)
            if self.u_v is None:
                self.u_v = u_v.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                v_sample = sample_q_v(self.mu_v, self.log_var_v, self.u_v)
                if self.sobolev_grad:
                    v_sample = SobolevGrad.apply(v_sample, self.S_x, self.S_y, self.S_z, self.padding_sz)

                transformation, displacement = self.transformation_model(v_sample)
                transformation = add_noise_uniform(transformation)

                im_moving_warped = self.registration_module(im_moving, transformation)
                n_F, n_M = self.data_loss.map(self.im_fixed, im_moving_warped)
                res = n_F - n_M

                if self.use_moving_mask:
                    mask_moving_warped = self.registration_module(mask_moving.float(), transformation, mode='nearest').bool()
                    mask = (self.mask_fixed * mask_moving_warped)
                    res_masked = res[mask]
                else:
                    res_masked = res[self.mask_fixed]

                res_mean = torch.mean(res_masked)
                res_var = torch.mean(torch.pow(res_masked - res_mean, 2))
                res_std = torch.sqrt(res_var)

                self.data_loss.init_parameters(res_std)
                alpha = 1.0

            if self.vd:  # virtual decimation
                if self.use_moving_mask:
                    res_rescaled = rescale_residuals(res, mask, self.data_loss)
                    alpha = vd(res_rescaled, mask)  # virtual decimation
                else:
                    res_rescaled = rescale_residuals(res, self.mask_fixed, self.data_loss)
                    alpha = vd(res_rescaled, self.mask_fixed)

            # Gaussian mixture
            self._step_GMM(res_masked, alpha)

            # print value of the data term before registration
            with torch.no_grad():
                loss_unwarped = self.data_loss(res_masked) * alpha
                self.logger.info(f'PRE-REGISTRATION: {loss_unwarped.item():.5f}\n')

                iter_no = 0
                self.writer.set_step(iter_no)
                
                self.train_metrics_vi.update('VI/data_term', loss_unwarped.item())
                log_hist_res(self.writer, im_pair_idxs, res_masked, self.data_loss)

            """
            VI
            """

            if self.VI:
                self._step_VI(im_pair_idxs, im_moving, mask_moving)

            """
            MCMC
            """

            if self.MCMC:
                with torch.no_grad():
                    tau = self.config['optimizer_mala']['args']['lr']
                    sqrt_tau_twice = np.sqrt(2.0 * tau)

                    self.sigma_scaled = sqrt_tau_twice * transform_coordinates(torch.exp(0.5 * self.log_var_v))
                    self.u_v_scaled = sqrt_tau_twice * transform_coordinates(self.u_v)

                    if self.v_curr_state is None:
                        self.v_curr_state = self.mu_v.clone()

                self._step_MCMC(im_pair_idxs, im_moving, mask_moving)

    def _save_checkpoint_vi(self, iter_no):
        """
        save a checkpoint (variational inference)
        """

        if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
            state = {
                'config': self.config,
                'iter_no': iter_no,
                'sample_no': self.start_sample,

                'mu_v': self.mu_v,
                'log_var_v': self.log_var_v,
                'u_v': self.u_v,
                'optimizer_v': self.optimizer_v.state_dict(),

                'data_loss': self.data_loss.state_dict(),
                'reg_loss': self.reg_loss.state_dict(),
                'optimizer_mixture_model': self.optimizer_mixture_model.state_dict(),
            }
        else:
            state = {
                'config': self.config,
                'iter_no': iter_no,
                'sample_no': self.start_sample,

                'mu_v': self.mu_v,
                'log_var_v': self.log_var_v,
                'u_v': self.u_v,
                'optimizer_v': self.optimizer_v.state_dict(),

                'data_loss': self.data_loss.state_dict(),
                'optimizer_mixture_model': self.optimizer_mixture_model.state_dict(),
            }

        filename = str(self.checkpoint_dir / f'checkpoint_vi_{iter_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _save_checkpoint_mcmc(self, sample_no):
        """
        save a checkpoint (Markov chain Monte Carlo)
        """

        if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
            state = {
                'config': self.config,
                'iter_no': self.no_iters_vi,
                'sample_no': sample_no,

                'mu_v': self.mu_v,
                'log_var_v': self.log_var_v,
                'u_v': self.u_v,

                'data_loss': self.data_loss.state_dict(),
                'reg_loss': self.reg_loss.state_dict(),
                'optimizer_mixture_model': self.optimizer_mixture_model.state_dict(),

                'v_curr_state': self.v_curr_state,
                'optimizer_mala': self.optimizer_mala.state_dict()
            }
        else:
            state = {
                'config': self.config,
                'iter_no': self.no_iters_vi,
                'sample_no': sample_no,

                'mu_v': self.mu_v,
                'log_var_v': self.log_var_v,
                'u_v': self.u_v,

                'data_loss': self.data_loss.state_dict(),
                'optimizer_mixture_model': self.optimizer_mixture_model.state_dict(),

                'v_curr_state': self.v_curr_state,
                'optimizer_mala': self.optimizer_mala.state_dict()
            }

        filename = str(self.checkpoint_dir / f'checkpoint_mcmc_{sample_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _resume_checkpoint_vi(self, resume_path):
        """
        resume from saved checkpoints (VI)
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {}..".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_iter = checkpoint['iter_no'] + 1
        self.start_sample = checkpoint['sample_no'] + 1

        # reg. loss
        if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
            self.reg_loss.load_state_dict(checkpoint['reg_loss'])

        # VI
        self.mu_v = checkpoint['mu_v']
        self.log_var_v = checkpoint['log_var_v']
        self.u_v = checkpoint['u_v']

        if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
            self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim,
                                                    [self.mu_v, self.log_var_v, self.u_v, self.reg_loss.log_w_reg])
        else:
            self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim,
                                                    [self.mu_v, self.log_var_v, self.u_v])

        self.optimizer_v.load_state_dict(checkpoint['optimizer_v'])

        # GMM
        self.data_loss.load_state_dict(checkpoint['data_loss'])

        self.optimizer_mixture_model = Adam([{'params': [self.data_loss.log_std], 'lr': 1e-1},
                                             {'params': [self.data_loss.logits], 'lr': 1e-2}],
                                            lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)
        self.optimizer_mixture_model.load_state_dict(checkpoint['optimizer_mixture_model'])

        self.logger.info("checkpoint loaded, resuming training..")

    def _resume_checkpoint_mcmc(self, resume_path):
        """
        resume from saved checkpoints (MCMC)
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {}..".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_iter = checkpoint['iter_no'] + 1
        self.start_sample = checkpoint['sample_no'] + 1

        # regularisation loss
        if type(self.reg_loss).__name__ == 'RegLossL2_Learnable':
            self.reg_loss.load_state_dict(checkpoint['reg_loss'])

        # VI
        self.mu_v = checkpoint['mu_v']
        self.log_var_v = checkpoint['log_var_v']
        self.u_v = checkpoint['u_v']

        # GMM
        self.data_loss.load_state_dict(checkpoint['data_loss'])

        self.optimizer_mixture_model = Adam([{'params': [self.data_loss.log_std], 'lr': 1e-1},
                                             {'params': [self.data_loss.logits], 'lr': 1e-2}],
                                            lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)
        self.optimizer_mixture_model.load_state_dict(checkpoint['optimizer_mixture_model'])

        # MCMC
        with torch.no_grad():
            self.v_curr_state = checkpoint['v_curr_state'] if 'v_curr_state' in checkpoint else self.mu_v.clone()

        if 'optimizer_mala' in checkpoint:
            self.optimizer_mala = self.config.init_obj('optimizer_mala', torch.optim, [self.v_curr_state])
            self.optimizer_mala.load_state_dict(checkpoint['optimizer_mala'])

        self.logger.info("checkpoint loaded, resuming training..")
