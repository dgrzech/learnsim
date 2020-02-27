from base import BaseTrainer
from logger import log_fields, log_hist_res, log_images, log_omega_norm, log_sample, print_log, \
    save_fields, save_grids, save_images, save_norms, save_sample
from optimizers import Adam
from utils import calc_asd, calc_det_J, calc_dice, get_module_attr, get_omega_norm_sq, inf_loop, \
    max_field_update, rescale_residuals, sample_q, sobolev_kernel_1d, transform_coordinates, vd, \
    GradientOperator, MetricTracker

from torch import nn

import math
import numpy as np
import pprint
import torch


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, data_loss, scale_prior, proportion_prior, reg_loss, reg_loss_prior_loc, reg_loss_prior_scale,
                 entropy_loss, transformation_model, registration_module,
                 metric_ftns_vi, metric_ftns_mcmc, structures_dict, config, data_loader):
        super().__init__(data_loss, scale_prior, proportion_prior, reg_loss, reg_loss_prior_loc, reg_loss_prior_scale,
                         entropy_loss, transformation_model, registration_module, config)

        self.config = config
        self.data_loader = data_loader
        self.im_fixed, self.mask_fixed, self.seg_fixed = None, None, None

        dim_x = config['data_loader']['args']['dim_x']
        dim_y = config['data_loader']['args']['dim_y']
        dim_z = config['data_loader']['args']['dim_z']

        self.dof = dim_x * dim_y * dim_z * 3.0

        # variational inference
        self.start_iter = 1
        self.mu_hat, self.log_var_hat, self.u_hat = None, None, None

        self.VI = config['trainer']['vi']
        self.optimizer_mixture_model, self.optimizer_w_reg, self.optimizer_v = None, None, None
        self.train_metrics_vi = MetricTracker(*[m for m in metric_ftns_vi], writer=self.writer)

        # Markov chain Monte Carlo
        self.start_sample = 1
        self.tau = None
        self.v_curr_state, self.sigma_scaled, self.u_scaled = None, None, None

        self.MCMC = config['trainer']['mcmc']
        self.optimizer_mala = None
        self.train_metrics_mcmc = MetricTracker(*[m for m in metric_ftns_mcmc], writer=self.writer)

        # segmentations
        self.structures_dict = structures_dict

        # whether to use the moving mask
        self.use_moving_mask = config['use_moving_mask']

        # virtual decimation
        self.vd = config['vd']

        # resuming
        if config.resume is not None and self.VI:
            self._resume_checkpoint_vi(config.resume)
        elif config.resume is not None and self.MCMC:
            self._resume_checkpoint_mcmc(config.resume)

        # frequency domain
        c = 2.0

        omega_norm_sq = get_omega_norm_sq((dim_x, dim_y, dim_z))
        omega_norm_inv = math.sqrt(c) / torch.sqrt(omega_norm_sq + c)
        self.omega_norm_inv = nn.Parameter(omega_norm_inv, requires_grad=False).to(self.device)

    def _step_GMM(self, res, alpha=1.0):
        if self.optimizer_mixture_model is None:  # initialise the optimiser
            self.optimizer_mixture_model = Adam([{'params': [self.data_loss.log_std], 'lr': 1e-1},
                                                 {'params': [self.data_loss.logits], 'lr': 1e-2}],
                                                lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)

        data_term = self.data_loss(res.detach()).sum() * alpha
        data_term -= self.scale_prior(self.data_loss.log_scales()).sum()
        data_term -= self.proportion_prior(self.data_loss.log_proportions()).sum()

        self.optimizer_mixture_model.zero_grad()
        data_term.backward()  # backprop
        self.optimizer_mixture_model.step()

    def _step_VI(self, im_pair_idxs, im_moving, mask_moving, seg_moving):
        self.mu_hat.requires_grad_(True)
        self.log_var_hat.requires_grad_(True)
        self.u_hat.requires_grad_(True)

        if self.optimizer_v is None:
            self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim,
                                                    [self.mu_hat])

        if self.reg_loss.__class__.__name__ is not 'RegLoss_L2':
            if self.optimizer_w_reg is None:
                self.optimizer_w_reg = Adam([{'params': [self.reg_loss.loc, self.reg_loss.log_scale]}],
                                            lr=1e-1, betas=(0.9, 0.95))

        for iter_no in range(self.start_iter, self.no_iters_vi + 1):
            self.train_metrics_vi.reset()

            z1_hat, z2_hat = sample_q(self.mu_hat, self.log_var_hat, self.u_hat, 
                                      omega_norm_inv=self.omega_norm_inv, no_samples=2)

            transformation1, displacement1 = self.transformation_model(z1_hat)
            transformation2, displacement2 = self.transformation_model(z2_hat)

            im_moving_warped1, im_moving_warped2 = self.registration_module(im_moving, transformation1), \
                                                   self.registration_module(im_moving, transformation2)
            
            n_F, n_M1 = self.data_loss.map(self.im_fixed, im_moving_warped1)
            n_F, n_M2 = self.data_loss.map(self.im_fixed, im_moving_warped2)
            res1, res2 = n_F - n_M1, n_F - n_M2

            alpha1, alpha2 = 1.0, 1.0
            alpha_mean = 1.0

            # virtual decimation
            if self.vd:
                if self.use_moving_mask:
                    mask_moving_warped1, \
                    mask_moving_warped2 = self.registration_module(mask_moving, transformation1), \
                                          self.registration_module(mask_moving, transformation2)

                    mask1, mask2 = (self.mask_fixed * mask_moving_warped1), (self.mask_fixed * mask_moving_warped2)

                    # rescale the residuals by the estimated voxel-wise standard deviation
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
            data_term = self.data_loss(res1_masked).sum() / 2.0 * alpha1
            data_term += self.data_loss(res2_masked).sum() / 2.0 * alpha2

            data_term -= self.scale_prior(self.data_loss.log_scales()).sum()
            data_term -= self.proportion_prior(self.data_loss.log_proportions()).sum()
            
            reg_term1, log_y1 = self.reg_loss(z1_hat)
            reg_term2, log_y2 = self.reg_loss(z2_hat)

            reg_term = reg_term1.sum() / 2.0 + reg_term2.sum() / 2.0

            entropy_term = \
                self.entropy_loss(z_sample=z1_hat, mu=self.mu_hat, log_var=self.log_var_hat, u=self.u_hat).sum() / 2.0
            entropy_term += \
                self.entropy_loss(z_sample=z2_hat, mu=self.mu_hat, log_var=self.log_var_hat, u=self.u_hat).sum() / 2.0
            entropy_term += self.entropy_loss(log_var=self.log_var_hat, u=self.u_hat).sum()

            loss_q_v = data_term + reg_term  # - entropy_term

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
            
            self.train_metrics_vi.update('other/VI/alpha', alpha_mean)

            if iter_no % self.log_period == 0 or iter_no == self.no_iters_vi:
                with torch.no_grad():
                    sigmas = torch.exp(self.data_loss.log_scales())
                    proportions = torch.exp(self.data_loss.log_proportions())

                for idx in range(self.data_loss.num_components):
                    self.train_metrics_vi.update('GM/VI/sigma_' + str(idx), sigmas[idx])
                    self.train_metrics_vi.update('GM/VI/proportion_' + str(idx), proportions[idx])

                # dice scores
                seg_moving_warped = self.registration_module(seg_moving, transformation1)
                dsc = calc_dice(self.seg_fixed, seg_moving_warped, self.structures_dict)

                for structure in dsc:
                    score = dsc[structure]
                    self.train_metrics_vi.update('DSC/VI/' + structure, score)
                
                asd = calc_asd(self.seg_fixed, seg_moving_warped, self.structures_dict, self.data_loader.dataset.spacing)

                for structure in asd:
                    dist = asd[structure]
                    self.train_metrics_vi.update('ASD/VI/' + structure, dist)

                log = {'iter_no': iter_no}
                log.update(self.train_metrics_vi.result())
                print_log(self.logger, log)

            """
            outputs
            """

            if math.log2(iter_no).is_integer() or iter_no == self.no_iters_vi:
                with torch.no_grad():
                    nabla_v = GradientOperator()(transformation1)
                    nabla_x, nabla_y, nabla_z = nabla_v[:, :, :, :, :, 0], \
                                                nabla_v[:, :, :, :, :, 1], \
                                                nabla_v[:, :, :, :, :, 2]
                    log_det_J_transformation = torch.log(calc_det_J(nabla_x, nabla_y, nabla_z))

                    # tensorboard
                    log_fields(self.writer, im_pair_idxs, displacement1, log_det_J_transformation)
                    log_images(self.writer, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped1)
                    log_hist_res(self.writer, im_pair_idxs, res1_masked, self.data_loss)

                    # .nii.gz/.vtk
                    save_grids(self.data_loader, im_pair_idxs, transformation1)
                    save_images(self.data_loader, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped1, self.mask_fixed)

            # checkpoint
            if iter_no % self.save_period == 0 or iter_no == self.no_iters_vi:
                self._save_checkpoint_vi(iter_no)

    def _train_epoch(self):
        for batch_idx, (im_pair_idxs, im_fixed, mask_fixed, seg_fixed, im_moving, mask_moving, seg_moving, mu_hat, log_var_hat, u_hat) \
                in enumerate(self.data_loader):
            if self.im_fixed is None:
                self.im_fixed = im_fixed.to(self.device, non_blocking=True)
            if self.mask_fixed is None:
                self.mask_fixed = mask_fixed.to(self.device, non_blocking=True)
            if self.seg_fixed is None:
                self.seg_fixed = seg_fixed.to(self.device, non_blocking=True)

            im_moving = im_moving.to(self.device, non_blocking=True)

            if mask_moving is not None:
                mask_moving = mask_moving.to(self.device, non_blocking=True)
            if seg_moving is not None:
                seg_moving = seg_moving.to(self.device, non_blocking=True)

            if self.mu_hat is None:
                self.mu_hat = mu_hat.to(self.device, non_blocking=True)
            if self.log_var_hat is None:
                self.log_var_hat = log_var_hat.to(self.device, non_blocking=True)
            if self.u_hat is None:
                self.u_hat = u_hat.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                z_hat = sample_q(self.mu_hat, self.log_var_hat, self.u_hat, omega_norm_inv=self.omega_norm_inv)

                transformation, displacement = self.transformation_model(z_hat)
                im_moving_warped = self.registration_module(im_moving, transformation)

                n_F, n_M = self.data_loss.map(self.im_fixed, im_moving_warped)
                res = n_F - n_M

                if self.use_moving_mask:
                    mask_moving_warped = self.registration_module(mask_moving, transformation)
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

                log_hist_res(self.writer, im_pair_idxs, res_masked, self.data_loss)  # residual histogram
                log_omega_norm(self.writer, self.omega_norm_inv)  # frequencies
                
                # dice scores
                dsc = calc_dice(self.seg_fixed, seg_moving, self.structures_dict)

                for structure in dsc:
                    score = dsc[structure]
                    self.train_metrics_vi.update('DSC/VI/' + structure, score)

                asd = calc_asd(self.seg_fixed, seg_moving, self.structures_dict, self.data_loader.dataset.spacing)

                for structure in asd:
                    dist = asd[structure]
                    self.train_metrics_vi.update('ASD/VI/' + structure, dist)
                
            """
            VI
            """

            if self.VI:
                self._step_VI(im_pair_idxs, im_moving, mask_moving, seg_moving)

    def _save_checkpoint_vi(self, iter_no):
        """
        save a checkpoint (variational inference)
        """

        state = {
            'config': self.config,
            'iter_no': iter_no,
            'sample_no': self.start_sample,

            'mu_hat': self.mu_hat,
            'log_var_hat': self.log_var_hat,
            'u_hat': self.u_hat,
            'optimizer_v': self.optimizer_v.state_dict(),

            'data_loss': self.data_loss.state_dict(),
            'optimizer_mixture_model': self.optimizer_mixture_model.state_dict(),
            'reg_loss': self.reg_loss.state_dict(),
        }

        if self.reg_loss.__class__.__name__ is not 'RegLoss_L2':
            state['reg_loss_prior_loc'] = self.reg_loss_prior_loc.state_dict()
            state['reg_loss_prior_scale'] = self.reg_loss_prior_scale.state_dict()
            state['optimizer_w_reg'] = self.optimizer_w_reg.state_dict()

        filename = str(self.checkpoint_dir / f'checkpoint_vi_{iter_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _save_checkpoint_mcmc(self, sample_no):
        """
        save a checkpoint (Markov chain Monte Carlo)
        """

        state = {
            'config': self.config,
            'iter_no': self.no_iters_vi,
            'sample_no': sample_no,

            'mu_hat': self.mu_hat,
            'log_var_hat': self.log_var_hat,
            'u_hat': self.u_hat,

            'data_loss': self.data_loss.state_dict(),
            'optimizer_mixture_model': self.optimizer_mixture_model.state_dict(),
            'reg_loss': self.reg_loss.state_dict(),

            'v_curr_state': self.v_curr_state,
            'optimizer_mala': self.optimizer_mala.state_dict()
        }

        if self.reg_loss.__class__.__name__ is not 'RegLoss_L2':
            state['reg_loss_prior_loc'] = self.reg_loss_prior_loc.state_dict()
            state['reg_loss_prior_scale'] = self.reg_loss_prior_scale.state_dict()
            state['optimizer_w_reg'] = self.optimizer_w_reg.state_dict()

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

        # VI
        self.mu_hat = checkpoint['mu_hat']
        self.log_var_hat = checkpoint['log_var_hat']
        self.u_hat = checkpoint['u_hat']

        self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [self.mu_hat, self.log_var_hat, self.u_hat])
        self.optimizer_v.load_state_dict(checkpoint['optimizer_v'])

        # GMM
        self.data_loss.load_state_dict(checkpoint['data_loss'])

        self.optimizer_mixture_model = Adam([{'params': [self.data_loss.log_std], 'lr': 1e-1},
                                             {'params': [self.data_loss.logits], 'lr': 1e-2}],
                                            lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)
        self.optimizer_mixture_model.load_state_dict(checkpoint['optimizer_mixture_model'])

        # regularisation loss
        self.reg_loss.load_state_dict(checkpoint['reg_loss'])

        if self.reg_loss.__class__.__name__ is not 'RegLoss_L2':
            self.reg_loss_prior_loc.load_state_dict(checkpoint['reg_loss_prior_loc'])
            self.reg_loss_prior_scale.load_state_dict(checkpoint['reg_loss_prior_scale'])

            self.optimizer_w_reg = Adam([{'params': [self.reg_loss.loc, self.reg_loss.log_scale]}],
                                        lr=5e-1, betas=(0.9, 0.95))
            self.optimizer_w_reg.load_state_dict(checkpoint['optimizer_w_reg'])

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

        # VI
        self.mu_hat = checkpoint['mu_hat']
        self.log_var_hat = checkpoint['log_var_hat']
        self.u_hat = checkpoint['u_hat']

        # GMM
        self.data_loss.load_state_dict(checkpoint['data_loss'])

        self.optimizer_mixture_model = Adam([{'params': [self.data_loss.log_std], 'lr': 1e-1},
                                             {'params': [self.data_loss.logits], 'lr': 1e-2}],
                                            lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)
        self.optimizer_mixture_model.load_state_dict(checkpoint['optimizer_mixture_model'])

        # regularisation loss
        self.reg_loss.load_state_dict(checkpoint['reg_loss'])

        if self.reg_loss.__class__.__name__ is not 'RegLoss_L2':
            self.reg_loss_prior_loc.load_state_dict(checkpoint['reg_loss_prior_loc'])
            self.reg_loss_prior_scale.load_state_dict(checkpoint['reg_loss_prior_scale'])

            self.optimizer_w_reg = Adam([{'params': [self.reg_loss.loc, self.reg_loss.log_scale]}],
                                        lr=5e-1, betas=(0.9, 0.95))
            self.optimizer_w_reg.load_state_dict(checkpoint['optimizer_w_reg'])

        # MCMC
        with torch.no_grad():
            self.v_curr_state = checkpoint['v_curr_state'] if 'v_curr_state' in checkpoint \
                else sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=1).detach()

        if 'optimizer_mala' in checkpoint:
            self.optimizer_mala = self.config.init_obj('optimizer_mala', torch.optim, [self.v_curr_state])
            self.optimizer_mala.load_state_dict(checkpoint['optimizer_mala'])

        self.logger.info("checkpoint loaded, resuming training..")
