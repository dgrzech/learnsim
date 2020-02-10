from base import BaseTrainer
from logger import log_fields, log_hist_res, log_images, log_sample, print_log, \
    save_fields, save_grids, save_images, save_norms, save_sample
from utils import add_noise, add_noise_uniform, calc_det_J, get_module_attr, inf_loop, max_field_update, sample_q_v, \
    save_optimiser_to_disk, separable_conv_3d, sobolev_kernel_1d, transform_coordinates, MetricTracker, SobolevGrad

import math
import numpy as np
import torch


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, data_loss, scale_prior, proportion_prior, reg_loss, entropy_loss, transformation_model, registration_module,
                 metric_ftns_vi, metric_ftns_mcmc, config, data_loader):
        super().__init__(data_loss, scale_prior, proportion_prior, reg_loss, entropy_loss,
                         transformation_model, registration_module, config)

        self.config = config
        self.data_loader = data_loader
        self.im_fixed, self.seg_fixed, self.mask_fixed = None, None, None

        # variational inference
        self.start_iter = 1
        self.mu_v, self.log_var_v, self.u_v = None, None, None

        self.VI = config['trainer']['vi']
        self.optimizer_v = None
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

        # resuming
        if config.resume is not None and self.VI:
            self._resume_checkpoint_vi(config.resume)
        elif config.resume is not None and self.MCMC:
            self._resume_checkpoint_mcmc(config.resume)

    def _step_VI(self, im_pair_idxs, im_moving):
        if self.optimizer_v is None:
            self.mu_v.requires_grad_(True)
            self.log_var_v.requires_grad_(True)
            self.u_v.requires_grad_(True)

            self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [self.mu_v, self.log_var_v, self.u_v])

        for iter_no in range(self.start_iter, self.no_iters_vi + 1):
            self.train_metrics_vi.reset()

            self.optimizer_mixture_model.zero_grad()
            self.optimizer_v.zero_grad()

            if iter_no % self.log_period == 0 or iter_no == self.no_iters_vi:
                mu_v_old = self.mu_v.detach().clone()  # neded to calculate the maximum update in terms of the L2 norm
                log_var_v_old = self.log_var_v.detach().clone()
                u_v_old = self.u_v.detach().clone()
            
            data_term = 0.0
            reg_term = 0.0
            entropy_term = 0.0

            v_sample1, v_sample2 = sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=2)
            if self.sobolev_grad:
                v_sample1 = SobolevGrad.apply(v_sample1, self.S_x, self.S_y, self.S_z, self.padding_sz)
                v_sample2 = SobolevGrad.apply(v_sample2, self.S_x, self.S_y, self.S_z, self.padding_sz)

            transformation1, displacement1 = self.transformation_model(v_sample1)
            transformation2, displacement2 = self.transformation_model(v_sample2)

            # add noise to account for interpolation uncertainty
            transformation1, displacement1 = add_noise_uniform(transformation1, self.log_var_v), \
                                             add_noise_uniform(displacement1, self.log_var_v)

            transformation2, displacement2 = add_noise_uniform(transformation2, self.log_var_v), \
                                             add_noise_uniform(displacement2, self.log_var_v)

            im_moving_warped1 = self.registration_module(im_moving, transformation1)
            im_moving_warped2 = self.registration_module(im_moving, transformation2)

            n_F, n_M1 = self.data_loss.map(self.im_fixed, im_moving_warped1)
            n_F, n_M2 = self.data_loss.map(self.im_fixed, im_moving_warped2)

            res1 = ((n_F - n_M1) * self.mask_fixed).view(1, -1, 1)
            res2 = ((n_F - n_M2) * self.mask_fixed).view(1, -1, 1)

            if iter_no == 1:
                no_warm_up_steps = 50

                for step in range(1, no_warm_up_steps + 1):
                    loss_gmm = 0.0
                    self.optimizer_mixture_model.zero_grad()

                    loss_gmm += self.data_loss(res1) / 2.0
                    loss_gmm += self.data_loss(res2) / 2.0

                    loss_gmm -= torch.sum(self.scale_prior(self.data_loss.log_scales()))
                    loss_gmm -= torch.sum(self.proportion_prior(self.data_loss.log_proportions()))

                    loss_gmm.backward(retain_graph=True)
                    self.optimizer_mixture_model.step()

            data_term += self.data_loss(res1) / 2.0
            data_term += self.data_loss(res2) / 2.0

            data_term -= torch.sum(self.scale_prior(self.data_loss.log_scales()))
            data_term -= torch.sum(self.proportion_prior(self.data_loss.log_proportions()))

            reg_term += self.reg_loss(v_sample1).sum() / 2.0
            reg_term += self.reg_loss(v_sample2).sum() / 2.0

            entropy_term += self.entropy_loss(v_sample=v_sample1,
                                              mu_v=self.mu_v, log_var_v=self.log_var_v, u_v=self.u_v).sum() / 2.0
            entropy_term += self.entropy_loss(v_sample=v_sample2,
                                              mu_v=self.mu_v, log_var_v=self.log_var_v, u_v=self.u_v).sum() / 2.0
            entropy_term += self.entropy_loss(log_var_v=self.log_var_v, u_v=self.u_v).sum()

            loss_q_v = data_term + reg_term - entropy_term
            loss_q_v.backward()

            self.optimizer_mixture_model.step()  # backprop
            self.optimizer_v.step()

            """
            metrics and prints
            """

            self.writer.set_step(iter_no)

            self.train_metrics_vi.update('VI/data_term', data_term.item())
            self.train_metrics_vi.update('VI/reg_term', reg_term.item())
            self.train_metrics_vi.update('VI/entropy_term', entropy_term.item())
            self.train_metrics_vi.update('VI/total_loss', loss_q_v.item())

            if iter_no % self.log_period == 0 or iter_no == self.no_iters_vi:
                with torch.no_grad():
                    max_update_mu_v, max_update_mu_v_idx = max_field_update(mu_v_old, self.mu_v)
                    max_update_log_var_v, max_update_log_var_v_idx = max_field_update(log_var_v_old, self.log_var_v)
                    max_update_u_v, max_update_u_v_idx = max_field_update(u_v_old, self.u_v)

                self.train_metrics_vi.update('max_updates/mu_v', max_update_mu_v.item())
                self.train_metrics_vi.update('max_updates/log_var_v', max_update_log_var_v.item())
                self.train_metrics_vi.update('max_updates/u_v', max_update_u_v.item())

                log = {'iter_no': iter_no}
                log.update(self.train_metrics_vi.result())
                print_log(self.logger, log)

            """
            outputs
            """

            if math.log2(iter_no).is_integer():
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
                    log_det_J_transformation = torch.log10(calc_det_J(nabla_x, nabla_y, nabla_z))

                    # tensorboard
                    log_fields(self.writer, im_pair_idxs, var_params, displacement, log_det_J_transformation)
                    log_images(self.writer, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped)

                    if iter_no == 1:
                        log_hist_res(self.writer, im_pair_idxs, res1, self.data_loss)

                    # .nii.gz/.vtk
                    save_fields(
                        self.data_loader.save_dirs, im_pair_idxs, var_params, displacement, log_det_J_transformation)
                    save_grids(self.data_loader.save_dirs, im_pair_idxs, transformation)
                    save_images(self.data_loader.save_dirs, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped)
                    save_norms(self.data_loader.save_dirs, im_pair_idxs, var_params, displacement)

            # checkpoint
            if iter_no % self.save_period == 0 or iter_no == self.no_iters_vi:
                self._save_checkpoint_vi(iter_no)

        self.mu_v.requires_grad_(False)
        self.log_var_v.requires_grad_(False)
        self.u_v.requires_grad_(False)

    def _step_MCMC(self, im_pair_idxs, im_moving):
        if self.optimizer_mala is None:
            self.v_curr_state.requires_grad_(True)
            self.optimizer_mala = self.config.init_obj('optimizer_mala', torch.optim, [self.v_curr_state])

        self.logger.info('\nBURNING IN THE MARKOV CHAIN\n')

        for sample_no in range(self.start_sample, self.no_samples + 1):
            self.train_metrics_mcmc.reset()
            self.optimizer_mala.zero_grad()

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
                reg_term = self.reg_loss(v_curr_state_noise_smoothed).sum()
            else:
                transformation, displacement = self.transformation_model(v_curr_state_noise)
                reg_term = self.reg_loss(v_curr_state_noise).sum()

            transformation, displacement = add_noise_uniform(transformation, self.log_var_v), \
                                           add_noise_uniform(displacement, self.log_var_v)

            im_moving_warped = self.registration_module(im_moving, transformation)
            data_term = self.data_loss(self.im_fixed, im_moving_warped, self.mask_fixed).sum()

            loss = data_term + reg_term
            loss.backward()
            self.optimizer_mala.step()  # backprop

            """
            metrics and prints
            """

            self.writer.set_step(sample_no)

            self.train_metrics_mcmc.update('MCMC/data_term', data_term.item())
            self.train_metrics_mcmc.update('MCMC/reg_term', reg_term.item())

            if sample_no == self.no_iters_burn_in:
                self.logger.info('\nENDED BURNING IN\n')

            # tensorboard
            if sample_no > self.no_iters_burn_in and sample_no % 10000 == 0:
                with torch.no_grad():
                    log = {'sample_no': sample_no}
                    log.update(self.train_metrics_mcmc.result())
                    print_log(self.logger, log)

                    if self.sobolev_grad:
                        log_sample(self.writer, im_pair_idxs,
                                   im_moving_warped, v_curr_state_noise_smoothed, displacement)
                    else:
                        log_sample(self.writer, im_pair_idxs, im_moving_warped, self.v_curr_state, displacement)
            
            """
            outputs
            """

            if sample_no % 10000 == 0 or sample_no == self.no_samples:
                with torch.no_grad():
                    if self.sobolev_grad:
                        save_sample(self.data_loader.save_dirs, im_pair_idxs,
                                    sample_no, im_moving_warped, v_curr_state_noise_smoothed)
                    else:
                        save_sample(self.data_loader.save_dirs, im_pair_idxs,
                                    sample_no, im_moving_warped, v_curr_state_noise)

                    self._save_checkpoint_mcmc(sample_no)  # checkpoint

    def _train_epoch(self):
        for batch_idx, (im_pair_idxs, im_fixed, mask_fixed, im_moving, mu_v, log_var_v, u_v) \
                in enumerate(self.data_loader):
            self.im_fixed = im_fixed.to(self.device, non_blocking=True)
            self.mask_fixed = mask_fixed.to(self.device, non_blocking=True)

            im_moving = im_moving.to(self.device, non_blocking=True)

            if self.mu_v is None:
                self.mu_v = mu_v.to(self.device, non_blocking=True)
            if self.log_var_v is None:
                self.log_var_v = log_var_v.to(self.device, non_blocking=True)
            if self.u_v is None:
                self.u_v = u_v.to(self.device, non_blocking=True)

            # print value of the data term before registration and initialsie the GMM
            with torch.no_grad():
                if self.sobolev_grad:
                    v_smoothed = SobolevGrad.apply(self.mu_v, self.S_x, self.S_y, self.S_z, self.padding_sz)
                    transformation, displacement = self.transformation_model(v_smoothed)
                else:
                    transformation, displacement = self.transformation_model(self.mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)

                n_F, n_M_unwarped = self.data_loss.map(self.im_fixed, im_moving)
                n_F, n_M_warped = self.data_loss.map(self.im_fixed, im_moving_warped)
                
                # initialise the GMM
                res_unwarped = ((n_F - n_M_unwarped) * self.mask_fixed).view(1, -1, 1)
                res_warped = ((n_F - n_M_warped) * self.mask_fixed).view(1, -1, 1)

                self.data_loss.initialise_parameters(torch.std(res_unwarped))
                self.logger.info('initialised the GMM\n')

                loss_unwarped = self.data_loss(res_unwarped)
                loss_warped = self.data_loss(res_warped)
                
                self.logger.info(f'PRE-REGISTRATION: ' +
                                 f'unwarped: {loss_unwarped:.5f}, warped w/ the current state: {loss_warped:.5f}\n')

            """
            VI
            """

            if self.VI:
                self._step_VI(im_pair_idxs, im_moving)

            """
            MCMC
            """

            if self.MCMC:
                tau = self.config['optimizer_mala']['args']['lr']
                sqrt_tau_twice = np.sqrt(2.0 * tau)

                self.sigma_scaled = sqrt_tau_twice * transform_coordinates(torch.exp(0.5 * self.log_var_v))
                self.u_v_scaled = sqrt_tau_twice * transform_coordinates(self.u_v)

                if self.v_curr_state is None:
                    self.v_curr_state = self.mu_v.clone()

                self._step_MCMC(im_pair_idxs, im_moving)

    def _save_checkpoint_vi(self, iter_no):
        """
        save a checkpoint
        """

        state = {
            'config': self.config,
            'iter': iter_no,

            'mu_v': self.mu_v,
            'log_var_v': self.log_var_v,
            'u_v': self.u_v,
            'optimizer_v': self.optimizer_v.state_dict()
        }

        filename = str(self.checkpoint_dir / f'checkpoint_vi_{iter_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _save_checkpoint_mcmc(self, sample_no):
        state = {
            'config': self.config,
            'sample_no': sample_no,

            'mu_v': self.mu_v,
            'log_var_v': self.log_var_v,
            'u_v': self.u_v,

            'v_curr_state': self.v_curr_state,
            'optimizer_mala': self.optimizer_mala.state_dict()
        }

        filename = str(self.checkpoint_dir / f'checkpoint_mcmc_{sample_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _resume_checkpoint_vi(self, resume_path):
        """
        resume from saved checkpoints

        :param resume_path: checkpoint path to be resumed
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {}..".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_iter = checkpoint['iter'] + 1

        self.mu_v = checkpoint['mu_v']
        self.log_var_v = checkpoint['log_var_v']
        self.u_v = checkpoint['u_v']

        self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [self.mu_v, self.log_var_v, self.u_v])
        self.optimizer_v.load_state_dict(checkpoint['optimizer_v'])

        self.logger.info("checkpoint loaded, resuming training..")

    def _resume_checkpoint_mcmc(self, resume_path):
        """
        resume from saved checkpoints

        :param resume_path: checkpoint path to be resumed
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {}..".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_sample = checkpoint['sample_no'] + 1

        self.mu_v = checkpoint['mu_v']
        self.log_var_v = checkpoint['log_var_v']
        self.u_v = checkpoint['u_v']

        self.v_curr_state = checkpoint['v_curr_state']

        self.optimizer_mala = self.config.init_obj('optimizer_mala', torch.optim, [self.v_curr_state])
        self.optimizer_mala.load_state_dict(checkpoint['optimizer_mala'])

        self.logger.info("checkpoint loaded, resuming training..")
