from datetime import datetime

import torch

from base import BaseTrainer
from logger import log_fields, log_hist_res, log_images, log_sample, print_log, save_fields, save_grids, save_images, \
    save_sample
from optimizers import Adam
from utils import MetricTracker, SGLD, SobolevGrad, Sobolev_kernel_1D, VD, add_noise_uniform, calc_ASD, calc_DSC, \
    calc_det_J, max_field_update, rescale_residuals, sample_q_v


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, data_loss, data_loss_scale_prior, data_loss_proportion_prior,
                 reg_loss, reg_loss_loc_prior, reg_loss_scale_prior,
                 entropy_loss, transformation_model, registration_module,
                 metric_ftns_VI, metric_ftns_MCMC, structures_dict, config, data_loader):
        super().__init__(data_loss, data_loss_scale_prior, data_loss_proportion_prior,
                         reg_loss, reg_loss_loc_prior, reg_loss_scale_prior,
                         entropy_loss, transformation_model, registration_module, config)
        self.config = config
        self.data_loader = data_loader
        self.im_fixed, self.mask_fixed, self.seg_fixed = None, None, None
        self.structures_dict = structures_dict  # segmentations

        cfg_trainer = config['trainer']
        cfg_data_loader = config['data_loader']['args']

        self.N = cfg_data_loader['dim_x'] * cfg_data_loader['dim_y'] * cfg_data_loader['dim_z']
        self.dof = self.N * 3.0

        # variational inference
        self.start_iter = 1
        self.mu_v, self.log_var_v, self.u_v = None, None, None

        self.VI = cfg_trainer['VI']
        self.optimizer_GMM, self.optimizer_w_reg, self.optimizer_v = None, None, None
        self.metrics_VI = MetricTracker(*[m for m in metric_ftns_VI], writer=self.writer)

        # stochastic gradient Markov chain Monte Carlo
        self.start_sample = 1
        self.v_curr_state = None
        self.SGLD_params = {'tau': None, 'sigma': None, 'u': None}

        self.MCMC = cfg_trainer['MCMC']
        self.optimizer_SG_MCMC = None
        self.metrics_MCMC = MetricTracker(*[m for m in metric_ftns_MCMC], writer=self.writer)

        # Sobolev gradients
        self.Sobolev_grad = config['Sobolev_grad']['enabled']

        if self.Sobolev_grad:
            _s = config['Sobolev_grad']['s']
            _lambda = config['Sobolev_grad']['lambda']
            self.padding_sz = _s // 2

            S, S_sqrt = Sobolev_kernel_1D(_s, _lambda)

            S = torch.from_numpy(S).float().unsqueeze(0)
            S = torch.stack((S, S, S), 0)

            S_x = S.unsqueeze(2).unsqueeze(2).to(self.device, non_blocking=True)
            S_y = S.unsqueeze(2).unsqueeze(4).to(self.device, non_blocking=True)
            S_z = S.unsqueeze(3).unsqueeze(4).to(self.device, non_blocking=True)

            self.S = {'x': S_x, 'y': S_y, 'z': S_z}

        # uniform noise magnitude
        self.add_noise_uniform = cfg_trainer['uniform_noise']['enabled']

        if self.add_noise_uniform:
            self.alpha = cfg_trainer['uniform_noise']['magnitude']

        # virtual decimation
        self.virutal_decimation = config['virtual_decimation']

        # resuming
        if config.resume is not None and self.VI:
            self._resume_checkpoint_VI(config.resume)
        elif config.resume is not None and self.MCMC:
            self._resume_checkpoint_MCMC(config.resume)

    def __init_optimizer_w_reg(self):
        self.optimizer_w_reg = Adam([{'params': [self.reg_loss.loc, self.reg_loss.log_scale]}], lr=1e-1,
                                    betas=(0.9, 0.95))

    def __init_optimizer_GMM(self):
        self.optimizer_GMM = Adam(
            [{'params': [self.data_loss.log_std], 'lr': 1e-1}, {'params': [self.data_loss.logits], 'lr': 1e-2}],
            lr=1e-2, betas=(0.9, 0.95), lr_decay=1e-3)

    def _step_GMM(self, res, alpha=1.0):
        if self.optimizer_GMM is None:  # initialise the optimiser
            self.__init_optimizer_GMM()

        data_term = self.data_loss(res.detach()).sum() * alpha
        data_term -= self.data_loss_scale_prior(self.data_loss.log_scales()).sum()
        data_term -= self.data_loss_proportion_prior(self.data_loss.log_proportions()).sum()

        self.optimizer_GMM.zero_grad()
        data_term.backward()
        self.optimizer_GMM.step()  # backprop

    def _run_VI(self, im_pair_idxs, im_moving, mask_moving, seg_moving):
        self.mu_v.requires_grad_(True)
        self.log_var_v.requires_grad_(True)
        self.u_v.requires_grad_(True)

        if self.optimizer_v is None:
            self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [self.mu_v, self.log_var_v, self.u_v])

        if self.reg_loss_type is not 'RegLoss_L2' and self.optimizer_w_reg is None:
            self.__init_optimizer_w_reg()

        for iter_no in range(self.start_iter, self.no_iters_VI + 1):
            self.metrics_VI.reset()

            if iter_no % self.log_period_VI == 0 or iter_no == self.no_iters_VI:
                # needed to calculate the maximum update in terms of the L2 norm
                mu_v_old = self.mu_v.detach().clone()
                log_var_v_old = self.log_var_v.detach().clone()
                u_v_old = self.u_v.detach().clone()

            v_sample1_unsmoothed, v_sample2_unsmoothed = sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=2)
            v_sample1 = SobolevGrad.apply(v_sample1_unsmoothed, self.S, self.padding_sz)
            v_sample2 = SobolevGrad.apply(v_sample2_unsmoothed, self.S, self.padding_sz)

            transformation1, displacement1 = self.transformation_model(v_sample1)
            transformation2, displacement2 = self.transformation_model(v_sample2)

            with torch.no_grad():
                nabla = self.diff_op(transformation1, transformation=True)
                log_det_J_transformation = torch.log(calc_det_J(nabla))
                no_non_diffeomorphic_voxels = torch.isnan(log_det_J_transformation).sum().item()

            if self.add_noise_uniform:  # add noise to account for interpolation uncertainty
                transformation1 = add_noise_uniform(transformation1, self.alpha)
                transformation2 = add_noise_uniform(transformation2, self.alpha)

            im_moving_warped1, im_moving_warped2 = self.registration_module(im_moving, transformation1), \
                                                   self.registration_module(im_moving, transformation2)

            n_F, n_M1 = self.data_loss.map(self.im_fixed, im_moving_warped1)
            n_F, n_M2 = self.data_loss.map(self.im_fixed, im_moving_warped2)

            res1, res2 = n_F - n_M1, n_F - n_M2

            alpha1, alpha2 = 1.0, 1.0
            alpha_mean = 1.0

            if self.virutal_decimation:
                # rescale the residuals by the estimated voxel-wise standard deviation
                res1_rescaled = rescale_residuals(res1.detach(), self.mask_fixed, self.data_loss)
                res2_rescaled = rescale_residuals(res2.detach(), self.mask_fixed, self.data_loss)

                with torch.no_grad():
                    alpha1, alpha2 = VD(res1_rescaled, self.mask_fixed), VD(res2_rescaled, self.mask_fixed)
                    alpha_mean = (alpha1.item() + alpha2.item()) / 2.0

                res1_masked, res2_masked = res1[self.mask_fixed], res2[self.mask_fixed]

            # Gaussian mixture
            self._step_GMM(res1_masked, alpha1)

            # q_v
            data_term1 = self.data_loss(res1_masked).sum() / 2.0 * alpha1
            data_term2 = self.data_loss(res2_masked).sum() / 2.0 * alpha2

            data_term_scale_prior = self.data_loss_scale_prior(self.data_loss.log_scales()).sum()
            data_term_proportion_prior = self.data_loss_proportion_prior(self.data_loss.log_proportions()).sum()

            data_term = data_term1 + data_term2 - data_term_scale_prior - data_term_proportion_prior

            reg_term1, log_y1 = \
                self.reg_loss(v_sample1, dof=self.dof) if self.reg_loss_type is not 'RegLoss_L2' \
                    else self.reg_loss(v_sample1)
            reg_term2, log_y2 = \
                self.reg_loss(v_sample2, dof=self.dof) if self.reg_loss_type is not 'RegLoss_L2' \
                    else self.reg_loss(v_sample2)

            reg_term = reg_term1.sum() / 2.0 + reg_term2.sum() / 2.0

            if self.reg_loss_type is not 'RegLoss_L2':
                reg_term_loc_prior1 = self.reg_loss_loc_prior(log_y1).sum() / 2.0
                reg_term_loc_prior2 = self.reg_loss_loc_prior(log_y2).sum() / 2.0
                reg_term_scale_prior = self.reg_loss_scale_prior(self.reg_loss.log_scale).sum()

                reg_term -= (reg_term_loc_prior1 + reg_term_loc_prior2 + reg_term_scale_prior)

            entropy_term1 = self.entropy_loss(v_sample=v_sample1_unsmoothed, mu_v=self.mu_v, log_var_v=self.log_var_v,
                                              u_v=self.u_v).sum() / 2.0
            entropy_term2 = self.entropy_loss(v_sample=v_sample2_unsmoothed, mu_v=self.mu_v, log_var_v=self.log_var_v,
                                              u_v=self.u_v).sum() / 2.0
            entropy_term3 = self.entropy_loss(log_var_v=self.log_var_v, u_v=self.u_v).sum()

            entropy_term = entropy_term1 + entropy_term2 + entropy_term3

            self.optimizer_v.zero_grad()

            if self.reg_loss_type is not 'RegLoss_L2':
                self.optimizer_w_reg.zero_grad()

            loss_q_v = data_term + reg_term - entropy_term
            loss_q_v.backward()  # backprop
            self.optimizer_v.step()

            if self.reg_loss_type is not 'RegLoss_L2':
                self.optimizer_w_reg.step()

            """
            outputs
            """

            self.writer.set_step(iter_no)

            self.metrics_VI.update('VI/train/data_term', data_term.item())
            self.metrics_VI.update('VI/train/reg_term', reg_term.item())
            self.metrics_VI.update('VI/train/entropy_term', entropy_term.item())
            self.metrics_VI.update('VI/train/total_loss', loss_q_v.item())

            self.metrics_VI.update('VI/train/no_non_diffeomorphic_voxels', no_non_diffeomorphic_voxels)
            self.metrics_VI.update('VI/train/alpha', alpha_mean)

            if iter_no % self.log_period_VI == 0 or iter_no == self.no_iters_VI:
                with torch.no_grad():
                    """
                    metrics and prints
                    """

                    sigmas = torch.exp(self.data_loss.log_scales())
                    proportions = torch.exp(self.data_loss.log_proportions())

                    for idx in range(self.data_loss.num_components):
                        self.metrics_VI.update('VI/train/GMM/sigma_' + str(idx), sigmas[idx])
                        self.metrics_VI.update('VI/train/GMM/proportion_' + str(idx), proportions[idx])

                    if self.reg_loss_type is not 'RegLoss_L2':
                        self.metrics_VI.update('VI/train/loc', self.reg_loss.loc.item())
                        self.metrics_VI.update('VI/train/log_scale', self.reg_loss.log_scale.item())

                    self.metrics_VI.update('VI/train/y', log_y1.exp().item())

                    max_update_mu_v, max_update_mu_v_idx = max_field_update(mu_v_old, self.mu_v)
                    max_update_log_var_v, max_update_log_var_v_idx = max_field_update(log_var_v_old, self.log_var_v)
                    max_update_u_v, max_update_u_v_idx = max_field_update(u_v_old, self.u_v)

                    self.metrics_VI.update('VI/train/max_updates/mu_v', max_update_mu_v.item())
                    self.metrics_VI.update('VI/train/max_updates/log_var_v', max_update_log_var_v.item())
                    self.metrics_VI.update('VI/train/max_updates/u_v', max_update_u_v.item())

                    # Dice scores
                    seg_moving_warped = self.registration_module(seg_moving, transformation1)
                    DSC = calc_DSC(self.seg_fixed, seg_moving_warped, self.structures_dict)

                    for structure in DSC:
                        score = DSC[structure]
                        self.metrics_VI.update('VI/train/DSC/' + structure, score)

                    # average surface distances
                    ASD = calc_ASD(self.seg_fixed, seg_moving_warped, self.structures_dict, self.data_loader.spacing)

                    for structure in ASD:
                        dist = ASD[structure]
                        self.metrics_VI.update('VI/train/ASD/' + structure, dist)

                    log = {'iter_no': iter_no}
                    log.update(self.metrics_VI.result())
                    print_log(self.logger, log)

                    """
                    logging
                    """

                    mu_v_smoothed = SobolevGrad.apply(self.mu_v, self.S, self.padding_sz)
                    log_var_v_smoothed = SobolevGrad.apply(self.log_var_v, self.S, self.padding_sz)
                    sigma_v_smoothed = torch.exp(0.5 * log_var_v_smoothed)
                    u_v_smoothed = SobolevGrad.apply(self.u_v, self.S, self.padding_sz)

                    var_params = {'mu_v': mu_v_smoothed, 'log_var_v': log_var_v_smoothed, 'u_v': u_v_smoothed}

                    # tensorboard
                    log_hist_res(self.writer, im_pair_idxs, res1_masked, self.data_loss)
                    log_images(self.writer, im_pair_idxs, self.im_fixed, im_moving, im_moving_warped1)
                    log_fields(self.writer, im_pair_idxs, var_params, displacement1, log_det_J_transformation)

                    if iter_no == self.no_iters_VI:
                        transformation, displacement = self.transformation_model(mu_v_smoothed)
                        im_moving_warped = self.registration_module(im_moving, transformation)
                    else:
                        transformation = transformation1.detach().clone()
                        displacement = displacement1.detach().clone()
                        im_moving_warped = im_moving_warped1.detach().clone()

                    # .nii.gz/.vtk
                    save_images(self.data_loader, im_pair_idxs,
                                im_fixed=self.im_fixed, im_moving=im_moving,
                                im_moving_warped=im_moving_warped, mask_fixed=self.mask_fixed)
                    save_fields(self.data_loader, im_pair_idxs,
                                mu_v=mu_v_smoothed, sigma_v=sigma_v_smoothed, u_v=u_v_smoothed,
                                displacement=displacement)
                    save_grids(self.data_loader, im_pair_idxs, transformation)

                    # checkpoint
                    self._save_checkpoint_VI(iter_no)

            if no_non_diffeomorphic_voxels > 0.001 * self.N:
                self.logger.info("detected " + str(
                    no_non_diffeomorphic_voxels) + " voxels where the sample transformation is not diffeomorphic, "
                                                   "exiting..")
                exit()

    def _test_VI(self, im_pair_idxs, im_moving, mask_moving, seg_moving):
        """
        metrics
        """

        with torch.no_grad():
            for test_sample_no in range(1, self.no_samples_VI_test + 1):
                self.writer.set_step(test_sample_no)

                v_sample = sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=1)
                v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding_sz)
                transformation, displacement = self.transformation_model(v_sample_smoothed)

                im_moving_warped = self.registration_module(im_moving, transformation)
                seg_moving_warped = self.registration_module(seg_moving, transformation)

                # Dice scores
                DSC = calc_DSC(self.seg_fixed, seg_moving_warped, self.structures_dict)
                # average surface distances
                ASD = calc_ASD(self.seg_fixed, seg_moving_warped, self.structures_dict, self.data_loader.spacing)

                for structure in DSC:
                    score = DSC[structure]
                    self.metrics_VI.update('VI/test/DSC/' + structure, score)

                for structure in ASD:
                    score = ASD[structure]
                    self.metrics_VI.update('VI/test/ASD/' + structure, score)

                nabla = self.diff_op(transformation, transformation=True)
                log_det_J_transformation = torch.log(calc_det_J(nabla))
                no_non_diffeomorphic_voxels = torch.isnan(log_det_J_transformation).sum().item()

                self.metrics_VI.update('VI/test/no_non_diffeomorphic_voxels', no_non_diffeomorphic_voxels)
                save_sample(self.data_loader, im_pair_idxs, test_sample_no, im_moving_warped, displacement, model='VI')

        """
        speed
        """

        with torch.no_grad():
            start = datetime.now()

            for VI_test_sample_no in range(1, self.no_samples_VI_test * 10 + 1):
                v_sample = sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=1)
                v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding_sz)

                transformation, displacement = self.transformation_model(v_sample_smoothed)

                im_moving_warped = self.registration_module(im_moving, transformation)
                seg_moving_warped = self.registration_module(seg_moving, transformation)

        stop = datetime.now()
        VI_sampling_speed = (self.no_samples_VI_test * 10 + 1) / (stop - start).total_seconds()

        self.logger.info(f'VI sampling speed: {VI_sampling_speed:.2f} samples/sec')

    def _run_MCMC(self, im_pair_idxs, im_moving, mask_moving, seg_moving):
        dof = self.dof if self.reg_loss_type is not 'RegLoss_L2' else 0.0

        self.mu_v.requires_grad_(False)
        self.log_var_v.requires_grad_(False)
        self.u_v.requires_grad_(False)
        self.v_curr_state.requires_grad_(True)

        if self.optimizer_SG_MCMC is None:
            self.optimizer_SG_MCMC = self.config.init_obj('optimizer_SG_MCMC', torch.optim, [self.v_curr_state])

        self.logger.info('\nBURNING IN THE MARKOV CHAIN\n')
        start = datetime.now()

        for sample_no in range(self.start_sample, self.no_samples_MCMC + 1):
            self.metrics_MCMC.reset()

            if sample_no < self.no_iters_burn_in and sample_no % self.log_period_MCMC == 0:
                self.logger.info('burn-in sample no. ' + str(sample_no) + '/' + str(self.no_iters_burn_in))

            """
            stochastic gradient Langevin dynamics
            """

            v_curr_state_noise = SGLD.apply(self.v_curr_state, self.SGLD_params['sigma'], self.SGLD_params['tau'])
            v_curr_state_noise_smoothed = SobolevGrad.apply(v_curr_state_noise, self.S, self.padding_sz)
            transformation, displacement = self.transformation_model(v_curr_state_noise_smoothed)

            reg, log_y = self.reg_loss(v_curr_state_noise_smoothed, dof=dof)
            reg_term = reg.sum()

            with torch.no_grad():
                nabla = self.diff_op(transformation, transformation=True)
                log_det_J_transformation = torch.log(calc_det_J(nabla))
                no_non_diffeomorphic_voxels = torch.isnan(log_det_J_transformation).sum().item()

            reg_term -= self.reg_loss_loc_prior(log_y).sum()
            reg_term -= self.reg_loss_scale_prior(self.reg_loss.log_scale).sum()

            if self.add_noise_uniform:
                transformation = add_noise_uniform(transformation, self.alpha)

            im_moving_warped = self.registration_module(im_moving, transformation)
            n_F, n_M = self.data_loss.map(self.im_fixed, im_moving_warped)
            res = n_F - n_M

            alpha = 1.0
            alpha_mean = alpha

            if self.virutal_decimation:
                res_rescaled = rescale_residuals(res.detach(), self.mask_fixed, self.data_loss)
                res_masked = res[self.mask_fixed]

                with torch.no_grad():
                    alpha = VD(res_rescaled, self.mask_fixed)
                    alpha_mean = alpha.item()

            # Gaussian mixture
            self._step_GMM(res_masked, alpha)

            # MCMC
            data_term = self.data_loss(res_masked).sum() * alpha
            data_term -= self.data_loss_scale_prior(self.data_loss.log_scales()).sum()
            data_term -= self.data_loss_proportion_prior(self.data_loss.log_proportions()).sum()

            loss = data_term + reg_term

            self.optimizer_SG_MCMC.zero_grad()
            self.optimizer_w_reg.zero_grad()

            loss.backward()  # backprop

            self.optimizer_SG_MCMC.step()
            self.optimizer_w_reg.step()

            """
            outputs
            """

            self.writer.set_step(sample_no)

            self.metrics_MCMC.update('MCMC/data_term', data_term.item())
            self.metrics_MCMC.update('MCMC/reg_term', reg_term.item())
            self.metrics_MCMC.update('MCMC/no_non_diffeomorphic_voxels', no_non_diffeomorphic_voxels)
            self.metrics_MCMC.update('MCMC/alpha', alpha_mean)
            self.metrics_MCMC.update('MCMC/loc', self.reg_loss.loc.item())
            self.metrics_MCMC.update('MCMC/log_scale', self.reg_loss.log_scale.item())

            if sample_no == self.no_iters_burn_in:
                self.logger.info('\nENDED BURNING IN')

                stop = datetime.now()
                MCMC_sampling_speed = self.no_iters_burn_in / (stop - start).total_seconds()
                self.logger.info(f'SG-MCMC sampling speed: {MCMC_sampling_speed:.2f} samples/sec\n')

            if sample_no > self.no_iters_burn_in and sample_no % self.log_period_MCMC == 0 \
                    or sample_no == self.no_samples_MCMC:
                with torch.no_grad():
                    """
                    metrics and prints
                    """

                    log = {'sample_no': sample_no}
                    log.update(self.metrics_MCMC.result())
                    print_log(self.logger, log)

                    sigmas = torch.exp(self.data_loss.log_scales())
                    proportions = torch.exp(self.data_loss.log_proportions())

                    for idx in range(self.data_loss.num_components):
                        self.metrics_MCMC.update('MCMC/GMM/sigma_' + str(idx), sigmas[idx])
                        self.metrics_MCMC.update('MCMC/GMM/proportion_' + str(idx), proportions[idx])

                    self.metrics_MCMC.update('MCMC/y', log_y.exp().item())

                    v_curr_state_smoothed = SobolevGrad.apply(self.v_curr_state, self.S, self.padding_sz)
                    transformation, displacement = self.transformation_model(v_curr_state_smoothed)

                    # Dice scores
                    seg_moving_warped = self.registration_module(seg_moving, transformation)
                    DSC = calc_DSC(self.seg_fixed, seg_moving_warped, self.structures_dict)

                    for structure in DSC:
                        score = DSC[structure]
                        self.metrics_MCMC.update('MCMC/DSC/' + structure, score)

                    # average surface distances
                    ASD = calc_ASD(self.seg_fixed, seg_moving_warped, self.structures_dict, self.data_loader.spacing)

                    for structure in ASD:
                        dist = ASD[structure]
                        self.metrics_MCMC.update('MCMC/ASD/' + structure, dist)

                    """
                    logging
                    """

                    # tensorboard
                    log_sample(self.writer, im_pair_idxs, self.data_loss,
                               im_moving_warped, res_masked, v_curr_state_smoothed, displacement,
                               log_det_J_transformation)

                    # .nii.gz/.vtk
                    save_sample(self.data_loader, im_pair_idxs, sample_no, im_moving_warped, displacement, model='MCMC')

                    # checkpoint
                    self._save_checkpoint_MCMC(sample_no)

            if no_non_diffeomorphic_voxels > 0.001 * self.N:
                self.logger.info("sample " + str(sample_no) + ", detected " + str(
                    no_non_diffeomorphic_voxels) + " voxels where the sample transformation is not diffeomorphic; "
                                                   "exiting..")
                exit()

    def _train_epoch(self):
        for batch_idx, (
                im_pair_idxs, im_fixed, mask_fixed, seg_fixed, im_moving, mask_moving, seg_moving, mu_v, log_var_v,
                u_v) \
                in enumerate(self.data_loader):
            if self.im_fixed is None:
                self.im_fixed = im_fixed.to(self.device, non_blocking=True)
            if self.mask_fixed is None:
                self.mask_fixed = mask_fixed.to(self.device, non_blocking=True)
            if self.seg_fixed is None:
                self.seg_fixed = seg_fixed.to(self.device, non_blocking=True)

            im_moving = im_moving.to(self.device, non_blocking=True)
            mask_moving = mask_moving.to(self.device, non_blocking=True)
            seg_moving = seg_moving.to(self.device, non_blocking=True)

            if self.mu_v is None:
                self.mu_v = mu_v.to(self.device, non_blocking=True)
            if self.log_var_v is None:
                self.log_var_v = log_var_v.to(self.device, non_blocking=True)
            if self.u_v is None:
                self.u_v = u_v.to(self.device, non_blocking=True)

            with torch.no_grad():
                v_sample = sample_q_v(self.mu_v, self.log_var_v, self.u_v)
                v_sample = SobolevGrad.apply(v_sample, self.S, self.padding_sz)

                transformation, displacement = self.transformation_model(v_sample)
                im_moving_warped = self.registration_module(im_moving, transformation)

                n_F, n_M = self.data_loss.map(self.im_fixed, im_moving_warped)
                res = n_F - n_M
                res_masked = res[self.mask_fixed]

                res_mean = torch.mean(res_masked)
                res_var = torch.mean(torch.pow(res_masked - res_mean, 2))
                res_std = torch.sqrt(res_var)

                self.data_loss.init_parameters(res_std)
                alpha = 1.0

            if self.virutal_decimation:
                res_rescaled = rescale_residuals(res, self.mask_fixed, self.data_loss)
                alpha = VD(res_rescaled, self.mask_fixed)

            # Gaussian mixture
            self._step_GMM(res_masked, alpha)

            # losses and metrics before registration
            with torch.no_grad():
                loss_unwarped = self.data_loss(res_masked) * alpha
                self.logger.info(f'PRE-REGISTRATION: {loss_unwarped.item():.5f}\n')

                iter_no = 0
                self.writer.set_step(iter_no)

                self.metrics_VI.update('VI/train/data_term', loss_unwarped.item())
                log_hist_res(self.writer, im_pair_idxs, res_masked, self.data_loss)  # residual histogram

                # Dice scores
                DSC = calc_DSC(self.seg_fixed, seg_moving, self.structures_dict)

                for structure in DSC:
                    score = DSC[structure]
                    self.metrics_VI.update('VI/train/DSC/' + structure, score)

                # average surface distances
                ASD = calc_ASD(self.seg_fixed, seg_moving, self.structures_dict, self.data_loader.spacing)

                for structure in ASD:
                    dist = ASD[structure]
                    self.metrics_VI.update('VI/train/ASD/' + structure, dist)

            """
            VI
            """

            if self.VI:
                # train
                start = datetime.now()
                self._run_VI(im_pair_idxs, im_moving, mask_moving, seg_moving)
                stop = datetime.now()

                VI_time = (stop - start).total_seconds()
                self.logger.info(f'VI took {VI_time:.2f} seconds')

                # test
                self._test_VI(im_pair_idxs, im_moving, mask_moving, seg_moving)

            """
            MCMC
            """

            if self.MCMC:
                with torch.no_grad():
                    self.SGLD_params['tau'] = self.config['optimizer_SG_MCMC']['args']['lr']

                    self.SGLD_params['sigma'] = torch.exp(0.5 * self.log_var_v).detach().clone()
                    self.SGLD_params['sigma'].requires_grad_(False)

                    self.SGLD_params['u'] = self.u_v.detach().clone()
                    self.SGLD_params['u'].requires_grad_(False)

                    if self.v_curr_state is None:
                        self.v_curr_state = sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=1).detach()

                self._run_MCMC(im_pair_idxs, im_moving, mask_moving, seg_moving)

    def _save_checkpoint_VI(self, iter_no):
        """
        save a checkpoint (variational inference)
        """

        state = {
            'config': self.config,
            'iter_no': iter_no,
            'sample_no': self.start_sample,

            'mu_v': self.mu_v,
            'log_var_v': self.log_var_v,
            'u_v': self.u_v,
            'optimizer_v': self.optimizer_v.state_dict(),

            'data_loss': self.data_loss.state_dict(),
            'optimizer_GMM': self.optimizer_GMM.state_dict(),
            'reg_loss': self.reg_loss.state_dict(),
        }

        if self.reg_loss_type is not 'RegLoss_L2':
            state['reg_loss_loc_prior'] = self.reg_loss_loc_prior.state_dict()
            state['reg_loss_scale_prior'] = self.reg_loss_scale_prior.state_dict()
            state['optimizer_w_reg'] = self.optimizer_w_reg.state_dict()

        filename = str(self.checkpoint_dir / f'checkpoint_VI_{iter_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _save_checkpoint_MCMC(self, sample_no):
        """
        save a checkpoint (Markov chain Monte Carlo)
        """

        state = {
            'config': self.config,
            'iter_no': self.no_iters_VI,
            'sample_no': sample_no,

            'mu_v': self.mu_v,
            'log_var_v': self.log_var_v,
            'u_v': self.u_v,

            'data_loss': self.data_loss.state_dict(),
            'optimizer_GMM': self.optimizer_GMM.state_dict(),
            'reg_loss': self.reg_loss.state_dict(),

            'v_curr_state': self.v_curr_state,
            'optimizer_SG_MCMC': self.optimizer_SG_MCMC.state_dict()
        }

        if self.reg_loss_type is not 'RegLoss_L2':
            state['reg_loss_loc_prior'] = self.reg_loss_loc_prior.state_dict()
            state['reg_loss_scale_prior'] = self.reg_loss_scale_prior.state_dict()
            state['optimizer_w_reg'] = self.optimizer_w_reg.state_dict()

        filename = str(self.checkpoint_dir / f'checkpoint_MCMC_{sample_no}.pth')
        self.logger.info("saving checkpoint: {}..".format(filename))
        torch.save(state, filename)
        self.logger.info("checkpoint saved\n")

    def _resume_checkpoint_VI(self, resume_path):
        """
        resume from saved checkpoints (VI)
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {}..".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_iter = checkpoint['iter_no'] + 1
        self.start_sample = checkpoint['sample_no'] + 1

        # VI
        self.mu_v = checkpoint['mu_v']
        self.log_var_v = checkpoint['log_var_v']
        self.u_v = checkpoint['u_v']

        self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [self.mu_v, self.log_var_v, self.u_v])
        self.optimizer_v.load_state_dict(checkpoint['optimizer_v'])

        # GMM
        self.data_loss.load_state_dict(checkpoint['data_loss'])

        self.__init_optimizer_GMM()
        self.optimizer_GMM.load_state_dict(checkpoint['optimizer_GMM'])

        # regularisation loss
        self.reg_loss.load_state_dict(checkpoint['reg_loss'])

        if self.reg_loss_type is not 'RegLoss_L2':
            self.reg_loss_loc_prior.load_state_dict(checkpoint['reg_loss_loc_prior'])
            self.reg_loss_scale_prior.load_state_dict(checkpoint['reg_loss_scale_prior'])

            self.__init_optimizer_w_reg()
            self.optimizer_w_reg.load_state_dict(checkpoint['optimizer_w_reg'])

        self.logger.info("checkpoint loaded, resuming training..")

    def _resume_checkpoint_MCMC(self, resume_path):
        """
        resume from saved checkpoints (MCMC)
        """

        resume_path = str(resume_path)
        self.logger.info("\nloading checkpoint: {}..".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_iter = checkpoint['iter_no'] + 1
        self.start_sample = checkpoint['sample_no'] + 1

        # VI
        self.mu_v = checkpoint['mu_v']
        self.log_var_v = checkpoint['log_var_v']
        self.u_v = checkpoint['u_v']

        # GMM
        self.data_loss.load_state_dict(checkpoint['data_loss'])

        self.__init_optimizer_GMM()
        self.optimizer_GMM.load_state_dict(checkpoint['optimizer_GMM'])

        # regularisation loss
        self.reg_loss.load_state_dict(checkpoint['reg_loss'])

        if self.reg_loss_type is not 'RegLoss_L2':
            self.reg_loss_loc_prior.load_state_dict(checkpoint['reg_loss_loc_prior'])
            self.reg_loss_scale_prior.load_state_dict(checkpoint['reg_loss_scale_prior'])

            self.__init_optimizer_w_reg()
            self.optimizer_w_reg.load_state_dict(checkpoint['optimizer_w_reg'])

        # MCMC
        with torch.no_grad():
            self.v_curr_state = checkpoint['v_curr_state'] if 'v_curr_state' in checkpoint \
                else sample_q_v(self.mu_v, self.log_var_v, self.u_v, no_samples=1).detach()

        if 'optimizer_SG_MCMC' in checkpoint:
            self.optimizer_SG_MCMC = self.config.init_obj('optimizer_SG_MCMC', torch.optim, [self.v_curr_state])
            self.optimizer_SG_MCMC.load_state_dict(checkpoint['optimizer_SG_MCMC'])

        self.logger.info("checkpoint loaded, resuming training..")
