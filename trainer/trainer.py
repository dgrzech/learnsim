from os import path

import torch
from torch import nn

from base import BaseTrainer
from logger import log_fields, log_images, log_q_f, save_optimizer, save_tensors
from utils import MetricTracker, SobolevGrad, Sobolev_kernel_1D, add_noise_uniform, calc_det_J, calc_metrics, sample_q_f, sample_q_v


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, config, data_loader, model, losses, transformation_model, registration_module, metrics, structures_dict):
        super().__init__(config, data_loader, model, losses, transformation_model, registration_module)

        # all-to-one registration
        fixed = self.data_loader.fixed
        var_params_q_f = self.data_loader.var_params_q_f

        self.fixed = dict()
        self.var_params_q_f = dict()

        for key in fixed:
            self.fixed[key] = fixed[key].to(self.device, non_blocking=True)

        for key in var_params_q_f:
            parameter = var_params_q_f[key].to(self.device, non_blocking=True)
            self.var_params_q_f[key] = nn.Parameter(parameter)

        # optimizers
        self._init_optimizers()

        # Sobolev gradients
        self.Sobolev_grad = config['Sobolev_grad']['enabled']

        if self.Sobolev_grad:
            self.__Sobolev_gradients_init()

        # uniform noise magnitude
        cfg_trainer = config['trainer']
        self.add_noise_uniform = cfg_trainer['uniform_noise']['enabled']

        if self.add_noise_uniform:
            self.alpha = cfg_trainer['uniform_noise']['magnitude']

        # metrics
        self.metrics = MetricTracker(*[m for m in metrics], writer=self.writer)
        self.structures_dict = structures_dict
        self.__pre_registration()

    def __calc_sample_loss(self, moving, v_sample_unsmoothed, var_params_q_v, im_fixed_sample=None):
        v_sample = SobolevGrad.apply(v_sample_unsmoothed, self.S, self.padding_sz)
        transformation, displacement = self.transformation_model(v_sample)

        if self.add_noise_uniform:
            transformation = add_noise_uniform(transformation, self.alpha)

        im_fixed = self.fixed['im'] if im_fixed_sample is None else im_fixed_sample
        im_moving_warped = self.registration_module(moving['im'], transformation)

        z = self.model(im_fixed, im_moving_warped, self.fixed['mask'])
        data_term = self.data_loss(z)

        if im_fixed_sample is not None:
            return data_term

        reg_term = self.reg_loss(v_sample)
        entropy_term = self.entropy_loss(sample=v_sample_unsmoothed, mu=var_params_q_v['mu'], log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])

        return data_term, reg_term, entropy_term, transformation, displacement, im_moving_warped

    def _step_q_v(self, im_pair_idxs, moving, var_params_q_v):
        for iter_no in range(1, self.no_iters_q_v + 1):
            n = len(im_pair_idxs)
            self.step_global += 1

            # get samples from q_v
            v_sample1_unsmoothed, v_sample2_unsmoothed = sample_q_v(var_params_q_v, no_samples=2)

            # calculate the loss
            data_term1, reg_term1, entropy_term1, transformation1, displacement1, im_moving_warped1 = self.__calc_sample_loss(moving, v_sample1_unsmoothed, var_params_q_v)
            data_term2, reg_term2, entropy_term2, _, _, _ = self.__calc_sample_loss(moving, v_sample2_unsmoothed, var_params_q_v)

            data_term = data_term1.sum() + data_term2.sum()
            reg_term = reg_term1.sum() + reg_term2.sum()

            entropy_term3 = self.entropy_loss(log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])
            entropy_term = entropy_term1.sum() / 2.0 + entropy_term2.sum() / 2.0 + entropy_term3.sum()

            loss_q_v = data_term / 2.0 + reg_term / 2.0 - entropy_term

            # backprop
            self.optimizer_q_v.zero_grad()
            loss_q_v.backward()
            self.optimizer_q_v.step()

            # tensorboard
            if iter_no % self.log_period == 0:
                self.writer.set_step(self.step_global)

                self.metrics.update('train/loss/data_term', data_term.item(), n=n)
                self.metrics.update('train/loss/reg_term', reg_term.item(), n=n)
                self.metrics.update('train/loss/entropy_term', entropy_term.item(), n=n)
                self.metrics.update('train/loss/q_v', loss_q_v.item(), n=n)

                with torch.no_grad():
                    var_params_q_v_smoothed = dict()

                    for param in var_params_q_v:
                        var_params_q_v_smoothed[param] = SobolevGrad.apply(var_params_q_v[param], self.S, self.padding_sz)

                    nabla = self.diff_op(transformation1, transformation=True)
                    log_det_J_transformation = torch.log(calc_det_J(nabla))
                    no_non_diffeomorphic_voxels = torch.isnan(log_det_J_transformation)

                    log_fields(self.writer, im_pair_idxs, var_params_q_v_smoothed, displacement1, log_det_J_transformation)
                    log_images(self.writer, im_pair_idxs, self.fixed['im'], moving['im'], im_moving_warped1)

                    segs_moving_warped = self.registration_module(moving['seg'], transformation1)
                    ASD, DSC = calc_metrics(im_pair_idxs, self.fixed['seg'], segs_moving_warped, self.structures_dict, self.data_loader.spacing)

                    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
                        no_non_diffeomorphic_voxels_im_pair = no_non_diffeomorphic_voxels[loop_idx].sum().item()
                        self.metrics.update('train/no_non_diffeomorphic_voxels/im_pair_' + str(im_pair_idx), no_non_diffeomorphic_voxels_im_pair)

                        for structure in self.structures_dict:
                            ASD_val = ASD[im_pair_idx][structure]
                            DSC_val = DSC[im_pair_idx][structure]

                            self.metrics.update('train/ASD/im_pair_' + str(im_pair_idx) + '/' + structure, ASD_val)
                            self.metrics.update('train/DSC/im_pair_' + str(im_pair_idx) + '/' + structure, DSC_val)

    def _step_q_f_q_phi(self, im_pair_idxs, moving, var_params_q_v):
        self.step_global += 1

        # draw a sample from q_v
        v_sample = sample_q_v(var_params_q_v, no_samples=1)
        term1 = self.__calc_sample_loss(moving, v_sample, var_params_q_v, im_fixed_sample=self.fixed['im'])

        # draw samples from q_f
        im_fixed_sample1, im_fixed_sample2 = sample_q_f(self.fixed['im'], self.var_params_q_f, no_samples=2)

        term2 = self.__calc_sample_loss(moving, v_sample, var_params_q_v, im_fixed_sample=im_fixed_sample1)
        term3 = self.__calc_sample_loss(moving, v_sample, var_params_q_v, im_fixed_sample=im_fixed_sample2)

        loss_q_f_q_phi = term1.sum() - term2.sum() / 2.0 - term3.sum() / 2.0
        n = len(im_pair_idxs)
        loss_q_f_q_phi /= n

        if self.optimize_q_f:
            self.optimizer_q_f.zero_grad()
        if self.optimize_q_phi:
            self.optimizer_q_phi.zero_grad()

        loss_q_f_q_phi.backward()  # backprop

        if self.optimize_q_f:
            self.optimizer_q_f.step()
        if self.optimize_q_phi:
            self.optimizer_q_phi.step()

        # tensorboard
        self.writer.set_step(self.step_global)

        with torch.no_grad():
            self.metrics.update('train/loss/q_f_q_phi', loss_q_f_q_phi.item(), n=n)
            log_q_f(self.writer, self.var_params_q_f)

    def _train_epoch(self, epoch):
        self.metrics.reset()

        for batch_idx, (im_pair_idxs, moving, var_params_q_v) in enumerate(self.data_loader):
            im_pair_idxs = im_pair_idxs.tolist()

            for key in moving:
                moving[key] = moving[key].to(self.device, non_blocking=True)
            for param_key in var_params_q_v:
                var_params_q_v[param_key] = var_params_q_v[param_key].to(self.device, non_blocking=True)

            """
            q_v
            """

            if self.optimize_q_v:
                self._enable_gradients_variational_parameters(var_params_q_v)
                self.__init_optimizer_q_v(batch_idx, var_params_q_v)
                self._step_q_v(im_pair_idxs, moving, var_params_q_v)
                self._disable_gradients_variational_parameters(var_params_q_v)

                save_tensors(im_pair_idxs, self.data_loader.save_dirs, var_params_q_v)
                save_optimizer(batch_idx, self.data_loader.save_dirs, self.optimizer_q_v, 'optimizer_q_v')

            """
            q_f and q_phi
            """

            if self.optimize_q_f or self.optimize_q_phi:
                if self.optimize_q_f:
                    self._enable_gradients_variational_parameters(self.var_params_q_f)
                if self.optimize_q_phi:
                    self.model.enable_gradients()

                self._step_q_f_q_phi(im_pair_idxs, moving, var_params_q_v)

                if self.optimize_q_f:
                    self._disable_gradients_variational_parameters(self.var_params_q_f)
                if self.optimize_q_phi:
                    self.model.disable_gradients()

        self._save_checkpoint(epoch)

    def __pre_registration(self):
        self.writer.set_step(self.step_global)

        for batch_idx, (im_pair_idxs, moving, var_params_q_v) in enumerate(self.data_loader):
            im_pair_idxs = im_pair_idxs.tolist()

            for key in moving:
                moving[key] = moving[key].to(self.device, non_blocking=True)
            for param_key in var_params_q_v:
                var_params_q_v[param_key] = var_params_q_v[param_key].to(self.device, non_blocking=True)

            with torch.no_grad():
                if batch_idx == 0:
                    if self.optimize_q_f:
                        log_q_f(self.writer, self.var_params_q_f)

                ASD, DSC = calc_metrics(im_pair_idxs, self .fixed['seg'], moving['seg'], self.structures_dict, self.data_loader.spacing)

                for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
                    for structure in self.structures_dict:
                        ASD_val = ASD[im_pair_idx][structure]
                        DSC_val = DSC[im_pair_idx][structure]

                        self.metrics.update('train/ASD/im_pair_' + str(im_pair_idx) + '/' + structure, ASD_val)
                        self.metrics.update('train/DSC/im_pair_' + str(im_pair_idx) + '/' + structure, DSC_val)

    def __Sobolev_gradients_init(self):
        _s = self.config['Sobolev_grad']['s']
        _lambda = self.config['Sobolev_grad']['lambda']
        self.padding_sz = _s // 2

        S, S_sqrt = Sobolev_kernel_1D(_s, _lambda)
        S = torch.from_numpy(S).float().unsqueeze(0)
        S = torch.stack((S, S, S), 0)

        S_x = S.unsqueeze(2).unsqueeze(2).to(self.device, non_blocking=True)
        S_y = S.unsqueeze(2).unsqueeze(4).to(self.device, non_blocking=True)
        S_z = S.unsqueeze(3).unsqueeze(4).to(self.device, non_blocking=True)

        self.S = {'x': S_x, 'y': S_y, 'z': S_z}

    def _init_optimizers(self):
        if self.optimize_q_v:
            self.optimizer_q_v = None

        if self.optimize_q_f:
            self._enable_gradients_variational_parameters(self.var_params_q_f)
            self.__init_optimizer_q_f()
            self._disable_gradients_variational_parameters(self.var_params_q_f)

        if self.optimize_q_phi:
            self.model.enable_gradients()
            self.__init_optimizer_q_phi()
            self.model.disable_gradients()

    def __init_optimizer_q_f(self):
        trainable_params_q_f = filter(lambda p: p.requires_grad, self.var_params_q_f.values())
        self.optimizer_q_f = self.config.init_obj('optimizer_q_f', torch.optim, trainable_params_q_f)

    def __init_optimizer_q_phi(self):
        trainable_params_q_phi = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer_q_phi = self.config.init_obj('optimizer_q_phi', torch.optim, trainable_params_q_phi)

    def __init_optimizer_q_v(self, batch_idx, var_params_q_v):
        trainable_params_q_v = filter(lambda p: p.requires_grad, var_params_q_v.values())
        self.optimizer_q_v = self.config.init_obj('optimizer_q_v', torch.optim, trainable_params_q_v)

        optimizer_path = path.join(self.data_loader.save_dirs['optimizers'], 'optimizer_q_v_' + str(batch_idx) + '.pt')

        if path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path)
            self.optimizer_q_v.load_state_dict(checkpoint)
