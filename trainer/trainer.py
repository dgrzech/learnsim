import torch
from tqdm import tqdm, trange

from base import BaseTrainer
from logger import log_images, log_model_samples, log_model_weights, save_sample, save_var_params
from utils import SGLD, SobolevGrad, Sobolev_kernel_1D, \
    add_noise_uniform_field, calc_metrics, calc_no_non_diffeomorphic_voxels, sample_q_v


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, config, data_loader, model, losses, transformation_module, registration_module, metrics, is_test=False):
        super().__init__(config, data_loader, model, losses, transformation_module, registration_module, metrics, is_test)

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

    def __calc_data_loss(self, fixed, moving, sample_v=None):
        output_dict = dict()

        if sample_v is not None:
            sample_v_smoothed = SobolevGrad.apply(sample_v, self.S, self.padding)
            transformation, displacement = self.transformation_module(sample_v_smoothed)
            transformation = add_noise_uniform_field(transformation, self.alpha) if self.add_noise_uniform else transformation

            im_moving_warped = self.registration_module(moving['im'], transformation)
            output_dict = {'sample_displacement': displacement, 'sample_transformation': transformation, 'sample_v': sample_v_smoothed}
        else:
            im_moving_warped = moving['im']

        output_dict = {**output_dict, 'sample_moving_warped': im_moving_warped}

        z = self.model(fixed['im'], im_moving_warped, fixed['mask'])
        data_term = self.data_loss(z)

        return data_term, output_dict

    def __calc_loss(self, fixed, moving, sample_v_unsmoothed, var_params_q_v):
        data_term, output_dict = self.__calc_data_loss(fixed, moving, sample_v=sample_v_unsmoothed)
        regularisation_term = self.reg_loss(output_dict['sample_v'])
        entropy_term = self.entropy_loss(sample=sample_v_unsmoothed, mu=var_params_q_v['mu'], log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])
        entropy_term += self.entropy_loss(log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])

        loss_terms_dict = {'data_term': data_term,
                           'regularisation_term': regularisation_term,
                           'negative_entropy_term': -1.0 * entropy_term}

        return loss_terms_dict, output_dict

    def __generate_samples_from_model(self, im_pair_idxs, fixed, moving, var_params_q_v):
        n = len(im_pair_idxs)

        with torch.no_grad():
            sample_v = sample_q_v(var_params_q_v, no_samples=1)
            sample_v_smoothed = SobolevGrad.apply(sample_v, self.S, self.padding)
            transformation, displacement = self.transformation_module(sample_v_smoothed)
            transformation = add_noise_uniform_field(transformation, self.alpha) if self.add_noise_uniform else transformation
            im_moving_warped = self.registration_module(moving['im'], transformation)

        self.x_plus, self.x_minus = fixed['im'].detach().clone(), fixed['im'].detach().clone()
        self.x_plus.requires_grad_(True), self.x_minus.requires_grad_(True)
        self.init_optimizer_LD()

        no_EBM_samples_used = self.no_samples_SGLD - self.no_samples_SGLD_burn_in
        x_plus_mean, x_minus_mean = torch.zeros_like(self.x_plus), torch.zeros_like(self.x_minus)
        sigma = torch.ones_like(self.x_plus)

        for sample_no in trange(1, self.no_samples_SGLD + 1, desc=f'sampling from EBM', colour='#808080', disable=self.tqdm_disable, dynamic_ncols=True, leave=False, unit='sample'):
            self.step += 1

            if sample_no > self.no_samples_SGLD_burn_in:
                x_plus_mean += self.x_plus.detach() / no_EBM_samples_used
                x_minus_mean += self.x_minus.detach() / no_EBM_samples_used

            x_plus, x_minus = SGLD.apply(self.x_plus, sigma, self.tau), SGLD.apply(self.x_minus, sigma, self.tau)
            z_plus, z_minus = self.model(self.x_plus, im_moving_warped, fixed['mask']), self.model(x_minus, im_moving_warped, fixed['mask'])
            loss_plus, loss_minus = self.data_loss(z_plus), -1.0 * self.data_loss(z_minus)

            # FIXME (DG): ugly hack
            loss_plus *= fixed['mask'].sum()
            loss_minus *= fixed['mask'].sum()

            self.optimizer_LD.zero_grad(set_to_none=True)
            loss_plus.backward(), loss_minus.backward()
            self.optimizer_LD.step()

            if self.rank == 0:
                with torch.no_grad():
                    self.writer.set_step(self.step)
                    self.metrics.update('loss/positive_sample_energy', loss_plus.item(), n=n)
                    self.metrics.update('loss/negative_sample_energy', loss_minus.item(), n=n)

        output_dict = {'sample_v': sample_v.detach(),
                       'im_moving_warped': im_moving_warped.detach(),
                       'samples_plus_mean': x_plus_mean.detach(), 'samples_minus_mean': x_minus_mean.detach()}

        return output_dict

    def _step_q_v(self, batch_idx, im_pair_idxs, fixed, moving, var_params_q_v):
        n = len(im_pair_idxs)

        for _ in trange(1, self.no_iters_q_v + 1, desc=f'batch {batch_idx+1}', colour='#808080', disable=self.tqdm_disable, dynamic_ncols=True, leave=False):
            samples_v = sample_q_v(var_params_q_v, no_samples=2)  # get samples from q_v
            loss = 0.0  # calculate the loss

            for sample_v in samples_v:
                loss_dict, output_dict = self.__calc_loss(fixed, moving, sample_v, var_params_q_v)
                loss_sample = 0.0

                for loss_term in loss_dict.values():
                    loss_sample += loss_term

                loss_dict['loss_q_v'] = loss_sample
                loss += loss_sample

            self.optimizer_q_v.zero_grad(set_to_none=True)
            loss.backward()  # backprop
            self.optimizer_q_v.step()

            # tensorboard
            with torch.no_grad():
                self.writer.set_step(self.step)

                for k, v in loss_dict.items():
                    self.metrics.update(f'loss/{k}', v.item(), n=n)

                no_non_diffeomorphic_voxels, log_det_J = calc_no_non_diffeomorphic_voxels(output_dict['sample_transformation'], self.diff_op)
                self.metrics.update(f'non_diffeomorphic_voxels/avg', no_non_diffeomorphic_voxels.mean())

            self.step += 1

        # tensorboard cont.
        with torch.no_grad():
            segs_moving_warped = self.registration_module(moving['seg'], output_dict['sample_transformation'])
            ASD, DSC = calc_metrics(im_pair_idxs, fixed['seg'], segs_moving_warped, self.structures_dict, self.im_spacing)
            self.metrics.update_ASD_and_DSC(self.structures_dict, ASD, DSC)

            if not self.is_test:
                batch_size = moving['im'].shape[0]
                grid_im = self.grid_im.clone().repeat(batch_size, 1, 1, 1, 1)
                grid_im = self.registration_module(grid_im, output_dict['sample_transformation'])
                var_params_q_v_smoothed = self.__get_var_params_smoothed(var_params_q_v)

                output_dict = {**output_dict, **var_params_q_v_smoothed, 'fixed': fixed['im'], 'moving': moving['im'], 'sample_transformation': grid_im}
                log_images(self.writer, output_dict)

    def _step_q_phi(self, im_pair_idxs, fixed, moving, var_params_q_v):
        n = len(im_pair_idxs)

        # implicit generation from the energy-based model
        output_dict = self.__generate_samples_from_model(im_pair_idxs, fixed, moving, var_params_q_v)

        self._enable_gradients_model()

        fixed_plus = {'im': output_dict['samples_plus_mean'], 'mask': fixed['mask']}
        fixed_minus = {'im': output_dict['samples_minus_mean'], 'mask': fixed['mask']}
        moving = {'im': output_dict['im_moving_warped']}

        loss_term1, _ = self.__calc_data_loss(fixed_plus, moving)
        loss_term2, _ = self.__calc_data_loss(fixed_minus, moving)

        alpha = 0.01  # L2 regularisation weight for the energies
        loss_q_phi = loss_term1 - loss_term2 + alpha * (loss_term1 ** 2 + loss_term2 ** 2)
        loss_q_phi /= n
        
        self.optimizer_q_phi.zero_grad(set_to_none=True)
        loss_q_phi.backward()  # backprop
        self.optimizer_q_phi.step()

        self._disable_gradients_model()

        # tensorboard
        with torch.no_grad():
            self.writer.set_step(self.step)
            self.metrics.update('loss/q_phi', loss_q_phi.item())

            log_model_samples(self.writer, output_dict)
            log_model_weights(self.writer, self.model)

        self.step += 1

    def _train_epoch(self, epoch=0):
        if not self.is_test:
            self.data_loader.sampler.set_epoch(epoch)

        self.metrics.reset()

        for batch_idx, (im_pair_idxs, fixed, moving, var_params_q_v) in enumerate(tqdm(self.data_loader, desc=f'epoch {epoch}', disable=self.tqdm_disable, dynamic_ncols=True, unit='batch')):
            self.logger.debug(f'epoch {epoch}, processing batch {batch_idx+1} out of {self.no_batches}..')

            im_pair_idxs = im_pair_idxs.tolist()
            self.__fixed_and_moving_init(fixed, moving, var_params_q_v)

            """
            q_v
            """

            self._enable_gradients_variational_parameters(var_params_q_v)
            self.__init_optimizer_q_v(var_params_q_v)
            self._step_q_v(batch_idx, im_pair_idxs, fixed, moving, var_params_q_v)
            self._disable_gradients_variational_parameters(var_params_q_v)

            if self.is_test == 0:
                self.logger.debug('saving tensors with the variational parameters of q_v..')

            save_var_params(im_pair_idxs, self.save_dirs, var_params_q_v)

            """
            q_phi
            """

            if not self.is_test:
                self._step_q_phi(im_pair_idxs, fixed, moving, var_params_q_v)

        if not self.is_test:
            self._save_checkpoint(epoch)
            self.writer.set_step(epoch)

    @torch.no_grad()
    def _test(self, no_samples):
        self.logger.debug('')
        
        for batch_idx, (im_pair_idxs, fixed, moving, var_params_q_v) in enumerate(tqdm(self.data_loader, desc='testing', disable=self.tqdm_disable, dynamic_ncols=True, unit='batch')):
            self.logger.debug(f'testing, processing batch {batch_idx+1} out of {self.no_batches}..')

            im_pair_idxs = im_pair_idxs.tolist()
            self.__fixed_and_moving_init(fixed, moving, var_params_q_v)

            for sample_no in range(1, no_samples+1):
                sample_v = sample_q_v(var_params_q_v, no_samples=1)
                sample_v_smoothed = SobolevGrad.apply(sample_v, self.S, self.padding)
                transformation, displacement = self.transformation_module(sample_v_smoothed)

                im_moving_warped = self.registration_module(moving['im'], transformation)
                segs_moving_warped = self.registration_module(moving['seg'], transformation)
                save_sample(im_pair_idxs, self.save_dirs, self.im_spacing, sample_no, im_moving_warped, displacement)

                no_non_diffeomorphic_voxels, _ = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)
                ASD, DSC = calc_metrics(im_pair_idxs, fixed['seg'], segs_moving_warped, self.structures_dict, self.im_spacing)

                self.writer.set_step(sample_no)

                for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
                    self.metrics.update(f'test/non_diffeomorphic_voxels/im_pair_{im_pair_idx}', no_non_diffeomorphic_voxels[loop_idx])

                    ASD_im_pair, DSC_im_pair = ASD[loop_idx], DSC[loop_idx]
                    self.metrics.update_ASD_and_DSC(self.structures_dict, ASD_im_pair, DSC_im_pair, im_pair_idx=im_pair_idx)

    @torch.no_grad()
    def __fixed_and_moving_init(self, fixed, moving, var_params_q_v):
        for key in fixed:
            fixed[key] = fixed[key].to(self.rank, memory_format=torch.channels_last_3d)
            moving[key] = moving[key].to(self.rank, memory_format=torch.channels_last_3d)

        for param_key in var_params_q_v:
            var_params_q_v[param_key] = var_params_q_v[param_key].to(self.rank)

    @torch.no_grad()
    def __Sobolev_gradients_init(self):
        cfg_sobolev_grad = self.config['Sobolev_grad']
        _s, _lambda = cfg_sobolev_grad['s'], cfg_sobolev_grad['lambda']
        _padding_sz = _s

        S, _ = Sobolev_kernel_1D(_s, _lambda)
        S = torch.from_numpy(S).float().unsqueeze(0)
        S = torch.stack((S, ) * 3, dim=0)

        S_x = S.unsqueeze(2).unsqueeze(2)
        S_y = S.unsqueeze(2).unsqueeze(4)
        S_z = S.unsqueeze(3).unsqueeze(4)

        self.padding = (_padding_sz, ) * 6
        self.S = dict(zip(['x', 'y', 'z'], [S_x, S_y, S_z]))

        for key in self.S:
            self.S[key] = self.S[key].to(self.rank)

    def __get_var_params_smoothed(self, var_params):
        return {k: SobolevGrad.apply(v, self.S, self.padding) for k, v in var_params.items()}

    def __init_optimizer_q_v(self, var_params_q_v):
        trainable_params_q_v = filter(lambda p: p.requires_grad, var_params_q_v.values())
        self.optimizer_q_v = self.config.init_obj('optimizer_q_v', torch.optim, trainable_params_q_v)
    
    def init_optimizer_LD(self):
        cfg_optimizer_LD = self.config['optimizer_LD']['args']
        self.optimizer_LD = torch.optim.SGD([{'params': [self.x_minus], 'lr': 0.01 * cfg_optimizer_LD['lr']},
                                             {'params': [self.x_plus]}], lr=cfg_optimizer_LD['lr'])
