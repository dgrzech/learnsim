import torch
import torch.distributed as dist
from tqdm import tqdm, trange

from base import BaseTrainer
from logger import log_fields, log_images, log_model_samples, log_model_weights, save_sample
from utils import SGLD, SobolevGrad, Sobolev_kernel_1D, \
    add_noise_uniform_field, calc_metrics, calc_no_non_diffeomorphic_voxels, sample_q_v


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, config, data_loader, model, losses, transformation_module, registration_module, metrics, test_only=False):
        super().__init__(config, data_loader, model, losses, transformation_module, registration_module, metrics, test_only)

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

        if self.atlas_mode and not self.test_only:
            self.__metrics_init()

    def __calc_data_loss(self, fixed, moving, v_sample_unsmoothed):
        v_sample = SobolevGrad.apply(v_sample_unsmoothed, self.S, self.padding)
        transformation, displacement = self.transformation_module(v_sample)

        if self.add_noise_uniform:
            transformation = add_noise_uniform_field(transformation, self.alpha)

        im_moving_warped = self.registration_module(moving['im'], transformation)
        z = self.model(fixed['im'], im_moving_warped, fixed['mask'])
        data_term = self.data_loss(z)

        return data_term, im_moving_warped, v_sample, transformation, displacement

    def __calc_loss(self, fixed, moving, v_sample_unsmoothed, var_params_q_v):
        data_term, im_moving_warped, v_sample, transformation, displacement = self.__calc_data_loss(fixed, moving, v_sample_unsmoothed)
        reg_term = self.reg_loss(v_sample)
        entropy_term = self.entropy_loss(sample=v_sample_unsmoothed, mu=var_params_q_v['mu'], log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])

        return data_term, reg_term, entropy_term, transformation, displacement, im_moving_warped

    def __generate_samples_from_model(self, im_pair_idxs, fixed, im_moving_warped):
        n = len(im_pair_idxs)
        total_no_samples = n * self.world_size

        self.curr_state = fixed['im'].detach().clone()
        self.curr_state.requires_grad_(True)
        self.__init_optimizer_LD()

        mean = torch.zeros_like(self.curr_state)

        for iter_no in range(1, self.no_samples_SGLD):
            self.curr_state = SGLD.apply(self.curr_state, torch.ones_like(self.curr_state), self.tau)
            mean += self.curr_state.detach() / self.no_samples_SGLD

            z_curr_state = self.model(self.curr_state, im_moving_warped, fixed['mask'])
            loss = self.data_loss(z_curr_state)

            self.optimizer_LD.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer_LD.step()

            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                with torch.no_grad():
                    self.writer.set_step(self.step + iter_no)
                    self.metrics.update('loss/neg_sample_energy', loss.item(), n=total_no_samples)

                    if iter_no % self.log_period_model_samples == 0:
                        log_model_samples(self.writer, im_pair_idxs, self.curr_state.detach())

        return mean

    def _step_q_v(self, epoch, im_pair_idxs, fixed, moving, var_params_q_v):
        im_pair_idxs_local = im_pair_idxs

        n = len(im_pair_idxs_local)
        total_no_samples = n * self.world_size

        for _ in trange(1, self.no_iters_q_v + 1, desc=f'batch {batch_idx+1}', colour='#808080', disable=self.tqdm_disable, dynamic_ncols=True, leave=False):
            self.step += 1

            # get samples from q_v
            v_sample1_unsmoothed, v_sample2_unsmoothed = sample_q_v(var_params_q_v, no_samples=2)

            # calculate the loss
            data_term1, reg_term1, entropy_term1, transformation1, displacement1, im_moving_warped1 = self.__calc_loss(fixed, moving, v_sample1_unsmoothed, var_params_q_v)
            data_term2, reg_term2, entropy_term2, _, _, _ = self.__calc_loss(fixed, moving, v_sample2_unsmoothed, var_params_q_v)

            data_term = data_term1.sum() + data_term2.sum()
            reg_term = reg_term1.sum() + reg_term2.sum()

            entropy_term3 = self.entropy_loss(log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])
            entropy_term = entropy_term1.sum() / 2.0 + entropy_term2.sum() / 2.0 + entropy_term3.sum()

            loss_q_v = data_term / 2.0 + reg_term / 2.0 - entropy_term

            # backprop
            self.optimizer_q_v.zero_grad(set_to_none=True)
            loss_q_v.backward()
            self.optimizer_q_v.step()

            with torch.no_grad():
                im_pair_idxs_tensor = torch.tensor(im_pair_idxs, device=self.rank)
                im_pair_idxs_tensor_list = [torch.zeros_like(im_pair_idxs_tensor) for _ in range(self.world_size)]
                dist.all_gather(im_pair_idxs_tensor_list, im_pair_idxs_tensor)

                no_non_diffeomorphic_voxels, log_det_J_transformation = calc_no_non_diffeomorphic_voxels(transformation1, self.diff_op)
                no_non_diffeomorphic_voxels_list = [torch.zeros_like(no_non_diffeomorphic_voxels) for _ in range(self.world_size)]
                dist.all_gather(no_non_diffeomorphic_voxels_list, no_non_diffeomorphic_voxels)

                dist.reduce(data_term, 0, op=dist.ReduceOp.SUM)
                dist.reduce(reg_term, 0, op=dist.ReduceOp.SUM)
                dist.reduce(entropy_term, 0, op=dist.ReduceOp.SUM)
                dist.reduce(loss_q_v, 0, op=dist.ReduceOp.SUM)

                # tensorboard
                self.writer.set_step(self.step)

                self.metrics.update('loss/data_term', data_term.item(), n=total_no_samples)
                self.metrics.update('loss/reg_term', reg_term.item(), n=total_no_samples)
                self.metrics.update('loss/entropy_term', entropy_term.item(), n=total_no_samples)
                self.metrics.update('loss/q_v', loss_q_v.item(), n=total_no_samples)

                im_pair_idxs_list = torch.cat(im_pair_idxs_tensor_list, dim=0).view(-1).tolist()
                no_non_diffeomorphic_voxels = torch.cat(no_non_diffeomorphic_voxels_list, dim=0).view(-1).tolist()

                for loop_idx, im_pair_idx in enumerate(im_pair_idxs_list):
                    no_non_diffeomorphic_voxels_im_pair = no_non_diffeomorphic_voxels[loop_idx]
                    self.metrics.update(f'no_non_diffeomorphic_voxels/im_pair_{im_pair_idx}', no_non_diffeomorphic_voxels_im_pair)

        # tensorboard cont.
        with torch.no_grad():
            batch_size = moving['im'].shape[0]
            grid_im1 = self.grid_im.clone().repeat(batch_size, 1, 1, 1, 1)
            grid_im1 = self.registration_module(grid_im1, transformation1)

            segs_moving_warped = self.registration_module(moving['seg'], transformation1)
            ASD, DSC = calc_metrics(im_pair_idxs_local, fixed['seg'], segs_moving_warped, self.structures_dict, self.im_spacing)

            ASD_list = [torch.zeros_like(ASD) for _ in range(self.world_size)]
            DSC_list = [torch.zeros_like(DSC) for _ in range(self.world_size)]
            dist.all_gather(ASD_list, ASD)
            dist.all_gather(DSC_list, DSC)

            self.writer.set_step(epoch)
            self.metrics.update_ASD_and_DSC(im_pair_idxs_list, self.structures_dict, ASD_list, DSC_list)

            if not self.test_only:
                var_params_q_v_smoothed = self.__get_var_params_smoothed(var_params_q_v)
                log_fields(self.writer, im_pair_idxs_local, var_params_q_v_smoothed, displacement1, grid_im1, log_det_J_transformation)
                log_images(self.writer, im_pair_idxs_local, fixed['im'], moving['im'], im_moving_warped1)

    def _step_q_phi(self, im_pair_idxs, fixed, moving, var_params_q_v):
        self.step += 1

        n = len(im_pair_idxs)
        total_no_samples = self.world_size

        # draw a sample from q_v
        v_sample = sample_q_v(var_params_q_v, no_samples=1)
        term1, im_moving_warped1, _, _, _ = self.__calc_data_loss(fixed, moving, v_sample)

        # implicit generation from the energy-based model
        im_fixed_sample = self.__generate_samples_from_model(im_pair_idxs, fixed, im_moving_warped1)
        term2, _, _, _, _ = self.__calc_data_loss({'im': im_fixed_sample, 'mask': fixed['mask']}, moving, v_sample)

        loss_q_phi = term1.sum() - term2.sum()
        loss_q_phi /= n
        
        self.optimizer_q_f.zero_grad(set_to_none=True)
        self.optimizer_q_phi.zero_grad(set_to_none=True)
        loss_q_phi.backward()  # backprop
        self.optimizer_q_f.step()
        self.optimizer_q_phi.step()

        dist.reduce(loss_q_phi, 0, op=dist.ReduceOp.SUM)

        # tensorboard
        self.writer.set_step(self.step)
        self.metrics.update('loss/q_phi', loss_q_phi.item(), n=total_no_samples)

        with torch.no_grad():
            log_model_weights(self.writer, self.model)

    def _train_epoch(self, epoch=0):
        self.data_loader.sampler.set_epoch(epoch)
        self.metrics.reset()

        for batch_idx, (im_pair_idxs, moving, var_params_q_v) in enumerate(tqdm(self.data_loader, desc=f'epoch {epoch}', disable=self.tqdm_disable, dynamic_ncols=True, unit='batch')):
            self.logger.debug(f'epoch {epoch}, processing batch {batch_idx+1} out of {self.no_batches}..')

            im_pair_idxs = im_pair_idxs.tolist()
            self.__fixed_and_moving_init(fixed, moving, var_params_q_v)

            """
            q_v
            """

            self._enable_gradients_variational_parameters(var_params_q_v)
            self.__init_optimizer_q_v(var_params_q_v)
            self._step_q_v(epoch, batch_idx, im_pair_idxs, fixed, moving, var_params_q_v)
            self._disable_gradients_variational_parameters(var_params_q_v)

            """
            q_phi
            """

            if not self.test_only:
                self._enable_gradients_model()
                self._step_q_phi(im_pair_idxs, fixed, moving, var_params_q_v)
                self._disable_gradients_model()

        if not self.test_only:
            self._save_checkpoint(epoch)
            self.writer.set_step(epoch)
            self.metrics.update_avg_metrics(self.structures_dict)

    @torch.no_grad()
    def _test(self, no_samples):
        self.logger.debug('')
        
        for batch_idx, (im_pair_idxs, moving, var_params_q_v) in enumerate(tqdm(self.data_loader, desc='testing', disable=self.tqdm_disable, dynamic_ncols=True, unit='batch')):
                self.logger.debug(f'testing, processing batch {batch_idx+1} out of {self.no_batches}..')

                im_pair_idxs = im_pair_idxs.tolist()
                im_pair_idxs_local = im_pair_idxs
                self.__fixed_and_moving_init(fixed, moving, var_params_q_v)

                for sample_no in range(1, no_samples+1):
                    v_sample = sample_q_v(var_params_q_v, no_samples=1)
                    v_sample_smoothed = SobolevGrad.apply(v_sample, self.S, self.padding)
                    transformation, displacement = self.transformation_module(v_sample_smoothed)

                    im_moving_warped = self.registration_module(moving['im'], transformation)
                    segs_moving_warped = self.registration_module(moving['seg'], transformation)
                    save_sample(im_pair_idxs_local, self.save_dirs, self.im_spacing, sample_no, im_moving_warped, displacement)

                    im_pair_idxs_tensor = torch.tensor(im_pair_idxs, device=self.rank)
                    im_pair_idxs_tensor_list = [torch.zeros_like(im_pair_idxs_tensor) for _ in range(self.world_size)]
                    dist.all_gather(im_pair_idxs_tensor_list, im_pair_idxs_tensor)

                    no_non_diffeomorphic_voxels, _ = calc_no_non_diffeomorphic_voxels(transformation, self.diff_op)
                    no_non_diffeomorphic_voxels_list = [torch.zeros_like(no_non_diffeomorphic_voxels) for _ in range(self.world_size)]
                    dist.all_gather(no_non_diffeomorphic_voxels_list, no_non_diffeomorphic_voxels)

                    ASD, DSC = calc_metrics(im_pair_idxs_local, fixed['seg'], segs_moving_warped, self.structures_dict, self.im_spacing)

                    ASD_list = [torch.zeros_like(ASD) for _ in range(self.world_size)]
                    DSC_list = [torch.zeros_like(DSC) for _ in range(self.world_size)]
                    dist.all_gather(ASD_list, ASD)
                    dist.all_gather(DSC_list, DSC)

                    im_pair_idxs_list = torch.cat(im_pair_idxs_tensor_list, dim=0).view(-1).tolist()
                    no_non_diffeomorphic_voxels = torch.cat(no_non_diffeomorphic_voxels_list, dim=0).view(-1).tolist()

                    self.writer.set_step(sample_no)

                    for loop_idx, im_pair_idx in enumerate(im_pair_idxs_list):
                        no_non_diffeomorphic_voxels_im_pair = no_non_diffeomorphic_voxels[loop_idx]

                        self.metrics.update(f'test/no_non_diffeomorphic_voxels/im_pair_{im_pair_idx}', no_non_diffeomorphic_voxels_im_pair)
                        self.metrics.update_ASD_and_DSC(im_pair_idxs_list, self.structures_dict, ASD_list, DSC_list, test=True)

    @torch.no_grad()
    def __fixed_and_moving_init(self, fixed, moving, var_params_q_v):
        for key in fixed:
            fixed[key] = fixed[key].to(self.rank, memory_format=torch.channels_last_3d)
        for key in moving:
            moving[key] = moving[key].to(self.rank, memory_format=torch.channels_last_3d)

        for param_key in var_params_q_v:
            var_params_q_v[param_key] = var_params_q_v[param_key].to(self.rank)

    @torch.no_grad()
    def __metrics_init(self):
        self.writer.set_step(self.step)

        for batch_idx, (im_pair_idxs, moving, var_params_q_v) in enumerate(tqdm(self.data_loader, desc='pre-registration metrics', disable=self.tqdm_disable, dynamic_ncols=True, unit='batch')):
            im_pair_idxs_local = im_pair_idxs.tolist()
            self.__fixed_and_moving_init(fixed, moving, var_params_q_v)

            ASD, DSC = calc_metrics(im_pair_idxs_local, fixed['seg'], moving['seg'], self.structures_dict, self.im_spacing)

            im_pair_idxs_tensor = torch.tensor(im_pair_idxs_local, device=self.rank)
            im_pair_idxs_tensor_list = [torch.zeros_like(im_pair_idxs_tensor) for _ in range(self.world_size)]
            dist.all_gather(im_pair_idxs_tensor_list, im_pair_idxs_tensor)

            ASD_list = [torch.zeros_like(ASD) for _ in range(self.world_size)]
            DSC_list = [torch.zeros_like(DSC) for _ in range(self.world_size)]
            dist.all_gather(ASD_list, ASD)
            dist.all_gather(DSC_list, DSC)

            im_pair_idxs_list = torch.cat(im_pair_idxs_tensor_list, dim=0).view(-1).tolist()
            self.metrics.update_ASD_and_DSC(im_pair_idxs_list, self.structures_dict, ASD_list, DSC_list)

        self.metrics.update_avg_metrics(self.structures_dict)

    @torch.no_grad()
    def __Sobolev_gradients_init(self):
        _s = self.config['Sobolev_grad']['s']
        _lambda = self.config['Sobolev_grad']['lambda']
        padding_sz = _s

        S, S_sqrt = Sobolev_kernel_1D(_s, _lambda)
        S = torch.from_numpy(S).float().unsqueeze(0)
        S = torch.stack((S, S, S), 0)

        S_x = S.unsqueeze(2).unsqueeze(2).to(self.rank)
        S_y = S.unsqueeze(2).unsqueeze(4).to(self.rank)
        S_z = S.unsqueeze(3).unsqueeze(4).to(self.rank)

        self.padding = (padding_sz, ) * 6
        self.S = {'x': S_x, 'y': S_y, 'z': S_z}

    def __get_var_params_smoothed(self, var_params):
        return {k: SobolevGrad.apply(v, self.S, self.padding) for k, v in var_params.items()}

    def __init_optimizer_q_v(self, var_params_q_v):
        trainable_params_q_v = filter(lambda p: p.requires_grad, var_params_q_v.values())
        self.optimizer_q_v = self.config.init_obj('optimizer_q_v', torch.optim, trainable_params_q_v)
    
    def __init_optimizer_LD(self):
        self.optimizer_LD = self.config.init_obj('optimizer_LD', torch.optim, self.curr_state)

