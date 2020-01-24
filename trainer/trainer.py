from os import path

from base import BaseTrainer
from model.metric import dice
from logger import log_log_det_J_transformation, log_images, log_q_v, log_q_f, registration_print, save_grids, save_images
from utils import get_module_attr, inf_loop, sample_qv, sample_qf, save_optimiser_to_disk, \
    separable_conv_3d, sobolev_kernel_1D, MetricTracker, SobolevGrad

import math
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

        self.sobolev_grad = config['sobolev_grad']['enabled']
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

        self.log_step = config['trainer']['log_step']
        self.train_metrics = MetricTracker('loss', *[m for m in self.metric_ftns], writer=self.writer)

        # Sobolev kernel
        if self.sobolev_grad:
            _s = config['sobolev_grad']['s']
            _lambda = config['sobolev_grad']['lambda']

            S, S_sqrt = sobolev_kernel_1D(_s, _lambda)

            S = torch.from_numpy(S).float()
            S.unsqueeze_(0)
            S = torch.stack((S, S, S), 0)
            self.S = S.to(self.device, non_blocking=True)

            S_sqrt = torch.from_numpy(S_sqrt).float()
            S_sqrt.unsqueeze_(0)
            S_sqrt = torch.stack((S_sqrt, S_sqrt, S_sqrt), 0)
            self.S_sqrt = S_sqrt.to(self.device, non_blocking=True)

            self.padding = (_s // 2, _s // 2)

        # optimisers
        self.optimizer_q_f = None
        self.optimizer_q_v = None

    def _save_mu_v(self, im_pair_idx, mu_v):
        torch.save(mu_v,
                   path.join(self.data_loader.save_dirs['mu_v'], 'mu_v_' + str(im_pair_idx) + '.pt'))

    def _save_log_var_v(self, im_pair_idx, log_var_v):
        torch.save(log_var_v,
                   path.join(self.data_loader.save_dirs['log_var_v'], 'log_var_v_' + str(im_pair_idx) + '.pt'))

    def _save_log_var_f(self, log_var_f):
        torch.save(log_var_f,
                   path.join(self.data_loader.save_dirs['log_var_f'], 'log_var_f.pt'))

    def _save_u_v(self, im_pair_idx, u_v):
        torch.save(u_v,
                   path.join(self.data_loader.save_dirs['u_v'], 'u_v_' + str(im_pair_idx) + '.pt'))

    def _save_u_f(self, u_f):
        torch.save(u_f,
                   path.join(self.data_loader.save_dirs['u_f'], 'u_f.pt'))

    def _save_tensors(self, im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f):
        """
        save the variational parameters to disk in order to load at the next epoch
        """
        im_pair_idxs = im_pair_idxs.tolist()
        
        log_var_f, u_f = torch.mean(log_var_f, dim=0), torch.mean(u_f, dim=0)
        log_var_f, u_f = log_var_f.cpu(), u_f.cpu()

        self._save_u_f(u_f)
        self._save_log_var_f(log_var_f)

        mu_v = mu_v.cpu()
        log_var_v, u_v = log_var_v.cpu(), u_v.cpu()
        
        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            self._save_mu_v(im_pair_idx, mu_v[loop_idx])
            self._save_log_var_v(im_pair_idx, log_var_v[loop_idx])
            self._save_u_v(im_pair_idx, u_v[loop_idx])

    def _load_optimiser_q_v(self, batch_idx):
        file_path = path.join(self.data_loader.save_dirs['optimisers'], 'optimiser_q_v_' + str(batch_idx) + '.pt')

        if path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.optimizer_q_v.load_state_dict(checkpoint)

    def _load_optimiser_q_f(self, batch_idx):
        file_path = path.join(self.data_loader.save_dirs['optimisers'], 'optimiser_q_f_' + str(batch_idx) + '.pt')

        if path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.optimizer_q_f.load_state_dict(checkpoint)

    def _save_optimiser_q_v(self, batch_idx):
        file_path = path.join(self.data_loader.save_dirs['optimisers'], 'optimiser_q_v_' + str(batch_idx) + '.pt')
        save_optimiser_to_disk(self.optimizer_q_v, file_path)

    def _save_optimiser_q_f(self, batch_idx):
        file_path = path.join(self.data_loader.save_dirs['optimisers'], 'optimiser_q_f_' + str(batch_idx) + '.pt')
        save_optimiser_to_disk(self.optimizer_q_f, file_path)

    def _registration_step(self, im_fixed, im_moving, mu_v, log_var_v=None, u_v=None):
        if self.learn_q_v:
            data_term = 0.0
            reg_term = 0.0

            if self.no_samples == 1:
                if self.sobolev_grad:
                    # draw a sample from q_v
                    v_sample = sample_qv(mu_v, log_var_v, u_v, self.S_sqrt, self.padding, 1)
                    v_sample = SobolevGrad.apply(v_sample, self.S, self.padding)
                else:
                    v_sample = sample_qv(mu_v, log_var_v, u_v, None, None, 1)

                transformation, displacement = self.transformation_model(v_sample)
                im_moving_warped = self.registration_module(im_moving, transformation)

                if self.learn_sim_metric:
                    im_out = self.enc(im_fixed, im_moving_warped)
                    data_term += self.data_loss(None, None, im_out).sum()
                else:
                    data_term += self.data_loss(im_fixed, im_moving_warped).sum()

                reg_term += self.reg_loss(v_sample).sum()
            elif self.no_samples == 2:
                if self.sobolev_grad:
                    v_sample1, v_sample2 = sample_qv(mu_v, log_var_v, u_v, self.S_sqrt, self.padding, 2)
                    v_sample1, v_sample2 = SobolevGrad.apply(v_sample1, self.S, self.padding), \
                                           SobolevGrad.apply(v_sample2, self.S, self.padding)
                else:
                    v_sample1, v_sample2 = sample_qv(mu_v, log_var_v, u_v, None, None, 2)

                transformation1, displacement1 = self.transformation_model(v_sample1)
                transformation2, displacement2 = self.transformation_model(v_sample2)

                im_moving_warped1 = self.registration_module(im_moving, transformation1)
                im_moving_warped2 = self.registration_module(im_moving, transformation2)

                if self.learn_sim_metric:
                    im_out1 = self.enc(im_fixed, im_moving_warped1)
                    im_out2 = self.enc(im_fixed, im_moving_warped2)

                    data_term += self.data_loss(None, None, im_out1).sum() / 2.0
                    data_term += self.data_loss(None, None, im_out2).sum() / 2.0
                else:
                    data_term += self.data_loss(im_fixed, im_moving_warped1).sum() / 2.0
                    data_term += self.data_loss(im_fixed, im_moving_warped2).sum() / 2.0

                reg_term += self.reg_loss(v_sample1).sum() / 2.0
                reg_term += self.reg_loss(v_sample2).sum() / 2.0

            return data_term, reg_term, self.entropy(log_var_v, u_v).sum()

        transformation, displacement = self.transformation_model(mu_v)
        im_moving_warped = self.registration_module(im_moving, transformation)

        return self.data_loss(im_fixed, im_moving_warped).sum(), self.reg_loss(mu_v).sum()

    def _step_q_v(self, epoch, batch_idx, curr_batch_size, im_pair_idxs, im_fixed, im_moving, mu_v, log_var_v, u_v):
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
            self._load_optimiser_q_v(batch_idx)

            # optimise q_v
            for iter_no in range(self.no_steps_v):
                self.optimizer_q_v.zero_grad()

                data_term, reg_term, entropy_term = self._registration_step(im_fixed, im_moving, mu_v, log_var_v, u_v)
                loss_q_v = data_term + reg_term + entropy_term

                loss_q_v.backward()
                self.optimizer_q_v.step()  # backprop

                # metrics
                step = epoch - 1
                global_step = (epoch - 1) * self.no_steps_v + iter_no + 1

                if iter_no % self.log_step == 0 or math.log2(global_step).is_integer():
                    with torch.no_grad():
                        if self.sobolev_grad:
                            mu_v_conv_sqrt = separable_conv_3d(mu_v, self.S_sqrt, self.padding)
                            transformation, displacement = self.transformation_model(mu_v_conv_sqrt)
                        else:
                            transformation, displacement = self.transformation_model(mu_v)

                        im_moving_warped = self.registration_module(im_moving, transformation)

                        if iter_no % self.log_step == 0:
                            self.writer.set_step(step)

                            if self.learn_sim_metric:
                                im_out = self.enc(im_fixed, im_moving_warped)
                                data_term_value = self.data_loss(None, None, im_out).sum().item() / curr_batch_size
                            else:
                                data_term_value = \
                                    self.data_loss(im_fixed, im_moving_warped).sum().item() / curr_batch_size
                            
                            if self.sobolev_grad:
                                reg_term_value = self.reg_loss(mu_v_conv_sqrt).sum().item() / curr_batch_size

                                log_var_v_conv_sqrt = separable_conv_3d(log_var_v, self.S_sqrt, self.padding)
                                u_v_conv_sqrt = separable_conv_3d(u_v, self.S_sqrt, self.padding)

                                entropy_term_value = self.entropy(log_var_v_conv_sqrt, u_v_conv_sqrt).sum().item() / curr_batch_size
                            else:
                                reg_term_value = self.reg_loss(mu_v).sum().item() / curr_batch_size
                                entropy_term_value = self.entropy(log_var_v, u_v).sum().item() / curr_batch_size

                            self.train_metrics.update('data_term', data_term_value)
                            self.train_metrics.update('reg_term', reg_term_value)
                            self.train_metrics.update('entropy_term', entropy_term_value)

                            registration_print(self.logger, iter_no, self.no_steps_v,
                                               loss_q_v.item() / curr_batch_size,
                                               data_term_value, reg_term_value, entropy_term_value)

                        if math.log2(global_step).is_integer():
                            self.writer.set_step(global_step)

                            log_images(self.writer, im_pair_idxs, im_fixed, im_moving, im_moving_warped)
                            log_log_det_J_transformation(
                                self.writer, im_pair_idxs, transformation, get_module_attr(self.reg_loss, 'diff_op'))

                            if self.sobolev_grad:
                                log_var_v_conv_sqrt = separable_conv_3d(log_var_v, self.S_sqrt, self.padding)
                                u_v_conv_sqrt = separable_conv_3d(u_v, self.S_sqrt, self.padding)
                                log_q_v(self.writer, im_pair_idxs, mu_v_conv_sqrt, displacement, log_var_v_conv_sqrt, u_v_conv_sqrt)
                            else:
                                log_q_v(self.writer, im_pair_idxs, mu_v_conv_sqrt, displacement, log_var_v, u_v)

            # save the optimiser
            self._save_optimiser_q_v(batch_idx)

            # disable gradients
            mu_v.requires_grad_(False)
            log_var_v.requires_grad_(False)
            u_v.requires_grad_(False)
        else:
            mu_v.requires_grad_(True)
            self.optimizer_q_v = self.config.init_obj('optimizer_v', torch.optim, [mu_v])
            self._load_optimiser_q_v(batch_idx)

            for iter_no in range(self.no_steps_v):
                self.optimizer_q_v.zero_grad()

                data_term, reg_term = self._registration_step(im_fixed, im_moving, mu_v)
                loss_q_v = data_term + reg_term

                loss_q_v.backward()
                self.optimizer_q_v.step()

                # metrics
                step = epoch - 1
                global_step = (epoch - 1) * self.no_steps_v + iter_no + 1

                if iter_no % self.log_step == 0 or math.log2(global_step).is_integer():
                    with torch.no_grad():
                        if self.sobolev_grad:
                            mu_v_conv_sqrt = separable_conv_3d(mu_v, self.S_sqrt, self.padding)
                            transformation, displacement = self.transformation_model(mu_v_conv_sqrt)
                        else:
                            transformation, displacement = self.transformation_model(mu_v)
                            
                        im_moving_warped = self.registration_module(im_moving, transformation)

                        if iter_no % self.log_step == 0:
                            self.writer.set_step(step)

                            if self.learn_sim_metric:
                                im_out = self.enc(im_fixed, im_moving_warped)
                                data_term_value = self.data_loss(None, None, im_out).sum().item() / curr_batch_size
                            else:
                                data_term_value = \
                                    self.data_loss(im_fixed, im_moving_warped).sum().item() / curr_batch_size

                            if self.sobolev_grad:
                                reg_term_value = self.reg_loss(mu_v_conv_sqrt).sum().item() / curr_batch_size
                            else:
                                reg_term_value = self.reg_loss(mu_v).sum().item() / curr_batch_size

                            self.train_metrics.update('data_term', data_term_value)
                            self.train_metrics.update('reg_term', reg_term_value)

                            registration_print(self.logger, iter_no, self.no_steps_v,
                                               loss_q_v.item() / curr_batch_size, data_term_value, reg_term_value)

                        if math.log2(global_step).is_integer():
                            self.writer.set_step(global_step)

                            log_images(self.writer, im_pair_idxs, im_fixed, im_moving, im_moving_warped)
                            log_log_det_J_transformation(
                                self.writer, im_pair_idxs, transformation, get_module_attr(self.reg_loss, 'diff_op'))

                            if self.sobolev_grad:
                                log_var_v_conv_sqrt = separable_conv_3d(log_var_v, self.S_sqrt, self.padding)
                                u_v_conv_sqrt = separable_conv_3d(u_v, self.S_sqrt, self.padding)
                                log_q_v(self.writer, im_pair_idxs, mu_v_conv_sqrt, displacement, log_var_v_conv_sqrt, u_v_conv_sqrt)
                            else:
                                log_q_v(self.writer, im_pair_idxs, mu_v_conv_sqrt, displacement, log_var_v, u_v)

            self._save_optimiser_q_v(batch_idx)
            mu_v.requires_grad_(False)

        return loss_q_v.item() / curr_batch_size

    def _step_q_f_q_phi(self, epoch, batch_idx, curr_batch_size,
                        im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f):
        """
        update parameters of q_f and q_phi
        """

        # enable gradients
        log_var_f.requires_grad_(True)
        u_f.requires_grad_(True)

        self.enc.train()
        get_module_attr(self.enc, 'set_grad_enabled')(True)
        
        # initialise the optimiser
        self.optimizer_q_f = self.config.init_obj('optimizer_f', torch.optim, [log_var_f, u_f])
        self._load_optimiser_q_f(batch_idx)

        # optimise the encoding function
        self.optimizer_q_f.zero_grad()
        self.optimizer_q_phi.zero_grad()

        loss_q_f_q_phi = 0.0
        
        if self.sobolev_grad:
            # draw a sample from q_v
            v_sample = sample_qv(mu_v, log_var_v, u_v, self.S_sqrt, self.padding, 1)
            v_sample = SobolevGrad.apply(v_sample, self.S, self.padding)
        else:
            v_sample = sample_qv(mu_v, log_var_v, u_v, None, None, 1)

        transformation, displacement = self.transformation_model(v_sample)
        im_moving_warped = self.registration_module(im_moving, transformation)

        if self.learn_sim_metric:
            im_out = self.enc(im_fixed, im_moving_warped)
            loss_q_f_q_phi += self.data_loss(None, None, im_out).sum()
        else:
            loss_q_f_q_phi += self.data_loss(im_fixed, im_moving_warped).sum()

        if self.no_samples == 1:
            im_fixed_sample = sample_qf(im_fixed, log_var_f, u_f, 1)  # draw sample from q_f

            if self.learn_sim_metric:
                im_out = self.enc(im_fixed_sample, im_moving_warped)
                loss_q_f_q_phi -= self.data_loss(None, None, im_out).sum()
            else:
                loss_q_f_q_phi -= self.data_loss(im_fixed_sample, im_moving_warped).sum()
        elif self.no_samples == 2:
            im_fixed_sample1, im_fixed_sample2 = sample_qf(im_fixed, log_var_f, u_f, 2)

            if self.learn_sim_metric:
                im_out1 = self.enc(im_fixed_sample1, im_moving_warped)
                im_out2 = self.enc(im_fixed_sample2, im_moving_warped)

                loss_q_f_q_phi -= self.data_loss(None, None, im_out1).sum() / 2.0
                loss_q_f_q_phi -= self.data_loss(None, None, im_out2).sum() / 2.0
            else:
                loss_q_f_q_phi -= self.data_loss(im_fixed_sample1, im_moving_warped).sum() / 2.0
                loss_q_f_q_phi -= self.data_loss(im_fixed_sample2, im_moving_warped).sum() / 2.0

        loss_q_f_q_phi /= curr_batch_size
        loss_q_f_q_phi.backward()

        self.optimizer_q_f.step()  # backprop
        self.optimizer_q_phi.step()  

        # save the optimiser
        self._save_optimiser_q_f(batch_idx)

        # disable gradients
        log_var_f.requires_grad_(False)
        u_f.requires_grad_(False)
        
        # metrics
        step = epoch - 1
        self.writer.set_step(step)

        with torch.no_grad():
            log_q_f(self.writer, log_var_f, u_f)

        self.enc.eval()
        get_module_attr(self.enc, 'set_grad_enabled')(False)

        return loss_q_f_q_phi.item()

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: integer, current training epoch
        :return: log that contains average loss and metric in this epoch
        """

        self.train_metrics.reset()

        for batch_idx, (im_pair_idxs, im_fixed, seg_fixed, im_moving, seg_moving,
                        mu_v, log_var_v, u_v, log_var_f, u_f) in enumerate(self.data_loader):
            curr_batch_size = float(im_pair_idxs.numel())

            im_fixed, im_moving = im_fixed.to(self.device, non_blocking=True), \
                                  im_moving.to(self.device, non_blocking=True)  # images to register

            seg_fixed, seg_moving = seg_fixed.to(self.device, non_blocking=True), \
                                    seg_moving.to(self.device, non_blocking=True)  # corresponding segmentations

            mu_v = mu_v.to(self.device, non_blocking=True)  # mean velocity field

            log_var_v, u_v = log_var_v.to(self.device, non_blocking=True), \
                             u_v.to(self.device, non_blocking=True)
            log_var_f, u_f = log_var_f.to(self.device, non_blocking=True), \
                             u_f.to(self.device, non_blocking=True)  # variational parameters

            # print value of the data term before registration
            with torch.no_grad():
                if self.sobolev_grad:
                    mu_v_conv_sqrt = separable_conv_3d(mu_v, self.S_sqrt, self.padding)
                    transformation, displacement = self.transformation_model(mu_v_conv_sqrt)
                else:
                    transformation, displacement = self.transformation_model(mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)

                if self.learn_sim_metric:
                    im_out_unwarped = self.enc(im_fixed, im_moving)
                    im_out = self.enc(im_fixed, im_moving_warped)
                
                    loss_unwarped = self.data_loss(None, None, im_out_unwarped).sum() / curr_batch_size
                    loss_warped = self.data_loss(None, None, im_out).sum() / curr_batch_size
                else:
                    loss_unwarped = self.data_loss(im_fixed, im_moving).sum() / curr_batch_size
                    loss_warped = self.data_loss(im_fixed, im_moving_warped).sum() / curr_batch_size

                self.logger.info(f'\nPRE-REGISTRATION: ' +
                                 f'unwarped: {loss_unwarped:.5f}' +
                                 f', warped w/ the current estimate: {loss_warped:.5f}\n'
                                 )

            """
            training
            """

            total_loss = 0.0

            # q_v
            loss_q_v = self._step_q_v(epoch, batch_idx, curr_batch_size, im_pair_idxs,
                                      im_fixed, im_moving, mu_v, log_var_v, u_v)
            total_loss += loss_q_v

            if self.learn_sim_metric:  # q_f and q_phi
                self.logger.info('\noptimising q_f and q_phi..')
                loss_q_f_q_phi = self._step_q_f_q_phi(epoch, batch_idx, curr_batch_size,
                                                      im_fixed, im_moving, mu_v, log_var_v, u_v, log_var_f, u_f)
                total_loss += loss_q_f_q_phi

            # save tensors with the updated variational parameters
            self._save_tensors(im_pair_idxs, mu_v, log_var_v, u_v, log_var_f, u_f) 

            # metrics
            step = epoch - 1
            self.writer.set_step(step)

            with torch.no_grad():
                if self.sobolev_grad:
                    mu_v_conv_sqrt = separable_conv_3d(mu_v, self.S_sqrt, self.padding)
                    transformation, displacement = self.transformation_model(mu_v_conv_sqrt)
                else:
                    transformation, displacement = self.transformation_model(mu_v)

                im_moving_warped = self.registration_module(im_moving, transformation)
                seg_moving_warped = self.registration_module(seg_moving, transformation, mode='nearest')

                dsc = dice(seg_fixed, seg_moving_warped)
                for class_idx, val in enumerate(dsc):
                    self.train_metrics.update('dice_' + str(class_idx + 1), val)

                # save images, fields etc.
                if self.sobolev_grad:
                    log_var_v_conv_sqrt = separable_conv_3d(log_var_v, self.S_sqrt, self.padding)
                    u_v_conv_sqrt = separable_conv_3d(u_v, self.S_sqrt, self.padding)

                    save_images(self.data_loader.save_dirs, im_pair_idxs, im_fixed, im_moving, im_moving_warped,
                                mu_v_conv_sqrt, log_var_v_conv_sqrt, u_v_conv_sqrt, log_var_f, u_f, displacement,
                                seg_fixed, seg_moving, seg_moving_warped)
                else:
                    save_images(self.data_loader.save_dirs, im_pair_idxs, im_fixed, im_moving, im_moving_warped,
                                mu_v, log_var_v, u_v, log_var_f, u_f, displacement,
                                seg_fixed, seg_moving, seg_moving_warped)

                # save grids
                save_grids(self.data_loader.save_dirs, im_pair_idxs, transformation)

                self.logger.info('\nsaved the output images and vector fields to disk\n')

            self.train_metrics.update('loss', total_loss)

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
