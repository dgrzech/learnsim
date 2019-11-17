import numpy as np
import os
import torch
import torch.nn.functional as F

from base import BaseTrainer
from utils import generate_id_grid, inf_loop, integrate_v, MetricTracker, Sampler


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer_v, optimizer_phi, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer_v, optimizer_phi, config)

        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        """
        utils
        """

        self.sampler = Sampler()
        self.identity_grid = None

    @staticmethod
    def _save_params(idx, v, log_var_v, u_v, optimizer_v, log_var_f, u_f):
        torch.save(v, './temp/vels/v_' + str(idx) + '.pt')
        torch.save(log_var_v, './temp/log_var_v/log_var_v' + str(idx) + '.pt')
        torch.save(optimizer_v.state_dict(), './temp/opt/opt_v_' + str(idx) + '.tar')

        torch.save(u_v, './temp/modes_of_variation_v/u_' + str(idx) + '.pt')
        torch.save(log_var_f, './temp/log_var_f/log_var_f_' + str(idx) + '.pt')
        torch.save(u_f, './temp/modes_of_variation_f/u_' + str(idx) + '.pt')

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: integer, current training epoch
        :return: log that contains average loss and metric in this epoch
        """

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (im_pair_idx, im1, im2, v, log_var_v, u_v, log_var_f, u_f) in enumerate(self.data_loader):
            im1 = im1.to(self.device)
            im2 = im2.to(self.device)

            if self.identity_grid is None:  # initialise the grid
                dim = im1.shape[2:]
                self.identity_grid = generate_id_grid(dim)
            
            """
            q_v
            """

            v = v.to(self.device).detach().requires_grad_(True)
            log_var_v = log_var_v.to(self.device).detach().requires_grad_(True)
            u_v = u_v.to(self.device).detach().requires_grad_(True)
            
            for p in self.model.parameters():
                p.requires_grad_(False)
            
            if not os.path.exists('./temp/opt/opt_' + str(im_pair_idx) + '.tar'):
                self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [v, log_var_v, u_v])
                torch.save(self.optimizer_v.state_dict(), './temp/opt/opt_v_' + str(int(im_pair_idx[0])) + '.tar')
            else:
                self.optimizer_v = self.config.init_obj('optimizer_v', torch.optim, [v, log_var_v, u_v])
                checkpoint = torch.load('./temp/opt/opt_v_' + str(int(im_pair_idx[0])) + '.tar')
                self.optimizer_v.load_state_dict(checkpoint['optimizer_state_dict'])
 
            total_loss = 0.0

            for iter_no in range(self.no_steps_v):
                self.optimizer_v.zero_grad()
                loss = 0.0

                for _ in range(self.no_samples):
                    v_sample = self.sampler.sample_qv(v, log_var_v, u_v)
                    warp_field = self.identity_grid + integrate_v(v_sample, self.identity_grid).permute([0, 2, 3, 4, 1])
                    im2_warped = F.grid_sample(im2, warp_field, padding_mode='border')

                    im_out = self.model(im1, im2_warped)
                    loss -= self.criterion(im_out)
                
                loss /= float(self.no_samples)
                loss -= self.criterion(None, v, log_var_v, u_v, self.model.diff_op)
                if iter_no == 0 or iter_no % 16 == 0 or iter_no == self.no_steps_v - 1:
                    print('iter ' + str(iter_no) + '/' + str(self.no_steps_v - 1) + ', cost: ' + str(loss.item()))

                loss.backward()
                self.optimizer_v.step()
                
            total_loss += loss.item()

            """
            q_phi
            """

            v.detach()
            log_var_v.detach()
            u_v.detach()

            for p in self.model.parameters():
                p.requires_grad_(True)

            log_var_f = log_var_f.to(self.device).detach().requires_grad_(True)
            u_f = u_f.to(self.device).detach().requires_grad_(True)

            self.optimizer_phi.zero_grad()
            loss = 0.0

            # first term
            for _ in range(self.no_samples):
                v_sample = self.sampler.sample_qv(v, log_var_v, u_v)
                warp_field = self.identity_grid + integrate_v(v_sample, self.identity_grid).permute([0, 2, 3, 4, 1])
                im2_warped = F.grid_sample(im2, warp_field, padding_mode='border')

                im_out = self.model(im1, im2_warped)
                loss -= self.criterion(im_out)

            # second term
            for _ in range(self.no_samples):
                loss_aux = 0.0

                v_sample = self.sampler.sample_qv(v, log_var_v, u_v)
                warp_field = self.identity_grid + integrate_v(v_sample, self.identity_grid).permute([0, 2, 3, 4, 1])
                im2_warped = F.grid_sample(im2, warp_field, padding_mode='border')

                for _ in range(self.no_samples):
                    f_sample = self.sampler.sample_qf(im1, log_var_f, u_f)

                    im_out = self.model(f_sample, im2_warped)
                    loss_aux -= self.criterion(im_out)

                loss_aux /= float(self.no_samples)
                loss -= loss_aux

            loss /= (2.0 * float(self.no_samples))
            loss.backward()
            self.optimizer_phi.step()

            total_loss += loss.item()

            log_var_f.detach()
            u_f.detach()

            self._save_params(int(im_pair_idx[0]), v, log_var_v, u_v, self.optimizer_v, log_var_f, u_f)  # save updated tensors
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', -1.0 * total_loss)

            with torch.no_grad():
                warp_field = self.identity_grid + integrate_v(v, self.identity_grid).permute([0, 2, 3, 4, 1])
                im2_warped = F.grid_sample(im2, warp_field, padding_mode='border')

                im_out = self.model(im1, im2_warped)

                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(im_out))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), -1.0 * loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

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
