import numpy as np
import torch
import torch.nn.functional as F

from base import BaseTrainer
from utils import generate_grid, inf_loop, integrate_vect, MetricTracker, Sampler


class Trainer(BaseTrainer):
    """
    trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

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

        self.sampler = Sampler(self.device)
        self.grid = None

    @staticmethod
    def _save_params(idx, v, sigma_voxel_v, u_v, sigma_voxel_f, u_f):
        torch.save(v, './temp/vels/v_' + str(idx) + '.pt')
        torch.save(sigma_voxel_v, './temp/sigmas_v/sigma_' + str(idx) + '.pt')
        torch.save(u_v, './temp/modes_of_variation_v/u_' + str(idx) + '.pt')
        torch.save(sigma_voxel_f, './temp/sigmas_f/sigma_' + str(idx) + '.pt')
        torch.save(u_f, './temp/modes_of_variation_f/u_' + str(idx) + '.pt')

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        :param epoch: integer, current training epoch
        :return: log that contains average loss and metric in this epoch
        """

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (im_pair_idx, im1, im2, v, sigma_voxel_v, u_v, sigma_voxel_f, u_f) in enumerate(self.data_loader):
            im1 = im1.to(self.device)
            im2 = im2.to(self.device)

            if self.grid is None:  # initialise the grid
                dim = im1.shape[2:]
                self.grid = generate_grid(dim)

            v.requires_grad = True
            v = v.to(self.device)

            sigma_voxel_v.requires_grad = True
            sigma_voxel_v = sigma_voxel_v.to(self.device)
            u_v.requires_grad = True
            u_v = u_v.to(self.device)

            """
            q_w
            """

            for p in self.model.parameters():
                p.requires_grad = False

            self.optimizer.zero_grad()
            total_loss = 0.0
            loss = 0.0

            for _ in range(self.no_samples):
                v_sample = self.sampler.sample_qv(v, sigma_voxel_v, u_v)
                warp_field = integrate_vect(v_sample)

                im2_warped = F.grid_sample(im2, warp_field + self.grid)
                im2_warped = im2_warped.to(self.device)

                im_out = self.model(im1, im2_warped)
                loss += self.criterion(im_out)

            loss /= float(self.no_samples)
            loss += self.criterion(None, v, sigma_voxel_v, u_v, self.model.diff_op)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            """
            q_phi
            """

            v.detach()
            sigma_voxel_v.detach()
            u_v.detach()

            for p in self.model.parameters():
                p.requires_grad = True

            sigma_voxel_f.requires_grad = True
            sigma_voxel_f = sigma_voxel_f.to(self.device)
            u_f.requires_grad = True
            u_f = u_f.to(self.device)

            self.optimizer.zero_grad()
            loss = 0.0

            # first term
            for _ in range(self.no_samples):
                v_sample = self.sampler.sample_qv(v, sigma_voxel_v, u_v)
                warp_field = integrate_vect(v_sample)

                im2_warped = F.grid_sample(im2, warp_field + self.grid)
                im2_warped = im2_warped.to(self.device)

                im_out = self.model(im1, im2_warped)
                loss += self.criterion(im_out)

            # second term
            for _ in range(self.no_samples):
                loss_aux = 0.0

                v_sample = self.sampler.sample_qv(v, sigma_voxel_v, u_v)
                warp_field = integrate_vect(v_sample)

                im2_warped = F.grid_sample(im2, warp_field + self.grid)
                im2_warped = im2_warped.to(self.device)

                for _ in range(self.no_samples):
                    f_sample = self.sampler.sample_qf(im1, sigma_voxel_f, u_f)

                    im_out = self.model(f_sample, im2_warped)
                    loss_aux += self.criterion(im_out)

                loss_aux /= self.no_samples
                loss += loss_aux

            loss /= self.no_samples
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            sigma_voxel_f.detach()
            u_f.detach()

            self._save_params(int(im_pair_idx[0]), v, sigma_voxel_v, u_v, sigma_voxel_f, u_f)  # save updated tensors
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', total_loss)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))

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
