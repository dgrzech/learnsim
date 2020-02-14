from datetime import datetime

import importlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from utils import calc_norm


def im_flip(array):
    return np.fliplr(np.flipud(np.transpose(array, (1, 0))))


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.hist_xlim = None
        self.hist_ylim = None

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio', 'add_figure',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


"""
images
"""


def im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices):
    """
    plot of input and output images to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['im_fixed', 'im_moving', 'im_moving_warped']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(im_fixed_slices[i]))
        axs[1, i].imshow(im_flip(im_moving_slices[i]))
        axs[2, i].imshow(im_flip(im_moving_warped_slices[i]))

    return fig


def log_hist_res(writer, im_pair_idxs, residuals_batch, gmm):
    """
    plot of the resiudal histogram to log in tensorboard
    """

    batch_size = im_pair_idxs.numel()
    im_pair_idxs = im_pair_idxs.tolist()

    device_temp = residuals_batch.device
    residuals_batch = residuals_batch.view(batch_size, -1).cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        residuals = residuals_batch[loop_idx]

        fig, ax = plt.subplots()
        sns.distplot(residuals, kde=False, norm_hist=True)

        if writer.hist_xlim is None or writer.hist_ylim is None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            xmin = -1.0 * xmax

            writer.hist_xlim = (xmin, xmax)
            writer.hist_ylim = (ymin, ymax + 1.5)
        else:
            xmin, xmax = writer.hist_xlim[0], writer.hist_xlim[1]

        x = torch.linspace(xmin, xmax, steps=10000).unsqueeze(0).unsqueeze(-1).to(device_temp)
        model_fit = torch.exp(gmm.log_pdf(x))

        sns.lineplot(x=x.detach().squeeze().cpu().numpy(),
                     y=model_fit.detach().squeeze().cpu().numpy(), color='green', ax=ax)

        plt.xlim(writer.hist_xlim[0], writer.hist_xlim[1])
        plt.ylim(writer.hist_ylim[0], writer.hist_ylim[1])

        writer.add_figure('hist_residuals/' + str(im_pair_idx), fig)


def log_images(writer, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    mid_x = int(im_fixed_batch.shape[4] / 2)
    mid_y = int(im_fixed_batch.shape[3] / 2)
    mid_z = int(im_fixed_batch.shape[2] / 2)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_fixed = im_fixed_batch[loop_idx, 0]
        im_fixed_slices = [im_fixed[:, :, mid_x], im_fixed[:, mid_y, :], im_fixed[mid_z, :, :]]

        im_moving = im_moving_batch[loop_idx, 0]
        im_moving_slices = [im_moving[:, :, mid_x], im_moving[:, mid_y, :], im_moving[mid_z, :, :]]

        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        im_moving_warped_slices = [im_moving_warped[:, :, mid_x],
                                   im_moving_warped[:, mid_y, :],
                                   im_moving_warped[mid_z, :, :]]

        writer.add_figure('im_pairs/' + str(im_pair_idx),
                          im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices))


"""
vector fields
"""


def fields_grid(mu_v_norm_slices, displacement_norm_slices, sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices):
    """
    plot of the norms of output vector fields to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True, figsize=(10, 10))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['mu_v_norm', 'displacement_norm', 'sigma_v_norm', 'u_v_norm', 'log_det_J']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(mu_v_norm_slices[i]))
        axs[1, i].imshow(im_flip(displacement_norm_slices[i]))
        axs[2, i].imshow(im_flip(sigma_v_norm_slices[i]))
        axs[3, i].imshow(im_flip(u_v_norm_slices[i]))
        axs[4, i].imshow(im_flip(log_det_J_slices[i]))

    return fig


def log_fields(writer, im_pair_idxs, var_params_batch, displacement_batch, log_det_J_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    mu_v_batch = var_params_batch['mu_v']
    log_var_v_batch = var_params_batch['log_var_v']
    sigma_v_batch = torch.exp(0.5 * log_var_v_batch)
    u_v_batch = var_params_batch['u_v']
    log_det_J_batch = log_det_J_batch.cpu().numpy()

    mid_x = int(mu_v_batch.shape[4] / 2)
    mid_y = int(mu_v_batch.shape[3] / 2)
    mid_z = int(mu_v_batch.shape[2] / 2)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        temp = calc_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()
        mu_v_norm_slices = [mu_v_norm[:, :, mid_x], mu_v_norm[:, mid_y, :], mu_v_norm[mid_z, :, :]]

        temp = calc_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()
        displacement_norm_slices = [displacement_norm[:, :, mid_x],
                                    displacement_norm[:, mid_y, :],
                                    displacement_norm[mid_z, :, :]]

        temp = calc_norm(sigma_v_batch[loop_idx])
        sigma_v_norm = temp[0].cpu().numpy()
        sigma_v_norm_slices = [sigma_v_norm[:, :, mid_x], sigma_v_norm[:, mid_y, :], sigma_v_norm[mid_z, :, :]]

        temp = calc_norm(u_v_batch[loop_idx])
        u_v_norm = temp[0].cpu().numpy()
        u_v_norm_slices = [u_v_norm[:, :, mid_x], u_v_norm[:, mid_y, :], u_v_norm[mid_z, :, :]]

        log_det_J = log_det_J_batch[loop_idx]
        log_det_J_slices = [log_det_J[:, :, mid_x], log_det_J[:, mid_y, :], log_det_J[mid_z, :, :]]

        writer.add_figure('q_v/' + str(im_pair_idx),
                          fields_grid(mu_v_norm_slices, displacement_norm_slices,
                                      sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices))


"""
samples
"""


def sample_grid(im_moving_warped_slices, mu_v_norm_slices, displacement_norm_slices):
    """
    plot of output images and vector fields related to a sample from MCMC to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['im_moving_warped', 'mu_v_norm', 'displacement_norm']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(im_moving_warped_slices[i]))
        axs[1, i].imshow(im_flip(mu_v_norm_slices[i]))
        axs[2, i].imshow(im_flip(displacement_norm_slices[i]))

    return fig


def log_sample(writer, im_pair_idxs, im_moving_warped_batch, v_batch, displacement_batch):
    im_pair_idxs = im_pair_idxs.tolist()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    mid_x = int(v_batch.shape[4] / 2)
    mid_y = int(v_batch.shape[3] / 2)
    mid_z = int(v_batch.shape[2] / 2)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        im_moving_warped_slices = [im_moving_warped[:, :, mid_x],
                                   im_moving_warped[:, mid_y, :],
                                   im_moving_warped[mid_z, :, :]]

        temp = calc_norm(v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()
        mu_v_norm_slices = [mu_v_norm[:, :, mid_x], mu_v_norm[:, mid_y, :], mu_v_norm[mid_z, :, :]]

        temp = calc_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()
        displacement_norm_slices = [displacement_norm[:, :, mid_x],
                                    displacement_norm[:, mid_y, :],
                                    displacement_norm[mid_z, :, :]]

        writer.add_figure('samples/' + str(im_pair_idx),
                          sample_grid(im_moving_warped_slices, mu_v_norm_slices, displacement_norm_slices))
