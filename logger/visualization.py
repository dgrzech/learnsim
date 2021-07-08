import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils import calc_norm, get_module_attr


class TensorboardWriter:
    def __init__(self, log_dir):
        log_dir = str(log_dir)
        self.rank = dist.get_rank()
        self.step = 0
        self.timer = datetime.now()

        if self.rank == 0:
            self.writer = SummaryWriter(log_dir)

        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio', 'add_figure',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_histogram'
        }

    def set_step(self, step):
        self.step = step

    def write_graph(self, model):
        if self.rank == 0:
            im_fixed = im_moving = torch.randn([1, 1, 128, 128, 128], device=self.rank)
            mask = torch.ones([1, 1, 128, 128, 128], device=self.rank).bool()
            inputs = (im_fixed, im_moving, mask)

            self.writer.add_graph(model, input_to_model=inputs)

    def write_hparams(self, config):  # NOTE (DG): should use add_hparams but it's not working with DDP..
        if self.rank == 0:
            text = json.dumps(config, indent=4, sort_keys=False)
            self.writer.add_text('hparams', text)

    def __getattr__(self, name):
        """
        if visualization is configured, return add_data() methods of tensorboard with additional information (step, tag) added; else return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:  # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


def pad_to_square(im_arr):
    sz = np.max(im_arr.shape)
    H, W = im_arr.shape[0], im_arr.shape[1]

    a, b = (sz - H) // 2, (sz - W) // 2
    aa, bb = sz - a - H, sz - b - W

    return np.pad(im_arr, pad_width=((a, aa), (b, bb)), mode='minimum')


"""
model
"""


def log_model_weights(writer,  model):
    if dist.get_rank() == 0:
        for name, p in model.named_parameters():
            writer.add_histogram(name, p, bins=np.arange(-1.5, 1.5, 0.1))


"""
energy-based model samples
"""


def model_samples_grid(samples_slices):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['sagittal', 'coronal', 'axial']
    rows = ['model_sample']

    for ax, col in zip(axs, cols):
        ax.set_title(col)

    for ax, row in zip(axs, rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[i].imshow(pad_to_square(samples_slices[i]), cmap='gray')

    return fig


def log_model_samples(writer, im_pair_idxs, model_samples):
    if dist.get_rank() == 0:
        model_samples = model_samples.cpu().numpy()
        mid_idxs = get_im_or_field_mid_slices_idxs(model_samples)

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            sample = model_samples[loop_idx, 0]
            sample_slices = get_slices(sample, mid_idxs)

            writer.add_figure(f'model_samples/{im_pair_idx}', model_samples_grid(sample_slices))


"""
images
"""


def im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices):
    """
    plot of input and output images to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['sagittal', 'coronal', 'axial']
    rows = ['im_fixed', 'im_moving', 'im_moving_warped']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(pad_to_square(im_fixed_slices[i]), cmap='gray')
        axs[1, i].imshow(pad_to_square(im_moving_slices[i]), cmap='gray')
        axs[2, i].imshow(pad_to_square(im_moving_warped_slices[i]), cmap='gray')

    return fig


def get_im_or_field_mid_slices_idxs(im_or_field):
    if len(im_or_field.shape) == 3:
        return int(im_or_field.shape[2] / 2), int(im_or_field.shape[1] / 2), int(im_or_field.shape[0] / 2)
    elif len(im_or_field.shape) == 4:
        return int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2), int(im_or_field.shape[1] / 2)
    elif len(im_or_field.shape) == 5:
        return int(im_or_field.shape[4] / 2), int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2)

    raise NotImplementedError


def get_slices(field, mid_idxs):
    return [field[:, :, mid_idxs[0]], field[:, mid_idxs[1], :], field[mid_idxs[2], :, :]]


def log_images(writer, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch):
    if dist.get_rank() == 0:
        im_fixed_batch = im_fixed_batch.cpu().numpy()
        im_moving_batch = im_moving_batch.cpu().numpy()
        im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

        mid_idxs = get_im_or_field_mid_slices_idxs(im_moving_batch)

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            im_fixed = im_fixed_batch[loop_idx, 0]
            im_moving = im_moving_batch[loop_idx, 0]
            im_moving_warped = im_moving_warped_batch[loop_idx, 0]

            im_fixed_slices = get_slices(im_fixed, mid_idxs)
            im_moving_slices = get_slices(im_moving, mid_idxs)
            im_moving_warped_slices = get_slices(im_moving_warped, mid_idxs)

            writer.add_figure(f'im_pairs/{im_pair_idx}',
                              im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices))


"""
vector fields
"""


def fields_grid(mu_v_norm_slices, displacement_norm_slices, grid_im_slices, sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices):
    """
    plot of the norms of output vector fields to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(10, 10))

    cols = ['sagittal', 'coronal', 'axial']
    rows = ['mu_v_norm', 'displacement_norm', 'grid', 'sigma_v_norm', 'u_v_norm', 'log_det_J']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(mu_v_norm_slices[i], cmap='hot')
        axs[1, i].imshow(displacement_norm_slices[i], cmap='hot')
        axs[2, i].imshow(grid_im_slices[i], cmap='gray')
        axs[3, i].imshow(sigma_v_norm_slices[i], cmap='hot')
        axs[4, i].imshow(u_v_norm_slices[i], cmap='hot')
        axs[5, i].imshow(log_det_J_slices[i])

    return fig


def log_fields(writer, im_pair_idxs, var_params_batch, displacement_batch, grid_im_batch, log_det_J_batch):
    if dist.get_rank() == 0:
        mu_v_norm_batch = calc_norm(var_params_batch['mu']).cpu().numpy()
        sigma_v_norm_batch = calc_norm(torch.exp(0.5 * var_params_batch['log_var'])).cpu().numpy()
        u_v_norm_batch = calc_norm(var_params_batch['u']).cpu().numpy()

        displacement_norm_batch = calc_norm(displacement_batch).cpu().numpy()
        grid_im_batch = grid_im_batch.cpu().numpy()
        log_det_J_batch = log_det_J_batch.cpu().numpy()

        mid_idxs = get_im_or_field_mid_slices_idxs(mu_v_norm_batch)

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
            mu_v_norm = mu_v_norm_batch[loop_idx, 0]
            sigma_v_norm = sigma_v_norm_batch[loop_idx, 0]
            u_v_norm = u_v_norm_batch[loop_idx, 0]

            displacement_norm = displacement_norm_batch[loop_idx, 0]
            grid_im = grid_im_batch[loop_idx, 0]
            log_det_J = log_det_J_batch[loop_idx]

            mu_v_norm_slices = get_slices(mu_v_norm, mid_idxs)
            sigma_v_norm_slices = get_slices(sigma_v_norm, mid_idxs)
            u_v_norm_slices = get_slices(u_v_norm, mid_idxs)

            displacement_norm_slices = get_slices(displacement_norm, mid_idxs)
            grid_im_slices = get_slices(grid_im, mid_idxs)
            log_det_J_slices = get_slices(log_det_J, mid_idxs)

            writer.add_figure(f'q_v/{im_pair_idx}',
                              fields_grid(mu_v_norm_slices, displacement_norm_slices, grid_im_slices, sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices))


"""
samples
"""


def sample_grid(im_moving_warped_slices, v_norm_slices, displacement_norm_slices, log_det_J_slices):
    """
    plot of output images and vector fields related to a sample from MCMC to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['sagittal', 'coronal', 'axial']
    rows = ['im_moving_warped', 'v_curr_state_norm', 'displacement_norm', 'log_det_J']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(pad_to_square(im_moving_warped_slices[i]), cmap='gray')
        axs[1, i].imshow(v_norm_slices[i], cmap='hot')
        axs[2, i].imshow(displacement_norm_slices[i], cmap='hot')
        axs[3, i].imshow(log_det_J_slices[i])

    return fig


def log_sample(writer, im_pair_idxs, im_moving_warped_batch, v_batch, displacement_batch, log_det_J_batch):
    if dist.get_rank() == 0:
        im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()
        v_norm_batch = calc_norm(v_batch).cpu().numpy()
        displacement_norm_batch = calc_norm(displacement_batch).cpu().numpy()
        log_det_J_batch = log_det_J_batch.cpu().numpy()

        mid_idxs = get_im_or_field_mid_slices_idxs(v_batch)

        for loop_idx, im_pair_idx in enumerate(im_pair_idxs.tolist()):
            im_moving_warped = im_moving_warped_batch[loop_idx, 0]
            v_norm = v_norm_batch[loop_idx, 0]
            displacement_norm = displacement_norm_batch[loop_idx, 0]
            log_det_J = log_det_J_batch[loop_idx]

            im_moving_warped_slices = get_slices(im_moving_warped, mid_idxs)
            v_norm_slices = get_slices(v_norm, mid_idxs)
            displacement_norm_slices = get_slices(displacement_norm, mid_idxs)
            log_det_J_slices = get_slices(log_det_J, mid_idxs)

            writer.add_figure(f'samples/{im_pair_idx}',
                              sample_grid(im_moving_warped_slices, v_norm_slices, displacement_norm_slices, log_det_J_slices))
