from datetime import datetime

import importlib
import matplotlib.pyplot as plt

from utils import compute_norm


def im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices):
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
        axs[0, i].imshow(im_fixed_slices[i])
        axs[1, i].imshow(im_moving_slices[i])
        axs[2, i].imshow(im_moving_warped_slices[i])

    return fig


def var_params_q_v_grid(mu_v_norm_slices, displacement_norm_slices, log_var_v_norm_slices, u_v_norm_slices):
    fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['mu_v_norm', 'displacement_norm', 'log_var_v_norm', 'u_v_norm']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(mu_v_norm_slices[i])
        axs[1, i].imshow(displacement_norm_slices[i])
        axs[2, i].imshow(log_var_v_norm_slices[i])
        axs[3, i].imshow(u_v_norm_slices[i])

    return fig


def v_grid(mu_v_norm_slices, displacement_norm_slices):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['mu_v_norm', 'displacement_norm']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(mu_v_norm_slices[i])
        axs[1, i].imshow(displacement_norm_slices[i])

    return fig


def log_det_J_transformation_grid(log_det_J_transformation_slices):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 8))
    cols = ['axial', 'coronal', 'sagittal']

    for i in range(3):
        ax = axs[i]
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        im = ax.imshow(log_det_J_transformation_slices[i])
        ax.set_title(cols[i])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, aspect=1)

    return fig


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

        writer.add_figure('im_pair_' + str(im_pair_idx),
                          im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices))


def log_v(writer, im_pair_idxs, mu_v_batch, displacement_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    mid_x = int(mu_v_batch.shape[4] / 2)
    mid_y = int(mu_v_batch.shape[3] / 2)
    mid_z = int(mu_v_batch.shape[2] / 2)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        temp = compute_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()
        mu_v_norm_slices = [mu_v_norm[:, :, mid_x],
                            mu_v_norm[:, mid_y, :],
                            mu_v_norm[mid_z, :, :]]

        temp = compute_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()
        displacement_norm_slices = [displacement_norm[:, :, mid_x],
                                    displacement_norm[:, mid_y, :],
                                    displacement_norm[mid_z, :, :]]

        writer.add_figure('v_' + str(im_pair_idx), v_grid(mu_v_norm_slices, displacement_norm_slices))


def log_q_v(writer, im_pair_idxs, mu_v_batch, displacement_batch, log_var_v_batch, u_v_batch):
    im_pair_idxs = im_pair_idxs.tolist()

    mid_x = int(mu_v_batch.shape[4] / 2)
    mid_y = int(mu_v_batch.shape[3] / 2)
    mid_z = int(mu_v_batch.shape[2] / 2)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        temp = compute_norm(mu_v_batch[loop_idx])
        mu_v_norm = temp[0].cpu().numpy()
        mu_v_norm_slices = [mu_v_norm[:, :, mid_x],
                            mu_v_norm[:, mid_y, :],
                            mu_v_norm[mid_z, :, :]]

        temp = compute_norm(displacement_batch[loop_idx])
        displacement_norm = temp[0].cpu().numpy()
        displacement_norm_slices = [displacement_norm[:, :, mid_x],
                                    displacement_norm[:, mid_y, :],
                                    displacement_norm[mid_z, :, :]]

        temp = compute_norm(log_var_v_batch[loop_idx])
        log_var_v_norm = temp[0].cpu().numpy()
        log_var_v_norm_slices = [log_var_v_norm[:, :, mid_x],
                                 log_var_v_norm[:, mid_y, :],
                                 log_var_v_norm[mid_z, :, :]]

        temp = compute_norm(u_v_batch[loop_idx])
        u_v_norm = temp[0].cpu().numpy()
        u_v_norm_slices = [u_v_norm[:, :, mid_x],
                           u_v_norm[:, mid_y, :],
                           u_v_norm[mid_z, :, :]]

        writer.add_figure('q_v_' + str(im_pair_idx),
                          var_params_q_v_grid(mu_v_norm_slices, displacement_norm_slices,
                                              log_var_v_norm_slices, u_v_norm_slices))


def log_log_det_J_transformation(writer, im_pair_idxs, log_det_J_transformation_batch):
    im_pair_idxs = im_pair_idxs.tolist()
    log_det_J_transformation_batch = log_det_J_transformation_batch.cpu().numpy()

    mid_x = int(log_det_J_transformation_batch.shape[3] / 2)
    mid_y = int(log_det_J_transformation_batch.shape[2] / 2)
    mid_z = int(log_det_J_transformation_batch.shape[1] / 2)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        log_det_J_transformation = log_det_J_transformation_batch[loop_idx]
        log_det_J_transformation_slices = [log_det_J_transformation[:, :, mid_x],
                                           log_det_J_transformation[:, mid_y, :],
                                           log_det_J_transformation[mid_z, :, :]]

        writer.add_figure('log_det_J_transformation_' + str(im_pair_idx),
                          log_det_J_transformation_grid(log_det_J_transformation_slices))


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
