import argparse
import nibabel as nib
import numpy as np
import os
import torch

from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as model_loss
import model.model as module_arch
import utils.registration as registration
import utils.transformation as transformation

from parse_config import ConfigParser
from utils.sampler import sample_qv


def save_to_disk(im, file_path, normalize=False):
    if normalize:
        im_min, im_max = torch.min(im), torch.max(im)
        im = 2.0 * (im - im_min) / (im_max - im_min) - 1.0

    im = im[0, 0, :, :, :].cpu().numpy()
    im = nib.Nifti1Image(im, np.eye(4))
    im.to_filename(file_path)


def main(config):
    logger = config.get_logger('test')

    """
    setup data loader instance
    """

    data_loader = config.init_obj('data_loader', module_data, save_dirs=config.save_dirs)

    """
    build the model architecture
    """

    enc = config.init_obj('arch', module_arch)
    transformation_model = config.init_obj('transformation_model', transformation)
    registration_module = config.init_obj('registration_module', registration)

    logger.info(enc)

    """
    initialise the losses
    """

    data_loss = config.init_obj('data_loss', model_loss)
    reg_loss = config.init_obj('reg_loss', model_loss)
    entropy = config.init_obj('entropy', model_loss)

    """
    prepare model for testing
    """

    logger.info('loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        enc = torch.nn.DataParallel(enc)
        transformation_model = torch.nn.DataParallel(transformation_model)
        registration_module = torch.nn.DataParallel(registration_module)

    enc.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = enc.to(device)
    transformation_model = transformation_model.to(device)
    registration_module = registration_module.to(device)

    enc.eval()
    transformation_model.eval()
    registration_module.eval()

    """
    test
    """

    no_steps_v = config['trainer']['no_steps_v']
    no_samples = config['trainer']['no_samples']

    for batch_idx, (im_pair_idxs, im1, im2, mu_v, log_var_v, u_v, _, _, identity_grid) in enumerate(tqdm(data_loader)):
        im1, im2 = im1.to(device, non_blocking=True), im2.to(device, non_blocking=True)

        mu_v = mu_v.to(device, non_blocking=True).requires_grad_(True)
        log_var_v, u_v = log_var_v.to(device, non_blocking=True).requires_grad_(True), \
                         u_v.to(device, non_blocking=True).requires_grad_(True)

        identity_grid = identity_grid.to(device, non_blocking=True).requires_grad_(False)

        file_path = os.path.join(data_loader.save_dirs['im2_warped'], 'im1_' + str(batch_idx) + '.nii.gz')
        save_to_disk(im1, file_path)
        file_path = os.path.join(data_loader.save_dirs['im2_warped'], 'im2_' + str(batch_idx) + '.nii.gz')
        save_to_disk(im2, file_path)

        """
        initialise the optimiser
        """

        optimizer_v = config.init_obj('optimizer_v', torch.optim, [mu_v, log_var_v, u_v])

        """
        optimise mu_v, log_var_v, and u_v on data
        """

        for iter_no in range(no_steps_v):
            optimizer_v.zero_grad()
            data_term = 0.0

            for _ in range(no_samples):
                v_sample = sample_qv(mu_v, log_var_v, u_v)
                warp_field = transformation_model.forward_3d_add(identity_grid, v_sample)

                im2_warped = registration_module(im2, warp_field)
                im_out = enc(im1, im2_warped)

                data_term_sample = data_loss(im_out).sum() / float(no_samples)
                data_term += data_term_sample

            reg_term = reg_loss(mu_v).sum()
            entropy_term = entropy(log_var_v, u_v).sum()

            loss_qv = data_term + reg_term - entropy_term
            loss_qv.backward()
            optimizer_v.step()

            # if iter_no == 0 or iter_no % 16 == 0 or iter_no == no_steps_v - 1:
            print(f'ITERATION ' + str(iter_no) + '/' + str(no_steps_v - 1) +
                  f', TOTAL ENERGY: {loss_qv.item():.2f}' +
                  f'\ndata: {data_term.item():.2f}' +
                  f', regularisation: {reg_term.item():.2f}' +
                  f', entropy: {entropy_term.item():.2f}'
                  )

            """
            save the warped moving image to disk
            """

            with torch.no_grad():
                warp_field = transformation_model.forward_3d_add(identity_grid, mu_v)
                im2_warped = registration_module(im2, warp_field)

                file_path = os.path.join(data_loader.save_dirs['im2_warped'], 'im2_warped_' + str(batch_idx) + '_' + str(iter_no) + '.nii.gz')
                save_to_disk(im2_warped, file_path, True)

    """
    calculate the metrics
    """

    with torch.no_grad():
        warp_field = transformation_model.forward_3d_add(identity_grid, mu_v)
        im2_warped = registration_module(im2, warp_field)
        im_out = enc(im1, im2_warped)

        data_term = data_loss(im_out).sum()
        reg_term = reg_loss(mu_v).sum()
        entropy_term = entropy(log_var_v, u_v).sum()

    total_loss = data_term + reg_term - entropy_term
    log = {'loss': total_loss.item(), 'SSD': data_term.item(), 'reg': reg_term.item(), 'entropy': entropy_term.item()}
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='LearnSim')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
