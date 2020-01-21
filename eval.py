from tqdm import tqdm

from logger import registration_print, save_images
from model.metric import dice
from parse_config import ConfigParser
from utils import get_module_attr, sample_qv

import argparse
import torch

import data_loader.data_loaders as module_data
import model.loss as model_loss
import model.model as module_arch
import utils.registration as registration
import utils.transformation as transform


def main(config):
    logger = config.get_logger('test')

    log_step = config['trainer']['log_step']
    learn_sim_metric = config['trainer']['learn_sim_metric']

    # data loader instance
    data_loader = config.init_obj('data_loader', module_data, save_dirs=config.save_dirs)

    # build the model architecture
    enc = None

    if config['trainer']['learn_sim_metric']:
        enc = config.init_obj('arch', module_arch)
        logger.info(enc)

    # initialise the transformation model and registration modules
    dim_x = config['data_loader']['args']['dim_x']
    dim_y = config['data_loader']['args']['dim_y']
    dim_z = config['data_loader']['args']['dim_z']

    transformation_model = config.init_obj('transformation_model', transform, dim_x, dim_y, dim_z)
    registration_module = config.init_obj('registration_module', registration)

    # initialise the losses
    data_loss = config.init_obj('data_loss', model_loss)
    reg_loss = config.init_obj('reg_loss', model_loss)
    entropy = config.init_obj('entropy', model_loss)

    # prepare model for testing
    logger.info('loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        transformation_model = torch.nn.DataParallel(transformation_model)
        registration_module = torch.nn.DataParallel(registration_module)

        if enc is not None:
            enc = torch.nn.DataParallel(enc)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if enc.is not None:
        enc.load_state_dict(state_dict)
        enc = enc.to(device)

        enc.eval()
        get_module_attr(enc, 'set_grad_enabled')(False)

    transformation_model = transformation_model.to(device)
    registration_module = registration_module.to(device)

    transformation_model.eval()
    registration_module.eval()

    """
    test
    """

    no_steps_v = config['trainer']['no_steps_v']
    no_samples = config['trainer']['no_samples']

    for batch_idx, (im_pair_idxs, im_fixed, seg_fixed, im_moving, seg_moving,
                    mu_v, log_var_v, u_v, log_var_f, u_f) in enumerate(tqdm(data_loader)):
        curr_batch_size = float(im_pair_idxs.numel())

        im_fixed, im_moving = im_fixed.to(device, non_blocking=True), \
                              im_moving.to(device, non_blocking=True)  # images to register

        seg_fixed, seg_moving = seg_fixed.to(device, non_blocking=True), \
                                seg_moving.to(device, non_blocking=True)  # corresponding segmentations

        mu_v = mu_v.to(device, non_blocking=True).requires_grad_(True)  # mean velocity field
        log_var_v, u_v = log_var_v.to(device, non_blocking=True).requires_grad_(True), \
                         u_v.to(device, non_blocking=True).requires_grad_(True)  # variational parameters

        # initialise the optimiser
        optimizer_v = config.init_obj('optimizer_v', torch.optim, [mu_v, log_var_v, u_v])

        """
        optimise q_v
        """

        for iter_no in range(no_steps_v):
            optimizer_v.zero_grad()
            data_term = 0.0

            for _ in range(no_samples):
                v_sample = sample_qv(mu_v, log_var_v, u_v)
                transformation, displacement = transformation_model(v_sample)
                im_moving_warped = registration_module(im_moving, transformation)

                if learn_sim_metric:
                    im_out = enc(im_fixed, im_moving_warped)
                    data_term_sample = data_loss(None, None, im_out).sum() / float(no_samples)
                else:
                    data_term_sample = data_loss(im_fixed, im_moving_warped).sum() / float(no_samples)

                data_term += data_term_sample
                reg_term = reg_loss(v_sample).sum() / float(no_samples)

            entropy_term = entropy(log_var_v, u_v).sum()

            loss_qv = data_term + reg_term + entropy_term
            loss_qv.backward()
            optimizer_v.step()

            if iter_no % log_step == 0:
                registration_print(logger, iter_no, no_steps_v, loss_qv.item() / curr_batch_size,
                                   data_term.item() / curr_batch_size, reg_term.item() / curr_batch_size,
                                   entropy_term.item() / curr_batch_size)

        # save the images
        with torch.no_grad():
            transformation, displacement = transformation_model(mu_v)

            im_moving_warped = registration_module(im_moving, transformation)
            seg_moving_warped = registration_module(seg_moving, transformation, mode='nearest')

            save_images(im_pair_idxs, data_loader.save_dirs, im_fixed, im_moving, im_moving_warped,
                        mu_v, log_var_v, u_v, log_var_f, u_f, displacement,
                        seg_fixed, seg_moving, seg_moving_warped)

            logger.info('\nsaved the output images and vector fields to disk\n')

    with torch.no_grad():
        transformation, displacement = transformation_model(mu_v)
        im_moving_warped = registration_module(im_moving, transformation)

        if learn_sim_metric:
            im_out = enc(im_fixed, im_moving_warped)
            data_term = data_loss(None, None, im_out).sum() / curr_batch_size
        else:
            data_term = data_loss(im_fixed, im_moving_warped).sum() / curr_batch_size

        reg_term = reg_loss(mu_v).sum() / curr_batch_size
        entropy_term = entropy(log_var_v, u_v).sum() / curr_batch_size

        seg_moving_warped = registration_module(seg_moving, transformation, mode='nearest')
        dsc = dice(seg_fixed, seg_moving_warped)

    total_loss = data_term + reg_term + entropy_term

    log = {'loss': total_loss.item(),
           'data': data_term.item(), 'reg': reg_term.item(), 'entropy': entropy_term.item()}

    for class_idx, val in enumerate(dsc):
        log['dice_' + str(class_idx + 1)] = val
    
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
