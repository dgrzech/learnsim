import numpy as np
import torch
import torch.distributed as dist

from utils import rescale_im_intensity


def get_im_or_field_mid_slices_idxs(im_or_field):
    if len(im_or_field.shape) == 3:
        return int(im_or_field.shape[2] / 2), int(im_or_field.shape[1] / 2), int(im_or_field.shape[0] / 2)
    elif len(im_or_field.shape) == 4:
        return int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2), int(im_or_field.shape[1] / 2)
    elif len(im_or_field.shape) == 5:
        return int(im_or_field.shape[4] / 2), int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2)

    raise NotImplementedError


def get_im_or_field_mid_slices(im_or_field):
    mid_idxs = get_im_or_field_mid_slices_idxs(im_or_field)

    if len(im_or_field.shape) == 3:
        return [im_or_field[:, :, mid_idxs[0]],
                im_or_field[:, mid_idxs[1], :],
                im_or_field[mid_idxs[2], :, :]]
    elif len(im_or_field.shape) == 4:
        return [im_or_field[:, :, :, mid_idxs[0]],
                im_or_field[:, :, mid_idxs[1], :],
                im_or_field[:, mid_idxs[2], :, :]]
    if len(im_or_field.shape) == 5:
        return [im_or_field[:, :, :, :, mid_idxs[0]],
                im_or_field[:, :, :, mid_idxs[1], :],
                im_or_field[:, :, mid_idxs[2], :, :]]

    raise NotImplementedError


def log_model_samples(writer, output_dict):
    if dist.get_rank() == 0:
        body_axes = ['sagittal', 'coronal', 'axial']

        positive_samples_slices = get_im_or_field_mid_slices(output_dict['positive_samples_mean'])
        negative_samples_slices = get_im_or_field_mid_slices(output_dict['negative_samples_mean'])

        for slice_idx, body_axis_name in enumerate(body_axes):
            writer.add_images(f'positive_samples_mean/{body_axis_name}', rescale_im_intensity(positive_samples_slices[slice_idx]))
            writer.add_images(f'negative_samples_mean/{body_axis_name}', rescale_im_intensity(negative_samples_slices[slice_idx]))


def log_model_weights(writer,  model):
    if dist.get_rank() == 0:
        for name, p in model.named_parameters():
            writer.add_histogram(name, p, bins=np.arange(-1.5, 1.5, 0.1))


def log_images(writer, output_dict):
    if dist.get_rank() == 0:
        body_axes = ['sagittal', 'coronal', 'axial']

        for im_name, im in output_dict.items():
            if im.size(1) == 3 and 'sample_transformation' not in im_name:
                output_dict[im_name] = torch.norm(im, p=2, dim=1, keepdim=True)

        output_dict = {im_name: get_im_or_field_mid_slices(im) for im_name, im in output_dict.items()}
        output_dict['sample_transformation'] = [output_dict['sample_transformation'][0][:, 2:3, ...],
                                                output_dict['sample_transformation'][1][:, 1:2, ...],
                                                output_dict['sample_transformation'][2][:, 0:1, ...]]

        for im_name in output_dict:
            for slice_idx, body_axis_name in enumerate(body_axes):
                writer.add_images(f'{im_name}/{body_axis_name}', rescale_im_intensity(output_dict[im_name][slice_idx]))
