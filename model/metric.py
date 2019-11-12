import torch


def SSD(im_out):
    return 0.5 * torch.sum(im_out)
