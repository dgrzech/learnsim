import torch

from base import BaseDataLoader
from .datasets import BiobankDataset


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, batch_size, dims, num_workers, data_dir, save_dirs, sigma_v_init, u_v_init, cps=None,
                 rescale_im=False, shuffle=True, no_GPUs=1, rank=0, test_split=0.2, test=False):
        self.data_dir, self.save_dirs = data_dir, save_dirs
        generator = torch.Generator().manual_seed(42)

        dataset = BiobankDataset(dims, data_dir, save_dirs, sigma_v_init, u_v_init, cps, rescale_im, rank)

        self.dims = dataset.dims
        self.fixed = dataset.fixed
        self.im_spacing = dataset.im_spacing

        train_len, test_len = (1 - test_split) * len(dataset), test_split * len(dataset)
        train_data, test_data = torch.utils.data.random_split(dataset, [int(train_len), int(test_len)], generator=generator)
        dataset = train_data if not test else test_data

        super().__init__(batch_size, dataset, no_GPUs, num_workers, shuffle, rank)

    def dims(self):
        return self.dims

    def fixed(self):
        return self.fixed

    @property
    def no_samples(self):
        return len(self.dataset)

    def im_spacing(self):
        return self.im_spacing
