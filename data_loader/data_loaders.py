import torch
from torch.utils.data import Subset

from base import BaseDataLoader
from .biobank_dataset import BiobankDataset


def get_train_test_split(dataset, test_split):
    generator = torch.Generator().manual_seed(42)
    train_len, test_len = (1 - test_split) * len(dataset), test_split * len(dataset)
    train_data, test_data = torch.utils.data.random_split(dataset, [int(train_len), int(test_len)], generator=generator)

    return train_data, test_data


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, save_dirs, dims, sigma_v_init, u_v_init, cps=None,
                 batch_size=1, no_GPUs=1, no_workers=0, rank=0, test=False, test_split=None):
        dataset = BiobankDataset(dims, data_dir, save_dirs, sigma_v_init, u_v_init, cps)

        if test_split is not None:
            train_data, test_data = get_train_test_split(dataset, test_split)

            if test:
                dataset = test_data
                shuffle = False
            else:
                dataset = train_data
                shuffle = True

        super().__init__(dataset, batch_size, shuffle, no_GPUs, no_workers, rank)

    @property
    def fixed(self):
        return self.dataset.dataset.fixed


class Learn2RegDataLoader(BaseDataLoader):
    def __init__(self, data_dir, save_dirs, dims, sigma_v_init, u_v_init, cps=None,
                 batch_size=1, no_GPUs=1, no_workers=0, rank=0, test=False):
        dataset = OasisDataset(dims, data_dir, save_dirs, sigma_v_init, u_v_init, cps)

        if test:
            dataset = Subset(dataset, dataset.val_pairs_idxs)
            shuffle = False
        else:
            dataset = Subset(dataset, dataset.train_pairs_idxs)
            shuffle = True

        super().__init__(dataset, batch_size, shuffle, no_GPUs, no_workers, rank)
