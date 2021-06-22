from base import BaseDataLoader

from .biobank_dataset import BiobankDataset
from .oasis_dataset import OasisDataset


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, save_dirs, im_pairs, dims, sigma_v_init, u_v_init, cps=None,
                 batch_size=1, no_GPUs=1, no_workers=0, rank=0, test=False):
        dataset = BiobankDataset(save_dirs, im_pairs, dims, sigma_v_init=sigma_v_init, u_v_init=u_v_init, cps=cps)
        shuffle = not test
        super().__init__(dataset, batch_size, no_GPUs, no_workers, rank, shuffle)

    @property
    def fixed(self):
        return self.dataset.fixed


class Learn2RegDataLoader(BaseDataLoader):
    def __init__(self, data_dir, save_dirs, im_pairs, dims, sigma_v_init, u_v_init, cps=None,
                 batch_size=1, no_GPUs=1, no_workers=0, rank=0, test=False):
        dataset = OasisDataset(save_dirs, im_pairs, dims, sigma_v_init=sigma_v_init, u_v_init=u_v_init, cps=cps)
        shuffle = not test
        super().__init__(dataset, batch_size, no_GPUs, no_workers, rank, shuffle)
