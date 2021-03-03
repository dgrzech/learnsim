from base import BaseDataLoader
from .datasets import BiobankDataset


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, batch_size, dims, num_workers, data_dir, save_dirs, rescale_im=False, shuffle=True,
                 no_GPUs=1, rank=0):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        dataset = BiobankDataset(dims, data_dir, save_dirs, rescale_im, rank)
        super().__init__(batch_size, dataset, no_GPUs, num_workers, shuffle, rank)

    @property
    def dims(self):
        return self.dataset.dims

    @property
    def fixed(self):
        return self.dataset.fixed

    @property
    def no_samples(self):
        return len(self.dataset)

    @property
    def im_spacing(self):
        return self.dataset.im_spacing
