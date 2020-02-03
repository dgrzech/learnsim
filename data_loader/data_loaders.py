from base import BaseDataLoader
from .datasets import BiobankDataset


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, dim_x, dim_y, dim_z, save_dirs=None, shuffle=False, num_workers=1):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        self.dataset = BiobankDataset(data_dir, save_dirs, dim_x, dim_y, dim_z)
        super().__init__(self.dataset, batch_size, shuffle, num_workers)
