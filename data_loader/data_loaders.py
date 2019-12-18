from base import BaseDataLoader
from .datasets import BiobankDataset, RGBDDataset


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, save_dirs=None, shuffle=True,
                 validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        self.dataset = BiobankDataset(data_dir, save_dirs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
