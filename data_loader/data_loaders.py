from base import BaseDataLoader
from .datasets import BiobankDataset


class LearnSimDataLoader(BaseDataLoader):
    def __init__(self, batch_size, data_dir, dims, num_workers, save_dirs=None):
        self.data_dir = data_dir
        self.save_dirs = save_dirs

        dataset = BiobankDataset(data_dir, save_dirs, dims)
        super().__init__(batch_size, dataset, num_workers)

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
    def spacing(self):  # TODO (DG): change the name to "voxel_spacing"
        return self.dataset.spacing

    @property
    def var_params_q_f(self):
        return self.dataset.var_params_q_f

