from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset, batch_size, no_GPUs, no_workers, rank, shuffle):
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': no_workers, 'pin_memory': True}
        sampler = DistributedSampler(dataset, num_replicas=no_GPUs, rank=rank, shuffle=shuffle)
        super().__init__(sampler=sampler, **init_kwargs)

    @property
    def atlas_mode(self):
        return self.dataset.atlas_mode

    @property
    def dims(self):
        return self.dataset.dims

    @property
    def im_spacing(self):
        return self.dataset.im_spacing

    @property
    def no_samples(self):
        return len(self.dataset)

    @property
    def save_dirs(self):
        return self.dataset.save_paths

    @property
    def structures_dict(self):
        return self.dataset.structures_dict

