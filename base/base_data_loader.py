from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, no_GPUs, no_workers, rank):
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': no_workers, 'pin_memory': True}
        sampler = DistributedSampler(dataset, num_replicas=no_GPUs, rank=rank, shuffle=shuffle)

        super().__init__(sampler=sampler, **init_kwargs)

        if rank == 0:
            dataset.dataset.write_idx_to_ID_json()

    @property
    def atlas_mode(self):
        return self.dataset.dataset.atlas_mode

    @property
    def dims(self):
        return self.dataset.dataset.dims

    @property
    def im_spacing(self):
        return self.dataset.dataset.im_spacing

    @property
    def no_samples(self):
        return len(self.dataset)

    @property
    def save_dirs(self):
        return self.dataset.dataset.save_paths

    @property
    def structures_dict(self):
        return self.dataset.dataset.structures_dict
