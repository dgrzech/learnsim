from torch.utils.data import DataLoader, RandomSampler
from utils import DistributedSamplerWrapper


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset, batch_size, no_GPUs, no_workers, rank, shuffle, no_im_pair_per_epoch=None):
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': no_workers, 'pin_memory': True}

        if no_im_pairs_per_epoch is None:
            no_samples = len(dataset)
        else:
            no_samples = no_im_pairs_per_epoch

        sampler_random = RandomSampler(dataset, replacement=True, num_samples=no_samples)
        sampler_dist = DistributedSamplerWrapper(sampler_random, num_replicas=no_GPUs, rank=rank, shuffle=shuffle)
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

