import operator

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler

    https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/dataset
    /torch.py#L13
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""

        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """

        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """

        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
            Sampler is assumed to be of constant size.

    https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/sampler.py
    #L522
    """

    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True):
        """
        Args:
            sampler: Sampler used for subsampling

            num_replicas (int, optional): Number of processes participating in
              distributed training

            rank (int, optional): Rank of the current process
              within ``num_replicas``

            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """

        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler), num_replicas=num_replicas,
                                                        rank=rank, shuffle=shuffle)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)

        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset

        return iter(operator.itemgetter(*indexes_of_indexes)(subsampler_indexes))


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset, batch_size, no_workers, shuffle, no_im_pairs_per_epoch=None):
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': no_workers, 'pin_memory': True}

        try:
            no_samples = len(dataset) if no_im_pairs_per_epoch is None else no_im_pairs_per_epoch
            sampler_random = RandomSampler(dataset, replacement=True, num_samples=no_samples)

            rank = dist.get_rank()
            no_gpus = dist.get_world_size()
            sampler = DistributedSamplerWrapper(sampler_random, num_replicas=no_gpus, rank=rank, shuffle=shuffle)
        except:
            sampler = None

        super().__init__(sampler=sampler, **init_kwargs)

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

