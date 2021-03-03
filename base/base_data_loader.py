from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, batch_size, dataset, no_GPUs, num_workers, shuffle, rank):
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': num_workers, 'pin_memory': True}
        sampler = DistributedSampler(dataset, num_replicas=no_GPUs, rank=rank, shuffle=shuffle) if no_GPUs > 1 else None

        super().__init__(sampler=sampler, **init_kwargs)
