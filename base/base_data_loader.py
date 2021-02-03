from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, batch_size, dataset, num_workers):
        # NOTE (DG): shuffle needs to be set to false for the optimisers to save correctly
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': num_workers, 'pin_memory': True, 'shuffle': False}
        super().__init__(sampler=None, **init_kwargs)

