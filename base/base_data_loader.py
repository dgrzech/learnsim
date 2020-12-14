from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset):
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.init_kwargs = {'dataset': dataset, 'pin_memory': True}

        super().__init__(sampler=None, **self.init_kwargs)
