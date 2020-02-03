from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset, 'batch_size': batch_size, 'shuffle': self.shuffle,
            'collate_fn': collate_fn, 'num_workers': num_workers, 'pin_memory': True
        }

        super().__init__(sampler=None, **self.init_kwargs)
