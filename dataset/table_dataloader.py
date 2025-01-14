from typing import Optional

from torch.utils.data import DataLoader

from dataset.table_dataset import TableDataset


class TableDataLoader(DataLoader):
    """Table data loader.

    Provides an iterable over the given dataset.

    Args:
        dataset: dataset from which to load the data.
        batch_size: how many samples per batch to load.
        shuffle: perform shuffling of dataset.
        collate_fn: merges a list of samples to form a mini-batch of Tensors.
        num_workers: how many subprocesses to use for data loading.
    """
    def __init__(
        self,
        dataset: TableDataset,
        batch_size: int,
        shuffle: bool = True,
        collate_fn: Optional[callable] = None,
        num_workers: Optional[int] = 0,
    ):
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
