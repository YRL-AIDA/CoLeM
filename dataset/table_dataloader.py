from typing import Optional

from torch.utils.data import DataLoader

from dataset.table_dataset import TableDataset

from torch.utils.data import Sampler


class TableDataLoader(DataLoader):
    """Table dataloader.

    Provides an iterable over the given dataset.

    Args:
        dataset: dataset from which to load the data.
        batch_size: how many samples per batch to load.
        sampler: dataset sampler.
        collate_fn: function that merges a list of samples to form a mini-batch of Tensors.
        num_workers: how many subprocesses to use for data loading.
    """
    def __init__(
        self,
        dataset: TableDataset,
        batch_size: int,
        sampler: Sampler,
        collate_fn: Optional[callable] = None,
        num_workers: Optional[int] = 0,
    ):
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            "sampler": sampler,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
