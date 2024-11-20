from typing import Optional, Union

import numpy as np
import pandas
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import BertTokenizer, PreTrainedTokenizerBase

from dataset import TableDataset

class CtaDataLoader(DataLoader):
    """Data loader.

    Combines train / validation samplers, and provides an iterable over
    the given dataset.

    Shuffles given dataset and splits into train / validation subsets.

    Args:
        dataset: dataset from which to load the data.
        batch_size: how many samples per batch to load.
        num_workers: how many subprocesses to use for data loading.
        collate_fn: merges a list of samples to form a mini-batch of Tensors.
    """
    def __init__(
            self,
            dataset: TableDataset,
            batch_size: int,
            split: Union[float, int] = 0.0,
            num_workers: Optional[int] = 0,
            collate_fn: Optional[callable] = None
    ):
        self.split = split
        self.num_samples = len(dataset)
        self.num_tables = dataset.df["table_id"].unique().shape[0]
        self.shuffle = False

        dataset_ids = np.arange(self.num_samples)
        np.random.shuffle(dataset_ids)
        if split == 0.0:
            self.train_sampler = SubsetRandomSampler(dataset_ids)
        else:
            self.train_sampler, self.valid_sampler = self._get_samplers(self.split, dataset_ids, dataset.df)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _get_samplers(
            self,
            split: Union[int, float],
            dataset_ids: np.ndarray,
            dataset: pandas.DataFrame
    ) -> tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """Create train / valid samplers.

        Args:
            split: Split size of the dataset, could be float (percentage of valid) or int (exact size of valid).
            dataset_ids: Dataframe rows ids.

        Returns:
            tuple: Train and valid random samplers.
        """
        if isinstance(split, int):
            assert 0 < split < self.num_tables
            len_valid = split
        else:
            len_valid = int(self.num_tables * split)

        valid_mask = pd.Series(dataset["table_id"].unique()).sample(n=len_valid, random_state=14)
        valid_df = dataset[dataset["table_id"].isin(valid_mask)]
        valid_ids = valid_df.index.to_numpy()

        train_df = dataset[~dataset["table_id"].isin(valid_mask)]
        train_ids = train_df.index.to_numpy()

        # valid_ids = dataset_ids[0:len_valid]
        # train_ids = np.delete(dataset_ids, np.arange(0, len_valid))
        self.num_samples = len(train_ids)

        return SubsetRandomSampler(train_ids), SubsetRandomSampler(valid_ids)

    def get_valid_dataloader(self) -> DataLoader:
        """Create dataloader of validation split."""
        assert self.valid_sampler is not None

        return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)



if __name__ == "__main__":
    pass


    # from config import Config
    # config = Config("./dataset/config.json")


    # t1 = TableDataset(
    #     data_dir="./" + config["dataset"]["data_dir"] + config["dataset"]["train_path"],
    #     # data_dir="",
	#    tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-uncased"),
    #     num_rows=100,
    # )
    # t = CtaDataLoader(dataset = t1, batch_size =32)
    # print(*t)
