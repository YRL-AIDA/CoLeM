from typing import Optional, Union
import numpy as np
import pandas as pd
from config import Config
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.augmenter import Augmenter
from dataset.table_dataset import TableDataset


def preprocess_delete_me(dataset: TableDataset) -> None:
    """TODO: delete me when preprocess will perform on creating dataset step."""
    dataset.df = dataset.df[~dataset.df["column_data"].isna()]


def collate(samples: list) -> torch.Tensor:
    """Preprocess data by batch.

    For every sample in the mini-batch with `N` samples create a pair, resulting
    `2 * N` mini-batch.

    Upply the first augmentation on the first element of the pair in mini-batch,
    and upply the second augmentation on the second element of the pair.

    Tokenize every sample in the mini-batch with BERT tokenizer. And pad every
    sample to the length of maximum tokenized sequence in a tokenized mini-batch. 

    Args:
        samples (list): `N` samples from mini-batch.

    Returns:
        torch.Tensor: Preprocessed mini-batch with `2 * N` samples.
    """
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config["model"]["pretrained_model_name"])
    batch_size = len(samples)

    augmented_batch = [0 for _ in range(2 * batch_size)]
    augmented_pointer = 0
    for sample in samples:
        # augmentations
        first_augmentation = Augmenter.drop_cells(sample)
        second_augmentation = Augmenter.shuffle_rows(sample)
        
        # tokenization
        first_augmentation = tokenizer.encode(
            first_augmentation,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).squeeze()
        second_augmentation = tokenizer.encode(
            second_augmentation,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).squeeze()

        # write tensors to augmented batch
        augmented_batch[augmented_pointer] = first_augmentation
        augmented_batch[augmented_pointer + 1] = second_augmentation
        augmented_pointer += 2
    
    # pad tensors to maximum sequence length in a batch
    pad_tensors(augmented_batch)

    return torch.stack(augmented_batch, dim=0)


def pad_tensors(batch: list) -> None:
    """Pad tensors in a batch to maximum length.

    Gets maximum sequence length and pads all tensors to `max_len` with zeros
    on the right side.
    
    Note:
        Padding performed inplace.

    Args:
        batch: List of tensors to be padded.
    
    Retruns:
        None
    """
    seq_max_len = get_max_seq_length(batch)
    for i in range(len(batch)):
        batch[i] = F.pad(
            input=batch[i],
            pad=(0, seq_max_len - batch[i].shape[0]),
            mode="constant",
            value=0
        )


def get_max_seq_length(batch: list) -> int:
    """Get maximum sequence length in a batch.
    
    Args:
        batch: List of tensors.
    
    Returns:
        int: Maximum length of tensor in a batch.
    """
    seq_max_len_in_batch = 0
    for sample in batch:
        cur_seq_len = sample.shape[0]
        if cur_seq_len > seq_max_len_in_batch:
            seq_max_len_in_batch = cur_seq_len
    return seq_max_len_in_batch


def prepare_device(n_gpu_use: int) -> tuple[torch.device, list]:
    """Prepare GPUs for training.

    Note:
        Supports multiple GPUs.

    Args:
        n_gpu_use: Number of GPUs to prepare.

    Returns:
        tuple: Device and list of available GPUs, if machine have multiple GPUs available.
    """
    num_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and num_gpu == 0:
        print("Warning: No GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > num_gpu:
        print(f"Warning: The number of GPU configured to use is {n_gpu_use}, but only {num_gpu} are available")
        n_gpu_use = num_gpu

    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_map_location() -> Optional[torch.device]:
    """Get device to perform model loading.

    Returns:
        Optional[torch.device]: device to perform load function.
    """
    map_location = None
    if not torch.cuda.is_available():
        map_location = torch.device("cpu")
    return map_location


def set_rs(seed: int) -> None:
    """Set random seed.

    Args:
        seed: Random seed.

    Returns:
        None
    """
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_samplers(dataset: pd.DataFrame, split: Union[float, int]) -> tuple:
    """Create train and validation samplers with respect to split parameter.

    Shuffle dataset ids and split into train and validation subsets.

    Note:
        `split` parameter could be `int`, then in validation subset would have `split` samples,
        and `split` could be `float`, then validation subset would have `split` % of `len(dataset)`.
    
    Args:
        dataset (pd.DataFrame): Dataset to be sampled.
        split (float|int): Validation split size.
    """
    if isinstance(split, int):
        assert 0 < split < len(dataset)
        valid_size = split
    else:
        valid_size = int(len(dataset) * split)

    dataset_ids = np.arange(len(dataset))

    valid_ids = np.random.choice(dataset_ids, size=valid_size, replace=False)
    train_ids = np.setdiff1d(dataset_ids, valid_ids)
    np.random.shuffle(train_ids)

    # TODO: move to tests
    assert len(np.intersect1d(train_ids, valid_ids)) == 0
    return SubsetRandomSampler(train_ids), SubsetRandomSampler(valid_ids)
