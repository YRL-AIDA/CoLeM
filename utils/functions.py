from datetime import timedelta
import os
from typing import Union
import numpy as np
import pandas as pd
from config import Config
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.augmenter import Augmenter

import torch.distributed as dist


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
    aug_settings = config["data"]["augmenter"]
    pretrained_model_name = config["model"].get("pretrained_model_name")
    assert pretrained_model_name is not None

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    batch_size = len(samples)

    augmented_batch = [0 for _ in range(2 * batch_size)]
    augmented_pointer = 0
    for sample in samples:
        # augmentations
        first_augmentation = Augmenter.drop_cells(
            sample,
            sep=aug_settings.get("cells_sep"),
            ratio=aug_settings.get("cells_del_ratio")
        )
        second_augmentation = Augmenter.shuffle_rows(sample, sep=aug_settings.get("cells_sep"))
        
        # tokenization
        max_length = config["model"].get("tokenizer_max_len")
        first_augmentation = tokenizer.encode(
            first_augmentation,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).squeeze()
        second_augmentation = tokenizer.encode(
            second_augmentation,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).squeeze()

        # write tensors to augmented batch
        augmented_batch[augmented_pointer] = first_augmentation
        augmented_batch[augmented_pointer + 1] = second_augmentation
        augmented_pointer += 2
    
    # pad tensors to maximum sequence length in a batch
    pad_tensors(augmented_batch, config["model"].get("padding_value"))

    return torch.stack(augmented_batch, dim=0)


def pad_tensors(batch: list, padding_value: float) -> None:
    """Pad tensors in a batch to maximum length.

    Gets maximum sequence length and pads all tensors to `max_len` with zeros
    on the right side.
    
    Note:
        Padding performed inplace on the right side of the input tensor.

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
            value=padding_value
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


def get_available_device_count(num_gpus: int) -> tuple[torch.device, list]:
    """Get number of available gpus.

    Args:
        num_gpus: Number of GPUs to prepare.

    Returns:
        int: Number of available GPUs.
    """
    available_devices = num_gpus
    world_size = torch.cuda.device_count()
    if num_gpus > 0 and world_size == 0:
        print("Warning: No GPU available on this machine, training will be performed on CPU.")
        available_devices = 0
    if num_gpus > world_size:
        print(f"Warning: The number of GPU configured to use is {num_gpus}, but only {world_size} are available")
        available_devices = world_size
    return available_devices


def setup(rank, world_size, config):
    os.environ["MASTER_ADDR"] = config["ddp"].get("master_addr", "localhost")
    os.environ["MASTER_PORT"] = config["ddp"].get("master_port", "12355")
    dist.init_process_group(
        backend=config["ddp"]["backend"],
        timeout=timedelta(minutes=config["ddp"]["timeout"]),
        rank=rank,
        world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


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
