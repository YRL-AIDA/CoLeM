import torch


from dataset.augmenter import Augmenter


def collate(samples: dict) -> dict:
    """Preprocess data by batch.

    Args:
        samples: Samples from batch.

    Returns:
        dict: Augmented columns.
    """
    data = torch.Tensor(map(Augmenter.drop_cells, samples["data"]))
    labels = torch.cat([sample["labels"] for sample in samples])

    batch = {"data": data, "labels": labels}
    return batch



if __name__ == "__main__":
    pass
