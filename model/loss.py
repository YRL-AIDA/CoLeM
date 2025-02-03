import torch
from torch.nn import functional as F


def nt_xent_loss(model_output: torch.Tensor, device: torch.device, temperature: float) -> torch.Tensor:
    """Calculate NT-Xent loss.

    NT-Xent loss (Normalized temperature Cross-entropy loss) was introduced in SimCLR paper.

    Args:
        model_output (torch.Tensor): Model output
        temperature (float): Loss temperature

    Returns:
        torch.Tensor: NT-Xent loss
    """
    assert len(model_output.size()) == 2
    batch_size = model_output.shape[0]

    # Cosine similarity
    similarity_matrix = F.cosine_similarity(
        model_output.reshape(1, model_output.size()[0], model_output.size()[1]),
        model_output.reshape(model_output.size()[0], 1, model_output.size()[1]),
        dim=-1
    )

    # Discard main diagonal
    similarity_matrix[torch.eye(batch_size).bool()] = float("-inf")

    # Labels
    labels = torch.arange(batch_size).to(device)
    labels[0::2] += 1
    labels[1::2] -= 1

    # Compute cross entropy loss
    return F.cross_entropy(similarity_matrix / temperature, labels, reduction="mean")
