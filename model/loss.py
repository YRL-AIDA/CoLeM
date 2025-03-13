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
    hidden_size = model_output.shape[1]

    # Cosine similarity
    similarity_matrix = F.cosine_similarity(
        model_output.reshape(1, batch_size, hidden_size),
        model_output.reshape(batch_size, 1, hidden_size),
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
