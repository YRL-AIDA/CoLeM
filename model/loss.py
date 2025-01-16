import torch
from torch.nn import functional as F


def nt_xent_loss(model_output: torch.Tensor, device: torch.device, temperature: float = 0.5) -> torch.Tensor:
    """Calculate NT-Xent loss.

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


if __name__ == "__main__":
    # TODO: move to tests
    torch.manual_seed(42)

    batch_size = 32
    hidden_size = 128
    batch = torch.randn(batch_size, hidden_size)
    for t in (0.001, 0.01, 0.1, 1.0, 10.0):
        print(f"Temperature: {t:.3f}, Loss: {nt_xent_loss(batch, temperature=t)}")
