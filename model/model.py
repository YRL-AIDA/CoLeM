import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from config import Config


class Colem(nn.Module):
    """Contrastive learning model for table understanding tasks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configuration
        self.model_config = Config()["model"]
        pretrained_model_name = self.model_config.get(
            "pretrained_model_name",
            "bert-base-uncased"
        )
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        # Layers
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.projector = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.model_config.get("loss_latent_space_dim", 128)
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """CoLeM forward pass.

        Args:
            input (torch.Tensor): batch(2 * batch_size, sequence_length) of columns

        Returns:
            torch.Tensor: Model output
        """
        encoder_last_hidden_state = self.encoder(input)[0]  # (2*batch_size, sequence_length, encoder_output)
        projector_output = self.projector(encoder_last_hidden_state)  # (2*batch_size, sequence_length, reduced_output)

        # Get column representations, i.e. encoder [CLS] token positions.
        output = projector_output[:, 0, :]  # (2 * batch_size, reduced_output)
        return output


if __name__ == "__main__":
    # Model arch
    model = Colem()
    print(model)

    # Test forward pass
    torch.manual_seed(42)

    x = torch.randint(0, 3, (2 * 32, 512))  # batch_size = 32 ; output = 512
    print(x.shape)

    loss = model(x)
    print(loss)
