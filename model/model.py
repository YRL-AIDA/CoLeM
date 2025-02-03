import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers import BertPreTrainedModel

from config import Config


class Colem(BertPreTrainedModel):
    """Contrastive learning model for table understanding tasks."""
    def __init__(self, config):
        super().__init__(config)

        # Configuration
        self.model_config = Config()["model"]
        pretrained_model_name = self.model_config.get("pretrained_model_name")
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        # Layers
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.projector = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.model_config.get("loss_latent_space_dim")
            )
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """CoLeM forward pass.

        Args:
            input (torch.Tensor): batch(2 * batch_size, sequence_length) of table columns.

        Returns:
            torch.Tensor: Model output.
        """
        encoder_last_hidden_state = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]  # (2 * batch_size, sequence_length, encoder_output)
        projector_output = self.projector(encoder_last_hidden_state)  # (2 * batch_size, sequence_length, reduced_output)

        # Get column representations, i.e. encoder [CLS] token positions.
        output = projector_output[:, 0, :]  # (2 * batch_size, reduced_output)
        return output
