import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Encoder layers
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        
        # Projector layers
        self.projector_lin1 = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size)
        self.projector_bn = nn.BatchNorm1d(num_features=self.config.hidden_size)
        self.projector_lin2 = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.model_config.get("loss_latent_space_dim")
        )
        
        # Initialize non-pretrained layers
        self.init_weights()

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
        
        x = F.relu(self.projector_lin1(encoder_last_hidden_state))
        x = self.projector_bn(x.permute(0, 2, 1))
        projector_output = self.projector_lin2(x.permute(0, 2, 1))

        # Get column representations, i.e. encoder [CLS] token positions.
        output = projector_output[:, 0, :]  # (2 * batch_size, reduced_output)
        return output
