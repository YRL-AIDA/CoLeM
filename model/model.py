"""TODO"""
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from config import Config


class Colem(nn.Module):
    """Contrastive learning model for table understanding tasks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_config = Config()["model"]

        pretrained_model_name = self.model_config.get(
            "pretrained_model_name", 
            "bert-base-uncased"
        )
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.projector = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.model_config.get("loss_latent_space_dim", 128)
            )
        )

    def forward(self):
        """TODO"""
        pass


if __name__ == "__main__":
    model = Colem()
    print(model)
