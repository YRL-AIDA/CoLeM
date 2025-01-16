from typing import Callable, Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from transformers import BertPreTrainedModel, BertTokenizer

from datetime import datetime

from config import Config
from logs.logger import Logger
from utils.functions import get_map_location


class Trainer:
    """Model trainer.

    Encapsulates training logic, saving and loading from checkpoints.

    Args:
        model: Training model.
        tokenizer: Tokenizer.
        loss_fn: Loss function.
        optimizer: GD optimizer.
        config: Training configuration.
        device: Torch device.
        batch_size: Size of batch.
        dataloader: Dataloader of train subset.
        lr_scheduler: Learning rate scheduler.
        num_epochs: Total number of epochs.
        logger: Filesystem logger.
    """
    def __init__(
            self,
            model: BertPreTrainedModel,
            tokenizer: BertTokenizer,
            loss_fn: Callable,
            optimizer: Optimizer,
            config: Config,
            device: torch.device,
            batch_size: int,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            lr_scheduler: Any = None,
            num_epochs: int = 1,
            train_logger: Logger = None,
            valid_logger: Logger = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.config = config
        self.device = device

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = batch_size

        self.lr_scheduler = lr_scheduler

        self.num_epochs = num_epochs
        self.start_epoch = 0
        self.validation_period_epochs = config["train"]["validation_period_epochs"]
        self.save_period_epochs = config["train"]["save_period_epochs"]

        self.train_logger = train_logger
        self.valid_logger = valid_logger

        self.losses = {
            "train": [],
            "validation": []
        }

        self.checkpoint_dir = config["train"]["checkpoints_dir"]
        if config["train"]["start_from_checkpoint"]:
            checkpoint = config["train"]["checkpoint_name"]
            assert checkpoint is not None
            self._load_checkpoint(self.checkpoint_dir + checkpoint)

    def train(self) -> dict:
        """Train model by epochs.

        Returns:
            tuple: Tuple of losses during training.
        """
        self.train_logger.info("--- New trainer initialized ---", "TRAINER")
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_logger.info(f"Epoch #{epoch} started.", "EPOCH")

            train_loss = self._train_epoch()
            self.losses["train"].append(train_loss["loss"])
            self.train_logger.info(
                f"Epoch {epoch}. Train loss: {train_loss['loss']}.",
                "LOSS"
            )

            if epoch % self.validation_period_epochs == 0:
                self.train_logger.nvidia_smi()

                valid_loss = self._validate_epoch()
                self.losses["validation"].append(valid_loss["loss"])
                self.valid_logger.info(
                    f"Epoch {epoch}. Valid loss: {valid_loss['loss']}.",
                    "LOSS"
                )

            if (epoch + 1) % self.save_period_epochs == 0:
                self._save_checkpoint(
                    epoch,
                    self.losses,
                )
                self.train_logger.info(
                    f"Epoch {epoch}. Model has been saved.",
                    "PERIODIC_SAVED"
                )
            self.train_logger.info("--- --- ---", "EPOCH")
        self.train_logger.info(f"Training successfully ended.", "TRAINER")
        return self.losses

    def _train_epoch(self) -> dict:
        """Train epoch.

        Returns:
            dict: Dictionary of training epoch loss.
        """
        self.model.train()

        running_loss = 0.0
        for batch in self.train_dataloader:
            batch = batch.to(self.device)
            attention_mask = torch.clone(batch != 0).to(self.device)
            
            output = self.model(batch, attention_mask=attention_mask)

            loss = self.loss_fn(output, self.device)
            running_loss += loss.item()
            loss.backward()

            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)  # set_to_none is more efficient
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return {"loss": running_loss / self.batch_size}

    def _validate_epoch(self) -> dict:
        """Validate epoch.

        Returns:
            dict: Dictionary of validation epoch loss and metrics.
        """
        self.model.eval()

        running_loss = 0.0
        with torch.no_grad():
            for batch in self.valid_dataloader:
                batch = batch.to(self.device)
                attention_mask = torch.clone(batch != 0)
                attention_mask.to(self.device)

                output = self.model(batch, attention_mask=attention_mask)
                # TODO: why it can return tuple(tensor), except for just tensor?
                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss_fn(output, self.device)
                running_loss += loss.item()
        return {
            "loss": running_loss / self.batch_size,
        }

    def _save_checkpoint(
            self,
            epoch: int,
            losses: dict,
    ) -> None:
        """Save model checkpoint.

        Model is saved in `config['checkpoints_dir']` directory.

        Parameters to be saved:
            - number of epoch
            - model state dict
            - optimizer state dict
            - train/validation losses

        Args:
            epoch: Number of current epoch.
            losses: Dictionary of model's train / validation losses.

        Returns:
            None
        """
        checkpoint_path = (
            f"{self.checkpoint_dir}model_epoch_{epoch}_"
            f"datetime-{datetime.now():%d-%m-%y_%H-%M-%S}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "losses": losses,
            },
            checkpoint_path
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Initialize model from checkpoint for further training.

        Args:
            checkpoint_path: Checkpoint filename in `checkpoints` directory.

        Returns:
            None
        """
        checkpoint = torch.load(checkpoint_path, map_location=get_map_location())
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.losses = checkpoint["losses"]
