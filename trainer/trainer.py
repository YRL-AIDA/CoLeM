from typing import Callable, Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from transformers import BertPreTrainedModel, BertTokenizer

from datetime import datetime

from config import Config
from logs.logger import Logger

import torch.distributed as dist


class Trainer:
    """Model trainer.

    Trainer encapsulates training logic, and performs saving and loading from checkpoints.

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
            world_size: int,
            batch_size: int,
            num_epochs: int,
            train_dataloader: DataLoader,
            eval_dataloader: DataLoader,
            train_logger: Logger,
            eval_logger: Logger,
            lr_scheduler: Any = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.config = config
        self.device = device
        self.world_size = world_size

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.batch_size = batch_size

        self.lr_scheduler = lr_scheduler

        self.num_epochs = num_epochs
        self.start_epoch = 0
        self.eval_period_epochs = config["train"].get("eval_period_epochs")
        self.save_period_epochs = config["train"].get("save_period_epochs")

        self.train_logger = train_logger
        self.eval_logger = eval_logger

        self.losses = {
            "train": [],
            "eval": []
        }

        self.do_log = self.device.type == "cpu" or self.device.index == 0

        self.checkpoint_dir = config["train"].get("checkpoints_dir", "checkpoints/")
        if config["train"].get("start_from_checkpoint", False):
            checkpoint = config["train"].get("checkpoint_name")
            assert checkpoint is not None
            self._load_checkpoint(self.checkpoint_dir + checkpoint)

    def train(self) -> dict:
        """Train model by epochs.

        Returns:
            dict: Dictionary of losses during training / evaluation.
        """
        if self.do_log:
            self.train_logger.info("--- New trainer initialized ---", "TRAINER")
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.do_log:
                self.train_logger.info(f"Epoch #{epoch} started.", "EPOCH")

            train_loss = self._train_epoch()
            self.losses["train"].append(train_loss["loss"])
            if self.do_log:
                self.train_logger.info(
                    f"Epoch {epoch}. Train loss: {train_loss['loss']}.",
                    "LOSS"
                )

            if self.do_log and epoch % self.eval_period_epochs == 0:
                self.train_logger.nvidia_smi()

                eval_loss = self._validate_epoch()
                self.losses["eval"].append(eval_loss["loss"])
                self.eval_logger.info(
                    f"Epoch {epoch}. Evaluation loss: {eval_loss['loss']}.",
                    "LOSS"
                )

            if self.do_log and epoch % self.save_period_epochs == 0:
                self._save_checkpoint(
                    epoch,
                    self.losses,
                )
                self.train_logger.info(
                    f"Epoch {epoch}. Model has been saved.",
                    "PERIODIC_SAVED"
                )
            if self.do_log:
                self.train_logger.info("--- --- ---", "EPOCH")
        if self.do_log:
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
            self.model.zero_grad(set_to_none=True)
            batch = batch.to(self.device)
            attention_mask = torch.clone(batch != 0).to(self.device)
            
            output = self.model(batch, attention_mask=attention_mask)

            loss = self.loss_fn(output, self.device, temperature=self.config["model"].get("loss_temperature"))
            running_loss += loss.item()
            loss.backward()

            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return {"loss": running_loss / self.batch_size}

    def _validate_epoch(self) -> dict:
        """Validate epoch.

        Returns:
            dict: Dictionary of evaluation epoch loss.
        """
        self.model.eval()

        running_loss = 0.0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = batch.to(self.device)
                attention_mask = torch.clone(batch != 0)
                attention_mask.to(self.device)

                output = self.model(batch, attention_mask=attention_mask)
                # TODO: why it can return tuple(tensor), except for just tensor?
                if isinstance(output, tuple):
                    output = output[0]

                loss = self.loss_fn(output, self.device, temperature=self.config["model"].get("loss_temperature"))
                running_loss += loss.item()
        return {"loss": running_loss / self.batch_size}

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
            - train/eval losses

        Args:
            epoch: Number of current epoch.
            losses: Dictionary of model's train / eval losses.

        Returns:
            None
        """
        checkpoint_path = (
            f"{self.checkpoint_dir}epoch-{epoch}_"
            f"{datetime.now():%d-%m-%Y_%H:%M:%S}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
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

        if self.world_size > 1:
            dist.barrier()
            map_location = {f"cuda:0": f"cuda:{self.device.index}"}
        else:
            map_location = self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        
        if self.world_size <= 1:
            consume_prefix_in_state_dict_if_present(checkpoint["model_state_dict"], "module.")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.losses = checkpoint["losses"]
