import numpy as np
import pandas as pd
import torch

from dataset.table_dataloader import TableDataLoader
from dataset.table_dataset import TableDataset

from logs.logger import Logger
from model.loss import nt_xent_loss
from model.model import Colem

from transformers import AutoTokenizer, AutoConfig

from config import Config
from trainer.trainer import Trainer
from utils.functions import cleanup, collate, setup

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler


def train(rank: int, world_size: int, config: Config):
    if config["ddp"].get("use"):
        print(f"Running train DDP on rank {rank}.")
        setup(rank, world_size, config)

    pretrained_model_name = config["model"].get(
        "pretrained_model_name",
        "bert-base-multilingual-uncased"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    train_dataset = TableDataset(
        data_dir=config["data"].get("train_dir"),
        sep=config["data"].get("sep", "|"),
        engine=config["data"].get("engine", "c"),
        quotechar=config["data"].get("quotechar", "\""),
        on_bad_lines=config["data"].get("on_bad_lines", "skip"),
        num_rows=config["data"].get("num_rows")
    )
    if config["ddp"].get("use"):
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = SubsetRandomSampler(np.arange(len(train_dataset.df)))

    eval_dataset = TableDataset(
        data_dir=config["data"].get("eval_dir"),
        sep=config["data"].get("sep", "|"),
        engine=config["data"].get("engine", "c"),
        quotechar=config["data"].get("quotechar", "\""),
        on_bad_lines=config["data"].get("on_bad_lines", "skip"),
        num_rows=config["data"].get("num_rows")
    )
    eval_sampler = SubsetRandomSampler(np.arange(len(eval_dataset.df)))

    batch_size = config["train"].get("batch_size", 64)
    num_workers = config["data"].get("num_workers", 0)
    train_dataloader = TableDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=num_workers
    )
    eval_dataloader = TableDataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        collate_fn=collate,
        num_workers=num_workers
    )

    model = Colem(AutoConfig.from_pretrained(pretrained_model_name))
    if config["ddp"].get("use"):
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
       device = rank
       model.to(device)
       model = DDP(model, device_ids=[rank])
    else:
        device = torch.device('cuda:0' if torch.cuda.device_count() == 0 else 'cpu')
        model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("optim_lr", 5e-5),
        eps=config["train"].get("optim_eps", 1e-6),
        fused=True
    )
    num_epochs = config["train"].get("num_epochs", 100)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        loss_fn=nt_xent_loss,
        optimizer=optimizer,
        config=config,
        device=device,
        batch_size=batch_size,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        train_logger=Logger(config, filename=config["logs"].get("train_filename", "train.log")),
        eval_logger=Logger(config, filename=config["logs"].get("validation_filename", "valid.log"))
    )

    trainer.train()
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


if __name__ == "__main__":
    config = Config()
    
    world_size = torch.cuda.device_count()
    if config["ddp"].get("use"):  # TODO: change to True to see
        mp.spawn(
            train,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        losses = train(None, world_size, config)
        print(losses)
