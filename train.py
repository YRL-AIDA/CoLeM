import pandas as pd
import torch

from dataset.table_dataloader import TableDataLoader
from dataset.table_dataset import TableDataset

from logs.logger import Logger
from model.loss import nt_xent_loss
from model.model import Colem

from transformers import BertTokenizer, BertConfig, get_linear_schedule_with_warmup

from config import Config
from trainer.trainer import Trainer
from utils.functions import create_samplers, prepare_device, collate, preprocess_delete_me, set_rs


def train(config: Config):
    set_rs(config["random_state"])

    # TODO: assert config variables assigned and correct
    tokenizer = BertTokenizer.from_pretrained(config["model"]["pretrained_model_name"])

    dataset = TableDataset(
        num_rows=10,
        data_dir="data/"
    )
    preprocess_delete_me(dataset)

    train_sampler, valid_sampler = create_samplers(dataset.df, 0.1, 13)
    train_dataloader = TableDataLoader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate
    )
    valid_dataloader = TableDataLoader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        sampler=valid_sampler,
        num_workers=0,
        collate_fn=collate
    )

    model = Colem(
        BertConfig.from_pretrained(config["model"]["pretrained_model_name"]))

    device, device_ids = prepare_device(config["train"]["num_gpus"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        loss_fn=nt_xent_loss,
        optimizer=optimizer,
        config=config,
        device=device,
        batch_size=config["train"]["batch_size"], # TODO: STRANGE ALREADY HAVE IN DATALOADER
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        lr_scheduler=get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * config["train"]["num_epochs"]
        ),
        num_epochs=config["train"]["num_epochs"],
        train_logger=Logger(config, filename=config["logs"]["train_filename"]),
        valid_logger=Logger(config, filename=config["logs"]["validation_filename"])
    )
    return trainer.train()


if __name__ == "__main__":
    results = pd.DataFrame()

    config = Config(config_path="config.yaml")

    losses = train(config)

    # TODO: plot_graphs(losses, metrics, conf)

    print(losses)
