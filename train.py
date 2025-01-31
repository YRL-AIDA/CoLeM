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

    pretrained_model_name = config["model"].get(
        "pretrained_model_name",
        "bert-base-multilingual-uncased"
    )

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    dataset = TableDataset(
        num_rows=config["data"].get("num_rows"),
        data_dir=config["data"].get("dir", "data/")
    )
    preprocess_delete_me(dataset)

    train_sampler, valid_sampler = create_samplers(dataset.df, 0.1)
    batch_size = config["train"].get("batch_size", 64)
    train_dataloader = TableDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate
    )
    valid_dataloader = TableDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0,
        collate_fn=collate
    )

    model = Colem(BertConfig.from_pretrained(pretrained_model_name))

    device, device_ids = prepare_device(config["train"].get("num_gpus", 4))
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("optim_lr", 5e-5),
        eps=config["train"].get("optim_eps", 1e-8)
    )
    num_epochs = config["train"].get("num_epochs", 100)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        loss_fn=nt_xent_loss,
        optimizer=optimizer,
        config=config,
        device=device,
        batch_size=batch_size,  # TODO: STRANGE ALREADY HAVE IN DATALOADER
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        lr_scheduler=get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # TODO: use warmup
            num_training_steps=len(train_dataloader) * num_epochs
        ),
        num_epochs=num_epochs,
        train_logger=Logger(config, filename=config["logs"].get("train_filename", "train.log")),
        valid_logger=Logger(config, filename=config["logs"].get("validation_filename", "valid.log"))
    )
    return trainer.train()


if __name__ == "__main__":
    results = pd.DataFrame()

    config = Config(config_path="config.yaml")

    losses = train(config)

    # plot_graphs(losses)

    print(losses)
