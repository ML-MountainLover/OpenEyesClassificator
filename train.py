import os

import gdown
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from model.dataset import EyesDataModule
from model.model import LightningEyesClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_classificator(
    data_csv_path,
    label_column,
    exp_name,
    chekpoint_path=None,
    num_epoch=2,
    device=DEVICE,
):

    data_module = EyesDataModule(
        data_csv_path=data_csv_path,
        label_column="labels",
    )

    if chekpoint_path is None:
        model = LightningEyesClassifier()
    else:
        model = LightningEyesClassifier.load_from_checkpoint(
            checkpoint_path=chekpoint_path,
            map_location=device,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{exp_name}",
        filename="{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=1,
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=num_epoch,
        accelerator=device,
        devices=1,
        enable_progress_bar=False,
        gradient_clip_val=1,
        logger=False,
    )

    trained_model = trainer.fit(model, data_module)

    return trained_model


if __name__ == "__main__":
    DRIVE_ID_KAGGLE_MODEL = "1sG9QWm0fZN_uaQFVUnktEjxW5l1vaYIH"

    if not os.path.exists("./checkpoints/kaggle_model.ckpt"):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        gdown.download(
            id=DRIVE_ID_KAGGLE_MODEL, output="./checkpoints/kaggle_model.ckpt"
        )

    train_classificator(
        data_csv_path="./test.csv",
        label_column="label",
        exp_name="test_exp",
        chekpoint_path="./checkpoints/kaggle_model.ckpt",
    )
