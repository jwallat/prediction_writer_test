import os
import pandas as pd
import seaborn as sn
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from typing import Sequence, Any

from model import LitMNIST
from prediction_writer import CustomWriter


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def main():
    print("In main")

    prediction_writer = CustomWriter(
        output_dir="/home/jonas/git/prediction_writer_test/preds",
        write_interval="batch",
    )

    model = LitMNIST()
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=1,
        callbacks=[TQDMProgressBar(refresh_rate=20), prediction_writer],
        logger=CSVLogger(save_dir="logs/"),
    )
    # trainer.fit(model)

    trainer.test(model)


if __name__ == "__main__":
    main()
