import os
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, Optional, Sequence, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException


# Batch, batch_index, predicted_labels (corresponds to the elements that are returned in test.step())
TestPredictionElement: Tuple[List[torch.Tensor], int, torch.IntTensor]


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module: pl.LightningModule,
        prediction: TestPredictionElement,
        batch_indices: Optional[List[Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        dataloader_path = os.path.join(self.output_dir, str(dataloader_idx))

        if not os.path.exists(dataloader_path):
            os.mkdir(dataloader_path)

        # x, y = batch
        # print(x.shape)
        # print(y.shape)
        # print(prediction[0][0].shape)

        # TODO: Potentially create the dataloader folder
        torch.save(
            prediction,
            os.path.join(self.output_dir, str(dataloader_idx), f"{str(batch_idx)}.pt"),
        )

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: pl.LightningModule,
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.interval.on_batch:
            return

        batch_indices = None

        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence[Any],
    ) -> None:
        if not self.interval.on_epoch:
            return
        epoch_batch_indices = trainer.test_loop.epoch_batch_indices
        self.write_on_epoch_end(
            trainer, pl_module, trainer.test_loop.predictions, epoch_batch_indices
        )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )

    def on_predict_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence[Any],
    ) -> None:
        if not self.interval.on_epoch:
            return
        epoch_batch_indices = trainer.predict_loop.epoch_batch_indices
        self.write_on_epoch_end(
            trainer, pl_module, trainer.predict_loop.predictions, epoch_batch_indices
        )
