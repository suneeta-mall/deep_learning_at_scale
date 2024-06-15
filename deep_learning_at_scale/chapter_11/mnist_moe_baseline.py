from pathlib import Path
from typing import Any, List, Optional

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from aim import Image
from aim.pytorch_lightning import AimLogger
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import Accuracy

from deep_learning_at_scale.chapter_2.data import MNISTDataModule

from ..utils import export

__all__: List[str] = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

"""
    deep-learning-at-scale chapter_11 mnist-baseline train
"""


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@app.command()
def train(
    name: str = typer.Option("chapter_11_moe", help="Name of the run"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(20, help="Total number of epochs"),
    learning_rate: float = typer.Option(
        2e-4, help="The learning rate to be used in training"
    ),
    include_test: bool = typer.Option(True, help="Whether to include test or not"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
):
    exp_name = f"chapter_2/train/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )
    model = MNISTModel(learning_rate=learning_rate)
    datamodule = MNISTDataModule(data_dir, num_workers=num_workers)
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")

    if include_test:
        trainer.test(model, datamodule)


class Model(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        input_shape=(1, 28, 28),
        num_classes: int = 10,
    ):
        super(Model, self).__init__()

        channels, width, height = input_shape
        self.block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.post_moe = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.post_moe(x)
        return x


@export
class MNISTModel(LightningModule):
    def __init__(
        self,
        hidden_size=64,
        learning_rate=2e-4,
        input_shape=(1, 28, 28),  #: Tuple[int, int, int]
        num_classes: int = 10,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.num_classes = num_classes
        self.dims = input_shape
        self.model = Model()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.pred_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_accuracy, prog_bar=True)

        self.logger.experiment.track(
            Image(x[0, ...]), name="train/images", step=batch_idx
        )

        return {"loss": loss, "logits": logits, "preds": preds}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_accuracy, prog_bar=True)

        self.logger.experiment.track(
            Image(x[0, ...]), name="val/images", step=batch_idx
        )

        return {"loss": loss, "logits": logits, "preds": preds}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.test_accuracy, prog_bar=True)

        self.logger.experiment.track(
            Image(x[0, ...]), name="test/images", step=batch_idx
        )

        return {"loss": loss, "logits": logits, "preds": preds}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = self.pred_accuracy(preds, y)

        self.logger.experiment.track(
            Image(x[0, ...]), name="pred/images", step=batch_idx
        )

        return {"loss": loss, "logits": logits, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
