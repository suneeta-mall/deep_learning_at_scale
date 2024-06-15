from typing import Any, List

import torch
from aim import Image
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from ..utils import export

__all__: List[str] = []


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
        channels, width, height = self.dims
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
            nn.LogSoftmax(dim=1),
        )

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.pred_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

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
        loss = F.nll_loss(logits, y)
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
        loss = F.nll_loss(logits, y)
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
