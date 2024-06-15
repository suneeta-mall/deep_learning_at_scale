from pathlib import Path
from typing import List, Optional

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import typer
from aim import Image
from aim.pytorch_lightning import AimLogger
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy

from deep_learning_at_scale.chapter_2.data import MNISTDataModule

from ..utils import export

__all__: List[str] = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

"""
    deep-learning-at-scale chapter_11 cl train
"""


class ContrastiveTransforms:
    def __init__(self):
        self.basic_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomChoice(
                    transforms=[
                        torchvision.transforms.RandomVerticalFlip(p=1.0),
                        torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    ],
                    p=[1 / 2] * 2,
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomChoice(
                    transforms=[
                        torchvision.transforms.RandomPerspective(p=1.0),
                        torchvision.transforms.RandomRotation(degrees=(-30, +30)),
                        torchvision.transforms.RandomVerticalFlip(p=1.0),
                        torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    ],
                    p=[1 / 4] * 4,
                ),
                torchvision.transforms.RandomChoice(
                    transforms=[
                        torchvision.transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                        ),
                        torchvision.transforms.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                        ),
                        torchvision.transforms.RandomEqualize(p=1.0),
                        torchvision.transforms.RandomGrayscale(p=1.0),
                        torchvision.transforms.GaussianBlur(kernel_size=5),
                        torchvision.transforms.RandomInvert(p=1.0),
                        torchvision.transforms.RandomAutocontrast(p=1.0),
                    ],
                    p=[1 / 7] * 7,
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __call__(self, x):
        # Apply 1st basic transform on the batch of images,
        # then apply rigrous transform on same images to obtain variants
        return [self.basic_transforms(x), self.transforms(x)]


class NCELoss(nn.Module):
    def __init__(self, temperature: float = 0.08) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor):
        # Calculate the cosine similarity
        similarity_index = F.cosine_similarity(
            inputs[:, None, :],
            inputs[None, :, :],
            dim=-1,
        )
        # mask out self similarity
        self_mask = torch.eye(
            similarity_index.shape[0],
            dtype=torch.bool,
            device=similarity_index.device,
        )
        similarity_index.masked_fill_(self_mask, -9e15)
        # Calculate InfoNCE loss https://arxiv.org/pdf/1807.03748v2.pdf
        pos_mask = self_mask.roll(shifts=similarity_index.shape[0] // 2, dims=0)
        similarity_index = similarity_index / self.temperature
        nll = -similarity_index[pos_mask] + torch.logsumexp(similarity_index, dim=-1)

        positive_ranking = self._to_positive_ranking(similarity_index, pos_mask)

        return positive_ranking, nll.mean()

    def _to_positive_ranking(
        self, similarity_index: torch.Tensor, pos_mask: torch.Tensor
    ):
        positive_ranking = torch.cat(
            [
                similarity_index[pos_mask][:, None],
                similarity_index.masked_fill(pos_mask, -9e15),
            ],
            dim=-1,
        )
        positive_ranking = positive_ranking.argsort(dim=-1, descending=True).argmin(
            dim=-1
        )
        return positive_ranking


@export
class MNISTModel(LightningModule):
    def __init__(
        self,
        hidden_size=64,
        learning_rate=2e-4,
        input_shape=(1, 28, 28),  #: Tuple[int, int, int]
        num_classes: int = 10,
        temperature: float = 0.08,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.num_classes = num_classes
        channels, width, height = input_shape
        # The main encoder being trained with SimCLR
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size * 4),
        )
        # Additonal MLP as projection head. This head is typically not reused in
        # fine tunning scenarios, typically because it tends to be invariant to
        # many features like the color as contrastive training focuses on
        # representations with aggresive augmentations.
        # These features can be important for downstream tasks.
        # Note: Also note the feature size of this fc layer is 10
        # same as number of output of MNIST digit classifier.
        # This is an intentional coincidence just to measure accuracy if the output
        # from this was to be treated as classifier logits. We should expect it to be
        # poor because classification is not in the training objective.
        # This is just for a fun exercise.
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, 4 * self.num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.num_classes, self.num_classes),
        )

        self.loss = NCELoss(temperature=temperature)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

    def _common(self, batch):
        x_tuple, y = batch
        x = torch.cat(x_tuple, dim=0)
        logits = self(x)

        positive_ranking, loss = self.loss(logits)

        #
        logits = nn.functional.log_softmax(logits, dim=1)
        mean_logits = torch.stack(torch.split(logits, y.shape[0]), dim=0).mean(dim=0)
        preds = torch.argmax(mean_logits, dim=1)
        return loss, (x, y), (positive_ranking, mean_logits), preds

    def _log_image(self, x, batch_size, log_key, batch_idx) -> None:
        img_grid = torch.split(x[:, 0, ...], batch_size)
        r1 = torch.cat([img_grid[0][0], img_grid[1][0]], dim=1)
        r2 = torch.cat([img_grid[1][1], img_grid[1][1]], dim=1)
        img_grid = torch.cat([r1, r2], dim=0)

        self.logger.experiment.track(Image(img_grid), name=log_key, step=batch_idx)

    def training_step(self, batch, batch_idx):
        loss, (x, y), (positive_ranking, logits), preds = self._common(batch)

        # as noted in comment on self.fc, this is just a fun exercise
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/top1", (positive_ranking == 0).float().mean(), prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        self._log_image(x, y.shape[0], "train/images", batch_idx)

        return {"loss": loss, "logits": logits, "preds": preds}

    def validation_step(self, batch, batch_idx):
        loss, (x, y), (positive_ranking, logits), preds = self._common(batch)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/top1", (positive_ranking == 0).float().mean(), prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self._log_image(x, y.shape[0], "val/images", batch_idx)

        return {"loss": loss, "logits": logits, "preds": preds}

    def test_step(self, batch, batch_idx):
        loss, (x, y), (positive_ranking, logits), preds = self._common(batch)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/top1", (positive_ranking == 0).float().mean(), prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)
        self._log_image(x, y.shape[0], "test/images", batch_idx)

        return {"loss": loss, "logits": logits, "preds": preds}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@app.command()
def train(
    name: str = typer.Option("chapter_11_cl", help="Name of the run"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(100, help="Total number of epochs"),
    learning_rate: float = typer.Option(
        2e-4, help="The learning rate to be used in training"
    ),
    include_test: bool = typer.Option(True, help="Whether to include test or not"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
):
    exp_name = f"chapter_11/contrastive/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )
    model = MNISTModel(learning_rate=learning_rate)
    datamodule = MNISTDataModule(
        data_dir,
        num_workers=num_workers,
        transform=ContrastiveTransforms(),
    )
    trainer = Trainer(
        accelerator="cpu",
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
