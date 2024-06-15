from pathlib import Path
from typing import List

import lightning as pl
import timm
import torch
import torchvision
import typer
from aim.pytorch_lightning import AimLogger
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import f1_score
from torchvision import transforms

from ..utils import export

__all__: List[str] = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    pl.seed_everything(seed, workers=True)


@app.command()
def train_student_baseline(
    name: str = typer.Option("chapter_11_student_baseline", help="Name of the run"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(15, help="Total number of epochs"),
    batch_size: int = typer.Option(10000, help="Batch size per trainer"),
    learning_rate: float = typer.Option(
        2e-4, help="The learning rate to be used in training"
    ),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
):
    exp_name = f"chapter_11/train/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )
    student_model = TimmLightningModule(
        name="efficientnet_b0", learning_rate=learning_rate
    )

    datamodule = MNISTDataModule(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            DeviceStatsMonitor(cpu_stats=True),
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
    )
    trainer.fit(student_model, datamodule)
    trainer.test(student_model, datamodule)


@app.command()
def train(
    name: str = typer.Option("chapter_11_distillation", help="Name of the run"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(15, help="Total number of epochs"),
    batch_size: int = typer.Option(10000, help="Batch size per trainer"),
    learning_rate: float = typer.Option(
        2e-4, help="The learning rate to be used in training"
    ),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
):
    exp_name = f"chapter_11/train/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    perf_dir = result_dir / "profiler"
    perf_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )
    teacher_model = TimmLightningModule(
        name="efficientnet_b4", learning_rate=learning_rate
    )
    datamodule = MNISTDataModule(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            DeviceStatsMonitor(cpu_stats=True),
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
    )
    trainer.fit(teacher_model, datamodule)
    trainer.test(teacher_model, datamodule)
    teacher_model.eval()

    distller = DistillationLightningModule(
        name="efficientnet_b0", teacher_model=teacher_model
    )
    dist_trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            DeviceStatsMonitor(cpu_stats=True),
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
    )
    dist_trainer.fit(distller, datamodule)
    dist_trainer.test(distller, datamodule)


@export
class MNISTDataModule(LightningDataModule):
    def __init__(
        self, data_dir: Path = Path("."), batch_size: int = 100, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        torchvision.datasets.FashionMNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = torchvision.datasets.FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = torchvision.datasets.FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def teardown(self, stage: str): ...


class TimmModel(torch.nn.Module):
    def __init__(
        self,
        name: str = "efficientnet_b0",
        in_channels: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.model = timm.create_model(
            name, pretrained=True, in_chans=in_channels, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class TimmLightningModule(LightningModule):
    def __init__(
        self,
        name: str,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = TimmModel(name=name, num_classes=num_classes)

    def forward(self, inputs):
        return self.model(inputs)

    def _common_step(self, batch, batch_idx, key: str):
        images, labels = batch
        outputs = self(images)
        predictions = outputs.argmax(1)

        loss = torch.nn.functional.cross_entropy(outputs, labels.long())
        iou = f1_score(
            predictions, labels, task="multiclass", num_classes=self.hparams.num_classes
        )

        self.log(f"{key}/loss", loss, prog_bar=True, sync_dist=key != "train")
        self.log(f"{key}/f1", iou, prog_bar=True, sync_dist=key != "train")

        return {"loss": loss, "preds": predictions, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer]


class DistillationLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.5,
        num_classes: int = 10,
    ):
        super().__init__()
        self.temperature = temperature
        self.register_buffer(
            "weights", torch.Tensor([self.temperature**2] * num_classes)
        )

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        teacher_probs = torch.nn.functional.softmax(
            teacher_logits / self.temperature, dim=0
        )
        kd_loss = torch.nn.functional.cross_entropy(
            student_logits / self.temperature, teacher_probs, self.weights
        )

        return kd_loss


class DistillationLightningModule(LightningModule):
    def __init__(
        self,
        name: str,
        teacher_model: torch.nn.Module,
        num_classes: int = 10,
        temprature: float = 0.5,
        learning_rate: float = 1e-3,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.model = TimmModel(name=name, num_classes=num_classes)
        self.distil_loss_fn = DistillationLoss(
            temperature=temprature, num_classes=num_classes
        )
        self.student_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        with torch.no_grad():
            teachers_logits = self.teacher_model(inputs)
        students_logits = self.model(inputs)
        return {"teachers_logits": teachers_logits, "students_logits": students_logits}

    def _common_step(self, batch, batch_idx, key: str):
        images, labels = batch
        result_dict = self(images)
        teachers_logits, students_logits = (
            result_dict["teachers_logits"],
            result_dict["students_logits"],
        )
        predictions = students_logits.argmax(1)

        distil_loss = self.distil_loss_fn(teachers_logits, students_logits)
        student_loss = self.student_loss_fn(students_logits, labels.long())
        loss = distil_loss + student_loss
        iou = f1_score(
            predictions, labels, task="multiclass", num_classes=self.num_classes
        )

        self.log(f"{key}/loss", loss, prog_bar=True, sync_dist=key != "train")
        self.log(f"{key}/f1", iou, prog_bar=True, sync_dist=key != "train")

        return {"loss": loss, "preds": predictions, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        return [optimizer]
        return [optimizer]
