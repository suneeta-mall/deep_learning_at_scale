from pathlib import Path
from typing import Optional

import typer
from aim.pytorch_lightning import AimLogger
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from .mnist import (
    DataDietLogger,
    IndexedMNISTDataModule,
    IndexedMNISTModel,
    SampleWiseStatsLogger,
)

__all__ = ["data_diet"]

"""
    deep-learning-at-scale chapter_11 data-diet train
"""

data_diet = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@data_diet.command()
def train(
    name: str = typer.Option("chapter_11}", help="Name of the run"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    learning_rate: float = typer.Option(
        2e-4, help="The learning rate to be used in training"
    ),
    include_test: bool = typer.Option(True, help="Whether to include test or not"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    seed: Optional[int] = typer.Option(42, help="Seed"),
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
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", dirpath=result_dir, filename="{epoch:02d}"
    )
    model = IndexedMNISTModel(learning_rate=learning_rate)
    datamodule = IndexedMNISTDataModule(data_dir, num_workers=num_workers)
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            checkpoint_callback,
            EarlyStopping(monitor="val/loss", mode="min"),
            SampleWiseStatsLogger(metric_save_fn=result_dir / "sample_wise_stats"),
            DataDietLogger(metric_save_fn=result_dir / "data_diet.csv"),
        ],
        logger=[
            exp_logger,
        ],
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")

    if include_test:
        trainer.test(model, datamodule)
