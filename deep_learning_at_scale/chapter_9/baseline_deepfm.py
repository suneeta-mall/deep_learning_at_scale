from pathlib import Path
from typing import Optional

import lightning as pl
import typer
from aim.pytorch_lightning import AimLogger
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from .deepfm import DeepFactorizationMachineModel, DeepFMModule, MovieLensModule

__all__ = ["app"]

"""
    deep-learning-at-scale chapter_9 deepfm train  
"""

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


__all__ = ["app"]


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    pl.seed_everything(seed, workers=True)


@app.command()
def train(
    name: str = typer.Option("chapter_9_reco", help="Name of the run"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(24, help="Size of batch for the run"),
    num_workers: int = typer.Option(0, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
):
    exp_name = f"chapter_4/train/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )

    datamodule = MovieLensModule(
        data_dir=result_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_genere=False,
    )
    datamodule.setup(None)

    model = DeepFactorizationMachineModel(
        in_features=datamodule.train.features,
    )
    pl_model = DeepFMModule(model=model)
    trainer = PLTrainer(
        accelerator="cpu",
        devices="auto",
        precision=32,
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            exp_logger,
        ],
    )
    trainer.fit(pl_model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")
