from pathlib import Path
from typing import Optional

import lightning as pl
import optuna
import typer
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from deep_learning_at_scale.chapter_4.dataset import SceneParsingModule
from deep_learning_at_scale.chapter_4.vision_model import VisionSegmentationModule

from ..utils import DictLogger

hpo = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

__all__ = ["hpo"]


"""
deep-learning-at-scale chapter_11 hpo train
Also, run `optuna-dashboard postgresql+psycopg2://postgres:postgres@hostname:5432/study_db` 
"""


## see https://github.com/optuna/optuna/issues/4689
class OptunaPruning(optuna.integration.PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@hpo.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    pl.seed_everything(seed, workers=True)


@hpo.command()
def train(
    name: str = typer.Option("chapter_11_hpo", help="Name of the run"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(5, help="Total number of epochs"),
    batch_size: int = typer.Option(30, help="Size of batch for the run"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    precision: str = typer.Option(
        "32-true", help="floating point format for precision"
    ),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
    study_db: Optional[str] = typer.Option(
        "study_db", help="Postgres DB name for Optuna store"
    ),
    study_db_user: Optional[str] = typer.Option(
        "postres", help="Postgres DB user for Optuna store"
    ),
    study_db_pwd: Optional[str] = typer.Option(
        "postres", help="Postgres DB password for Optuna store"
    ),
):
    exp_name = f"chapter_4/train/{name}"
    result_dir = out_dir / exp_name

    def _optuna_objectives(trial):
        ckpt_cb = ModelCheckpoint(
            monitor="val/loss", dirpath=result_dir, filename="{epoch:02d}"
        )
        logger = DictLogger()
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2)

        datamodule = SceneParsingModule(batch_size=batch_size, num_workers=num_workers)
        model = VisionSegmentationModule(
            learning_rate=learning_rate,
            adam_epsilon=1e-8,
            weight_decay=weight_decay,
        )

        trainer = PLTrainer(
            accelerator="auto",
            devices="auto",
            max_epochs=max_epochs,
            precision=precision,
            limit_train_batches=0.05,
            limit_val_batches=0.05,
            callbacks=[
                TQDMProgressBar(refresh_rate=refresh_rate),
                ckpt_cb,
                DeviceStatsMonitor(cpu_stats=True),
                EarlyStopping(monitor="val/loss", mode="min"),
                OptunaPruning(trial, monitor="val/loss"),
            ],
            logger=[logger],
        )
        trainer.fit(model, datamodule)

        return logger.metrics["val/loss"]

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        storage=f"postgresql://{study_db_user}:{study_db_pwd}@postgres:5432/{study_db}",
        study_name="scene-parsing",
    )

    study.optimize(_optuna_objectives, n_trials=2)
    optuna.visualization.plot_rank(study).show()

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
