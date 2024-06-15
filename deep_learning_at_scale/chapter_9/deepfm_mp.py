from pathlib import Path
from typing import List, Optional

import lightning as pl
import timm
import torch
import typer
from aim.pytorch_lightning import AimLogger
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from .deepfm import (
    DeepFMModule,
    FactorizationMachine,
    FeaturesEmbedding,
    LinearFeatureEmbedding,
    MovieLensModule,
)

__all__ = ["app"]

"""
    deep-learning-at-scale chapter_9 deepfm_mp train  
"""

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class ParallelDeepFactorizationMachineModel(torch.nn.Module):
    def __init__(
        self,
        in_features: List[int],
        num_gpus: int = 2,
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        if not (torch.cuda.is_available() and torch.cuda.device_count() != num_gpus):
            raise ValueError("Must have two NVIDIA GPU.")

        self.devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

        self.embedding = FeaturesEmbedding(in_features, hidden_features).to(
            device=self.devices[0]
        )
        self.linear = LinearFeatureEmbedding(in_features).to(device=self.devices[0])

        self.fm = FactorizationMachine().to(device=self.devices[1])

        self.deep_fm_embedding_size = len(in_features) * hidden_features
        self.mlp = torch.nn.Sequential(
            timm.layers.Mlp(
                in_features=self.deep_fm_embedding_size,
                out_features=hidden_features,
                drop=(dropout_rate, 0.0),
            ),
            timm.layers.Mlp(
                in_features=hidden_features,
                out_features=hidden_features,
                drop=(dropout_rate, 0.0),
            ),
            torch.nn.Linear(in_features=hidden_features, out_features=1),
        )
        self.mlp = self.mlp.to(device=self.devices[1])
        self.sigmoid = torch.nn.Sigmoid().to(device=self.devices[1])

    def forward(self, x):
        x = x.to(self.devices[0])
        embeddings = self.embedding(x)
        linear_features = self.linear(x)

        embeddings = embeddings.to(self.devices[1])
        linear_features = linear_features.to(self.devices[1])

        linear_features += self.fm(embeddings)
        linear_features += self.mlp(embeddings.view(-1, self.deep_fm_embedding_size))
        return self.sigmoid(linear_features.squeeze(1))


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
    exp_name = f"chapter_9/deepfm/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )

    datamodule = MovieLensModule(
        data_dir=result_dir, batch_size=batch_size, num_workers=num_workers
    )
    datamodule.setup(None)

    model = ParallelDeepFactorizationMachineModel(
        in_features=[
            datamodule.train.num_users,
            datamodule.train.num_movies,
            datamodule.train.num_generes,
        ]
    )
    pl_model = DeepFMModule(model=model)
    trainer = PLTrainer(
        accelerator="gpu",
        devices=2,
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
