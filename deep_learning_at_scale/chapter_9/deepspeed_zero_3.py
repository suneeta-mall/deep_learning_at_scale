from pathlib import Path
from typing import List, Optional

import lightning as pl
import typer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from deep_learning_at_scale.chapter_4.dataset import SceneParsingModule
from deep_learning_at_scale.chapter_4.vision_model import VisionSegmentationModule

__all__ = ["app"]


"""
    deep-learning-at-scale chapter_9 zero3 train
"""

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class DeepSpeedUNetSegmentationModule(VisionSegmentationModule):
    """
    Same Module as used in chapter_4/vision_model.py
    with an exception of deep_speed optimiser to allow off-loading
    """

    def __init__(
        self,
        num_classes: int = 151,
        learning_rate: float = 0.001,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        use_compile: bool = False,
        use_channel_last: bool = False,
    ):
        super().__init__(
            num_classes,
            learning_rate,
            adam_epsilon,
            weight_decay,
            use_compile,
            use_channel_last,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer]


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@app.command()
def train(
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(160, help="Size of batch for the run"),
    num_workers: int = typer.Option(60, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    precision: str = typer.Option(
        "32-true", help="floating point format for precision, e.g. 32-true, 16-mixed"
    ),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
    use_compile: bool = typer.Option(False, help="Whether to compile the graph or not"),
    use_channel_last: bool = typer.Option(
        False, help="Whether to use channel last memory format or not"
    ),
    devices: List[int] = typer.Option(
        None, help="List of GPUs, use all if not suuplied."
    ),
    nodes: int = typer.Option(1, help="Number of nodes to be used in training"),
):
    result_dir = out_dir / "chapter_9/zero/"
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss", dirpath=result_dir, filename="{epoch:02d}"
    )

    datamodule = SceneParsingModule(batch_size=batch_size, num_workers=num_workers)
    model = DeepSpeedUNetSegmentationModule(
        use_compile=use_compile,
        use_channel_last=use_channel_last,
    )

    trainer = PLTrainer(
        accelerator="auto",
        devices=devices if devices else "auto",
        num_nodes=nodes,
        max_epochs=max_epochs,
        precision=precision,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            ckpt_cb,
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
        strategy="deepspeed_stage_3_offload",
        log_every_n_steps=3,
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")
