import logging as logger
import os
from pathlib import Path
from typing import List, Optional

import lightning as pl
import torch
import torchvision
import typer
from aim.pytorch_lightning import AimLogger

try:
    from ffcv.fields import TorchTensorField
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import ToTensor
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
except Exception:
    logger.debug(
        "FFCV is not installed, please follow the instruction when running FFCV sample"
    )
from lightning import LightningDataModule
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from deep_learning_at_scale.chapter_4.callback import ImageLogger
from deep_learning_at_scale.chapter_4.dataset import (
    PILToTensorUnScaled,
    SceneParsingDataset,
)
from deep_learning_at_scale.chapter_4.vision_model import VisionSegmentationModule

__all__ = ["efficient_ddp"]


"""
    deep-learning-at-scale chapter_7 efficient-ddp data-to-beton
    deep-learning-at-scale chapter_7 efficient-ddp train
    deep-learning-at-scale chapter_7 efficient-ddp train --precision 16-mixed \
        --batch-size 105    
    Install instructions: pip install cupy-cuda11x
    Install lib turbo from https://libjpeg-turbo.org/Documentation/OfficialBinaries
"""

efficient_ddp = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@efficient_ddp.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@efficient_ddp.command()
def data_to_beton(
    out_dir: Path = typer.Option(
        Path(".tmp/benton/"), help="Path where to store output"
    ),
    num_workers: int = typer.Option(40, help="Refresh rates"),
    input_size: int = typer.Option(256, help="Input Size"),
):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            # ),
            torchvision.transforms.Resize((input_size, input_size), antialias=True),
        ]
    )
    target_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(input_size, input_size),
                interpolation=0,  # InterpolationMode.NEAREST,
                antialias=True,
            ),
            PILToTensorUnScaled(),
        ]
    )
    for set_name in ["train", "validation"]:
        dataset = SceneParsingDataset(
            set=set_name,
            transform=transform,
            target_transform=target_transform,
        )
        writer = DatasetWriter(
            out_dir / set_name,
            {
                "image": TorchTensorField(
                    dtype=torch.float32, shape=(3, input_size, input_size)
                ),
                "label": TorchTensorField(
                    dtype=torch.int32, shape=(input_size, input_size)
                ),
            },
            num_workers=num_workers,
        )
        writer.from_indexed_dataset(dataset)


class FFCVSceneParsingModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 85,
        num_workers: int = 4,
        input_size: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_pipeline: List[Operation] = [NDArrayDecoder(), ToTensor(), Squeeze()]
        self.image_pipeline: List[Operation] = [
            NDArrayDecoder(),
            ToTensor(),
        ]

    def setup(self, stage: str):
        if stage == "fit":
            self.train = self.data_dir / "train"
            self.validation = self.data_dir / "validation"
        else:
            self.test = self.data_dir / "validation"

    def prepare_data(self): ...

    def train_dataloader(self):
        return Loader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            os_cache=1,
            drop_last=True,
            distributed=1,
            pipelines={
                "image": self.image_pipeline,
                "label": self.label_pipeline,
            },
        )

    def val_dataloader(self):
        return Loader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=True,
            distributed=1,
            os_cache=1,
            pipelines={"image": self.image_pipeline, "label": self.label_pipeline},
        )

    def test_dataloader(self):
        return Loader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=True,
            distributed=1,
            os_cache=1,
            pipelines={"image": self.image_pipeline, "label": self.label_pipeline},
        )


@efficient_ddp.command()
def train(
    name: str = typer.Option("chapter_4_ffcv_ddp", help="Name of the run"),
    data_dir: Path = typer.Option(
        Path(".tmp/benton/"), help="Path where to store output"
    ),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(85, help="Size of batch for the run"),
    num_workers: int = typer.Option(40, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    precision: str = typer.Option(
        "32-true", help="floating point format for precision"
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
    print(f"==== Main process id is {os.getpid()} =====")
    exp_name = f"chapter_7/train/{name}"
    result_dir = out_dir / exp_name
    perf_dir = result_dir / "profiler"
    perf_dir.mkdir(exist_ok=True, parents=True)

    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss", dirpath=result_dir, filename="{epoch:02d}"
    )

    datamodule = FFCVSceneParsingModule(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )
    model = VisionSegmentationModule(
        use_compile=use_compile, use_channel_last=use_channel_last
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
            ImageLogger(),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
        strategy="ddp",
        log_every_n_steps=3,
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")
