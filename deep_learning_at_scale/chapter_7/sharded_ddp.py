import os
from pathlib import Path
from typing import List, Optional

import lightning as pl
import torch
import typer
from aim.pytorch_lightning import AimLogger
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from deep_learning_at_scale.chapter_4.dataset import SceneParsingModule
from deep_learning_at_scale.chapter_4.vision_model import VisionSegmentationModule

__all__ = ["ddp"]


"""
    deep-learning-at-scale chapter_7 sharded-ddp train
"""

ddp = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class DeepSpeedUNetSegmentationModule(VisionSegmentationModule):
    """
    Same Module as used in chapter_4/vision_model.py
    with an exception of deep_speed optimiser to allow off-loading
    """

    def __init__(
        self,
        use_deep_speed_optim: bool = False,
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
        self.use_deep_speed_optim = use_deep_speed_optim
        self.save_hyperparameters()

    def configure_optimizers(self):
        # return bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995))
        # Defaylt DeepSpeedCPUAdam uses AdamW as per adamw_mode
        optimizer_fn = (
            DeepSpeedCPUAdam if self.hparams.use_deep_speed_optim else torch.optim.AdamW
        )
        optimizer = optimizer_fn(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer]


@ddp.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@ddp.command()
def train(
    name: str = typer.Option("chapter_7_sharded_ddp", help="Name of the run"),
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
    # fsdp, fsdp_cpu_offload, deepspeed_stage_2_offload
    strategy: str = typer.Option("ddp", help="strategy"),
):
    print(f"==== Main process id is {os.getpid()} =====")
    exp_name = f"chapter_7/train/{name}"
    result_dir = out_dir / exp_name
    perf_dir = result_dir / "profiler"
    perf_dir.mkdir(exist_ok=True, parents=True)

    torch_profiler = PyTorchProfiler(
        dirpath=perf_dir,
        filename="perf_logs_pytorch",
        group_by_input_shapes=True,
        emit_nvtx=torch.cuda.is_available(),
        activities=(
            [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
            if torch.cuda.is_available()
            else [
                torch.profiler.ProfilerActivity.CPU,
            ]
        ),
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=5, repeat=10, skip_first=True
        ),
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(perf_dir / "trace")
        ),
    )
    exp_logger = AimLogger(
        experiment=exp_name,
        train_metric_prefix="train/",
        val_metric_prefix="val/",
        test_metric_prefix="test/",
    )

    ckpt_cb = ModelCheckpoint(
        monitor="val/loss", dirpath=result_dir, filename="{epoch:02d}"
    )

    datamodule = SceneParsingModule(batch_size=batch_size, num_workers=num_workers)
    model = DeepSpeedUNetSegmentationModule(
        use_compile=use_compile,
        use_channel_last=use_channel_last,
        use_deep_speed_optim=(strategy == "deepspeed_stage_2_offload"),
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
            DeviceStatsMonitor(cpu_stats=True),
            EarlyStopping(monitor="val/loss", mode="min"),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
        profiler=torch_profiler,
        strategy=strategy,
        log_every_n_steps=3,
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")
