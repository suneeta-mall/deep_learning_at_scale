from pathlib import Path
from typing import Optional

import lightning as pl
import torch
import typer
from aim.pytorch_lightning import AimLogger
from lightning import Trainer as PLTrainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from .callback import ImageLogger
from .dataset import SceneParsingModule, WikiDataModule
from .model import GPT2Module
from .vision_model import VisionSegmentationModule

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

__all__ = ["app"]


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@app.command()
def train_gpt2(
    name: str = typer.Option("chapter_4_gpt2", help="Name of the run"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(2, help="Total number of epochs"),
    batch_size: int = typer.Option(24, help="Size of batch for the run"),
    include_test: bool = typer.Option(True, help="Whether to include test or not"),
    num_workers: int = typer.Option(0, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    precision: str = typer.Option(
        "32-true", help="floating point format for precision"
    ),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
    graph_compile_mode: bool = typer.Option(
        None,
        help="""Indicate the mode of graph compilation default, max-autotune, 
        reduce-overhead or None""",
    ),
):
    model_name: str = "gpt2"
    exp_name = f"chapter_4/train/{name}"
    result_dir = out_dir / exp_name
    perf_dir = result_dir / "profiler"
    perf_dir.mkdir(exist_ok=True, parents=True)

    torch_profiler = PyTorchProfiler(
        dirpath=perf_dir,
        filename="perf_logs_pytorch",
        group_by_input_shapes=True,
        # emit_nvtx=torch.cuda.is_available(),  ## See why here https://github.com/pytorch/pytorch/issues/98124
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

    datamodule = WikiDataModule(
        name=model_name, batch_size=batch_size, num_workers=num_workers
    )
    model = GPT2Module(name=model_name, graph_compile_mode=graph_compile_mode)

    trainer = PLTrainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
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
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")

    if include_test:
        trainer.test(model, datamodule)

    ######
    ## https://huggingface.co/blog/how-to-generate
    statement = (
        "I've been waiting for a deep learning at scale book my whole life. "
        "Now that I have one, I shall read it. And I "
    )
    input_ids = trainer.model.tokenizer.encode(statement, return_tensors="pt")

    greedy_output = trainer.model.model.generate(input_ids, max_length=50)
    print(trainer.model.tokenizer.decode(greedy_output[0], skip_special_tokens=True))

    beam_output = trainer.model.model.generate(
        input_ids, max_length=50, num_beams=5, early_stopping=True
    )
    print(trainer.model.tokenizer.decode(beam_output[0], skip_special_tokens=True))

    # no repeat
    beam_output = trainer.model.model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    print(trainer.model.tokenizer.decode(beam_output[0], skip_special_tokens=True))

    beam_output = trainer.model.model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
    )
    for i, beam_output in enumerate(beam_output):
        print(
            "{}: {}".format(
                i, trainer.model.tokenizer.decode(beam_output, skip_special_tokens=True)
            )
        )

    beam_output = trainer.model.model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        # no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
        do_sample=True,
        top_k=0,
        temperature=0.7,
    )
    for i, beam_output in enumerate(beam_output):
        print(
            "{}: {}".format(
                i, trainer.model.tokenizer.decode(beam_output, skip_special_tokens=True)
            )
        )

    beam_output = trainer.model.model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        # no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
        do_sample=True,
        top_k=0,
        top_p=0.92,
        temperature=0.7,
    )
    for i, beam_output in enumerate(beam_output):
        print(
            "{}: {}".format(
                i, trainer.model.tokenizer.decode(beam_output, skip_special_tokens=True)
            )
        )


@app.command()
def train_vision_model(
    name: str = typer.Option("chapter_4_cv", help="Name of the run"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(100, help="Total number of epochs"),
    batch_size: int = typer.Option(85, help="Size of batch for the run"),
    include_test: bool = typer.Option(True, help="Whether to include test or not"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    precision: str = typer.Option(
        "32-true", help="floating point format for precision"
    ),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
    use_compile: bool = typer.Option(False, help="Whether to compile the graph or not"),
    use_channel_last: bool = typer.Option(
        False, help="Whether to use channel last memory format or not"
    ),
):
    exp_name = f"chapter_4/train/{name}"
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
    model = VisionSegmentationModule(
        use_compile=use_compile, use_channel_last=use_channel_last
    )

    trainer = PLTrainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        precision=precision,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            ckpt_cb,
            DeviceStatsMonitor(cpu_stats=True),
            EarlyStopping(monitor="val/loss", mode="min"),
            ImageLogger(),
        ],
        logger=[
            exp_logger,
            TensorBoardLogger(save_dir=result_dir / "logs"),
        ],
        profiler=torch_profiler,
    )
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(result_dir / "model.bin")

    if include_test:
        trainer.test(model, datamodule)
