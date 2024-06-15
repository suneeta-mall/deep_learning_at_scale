import logging as logger
import pathlib
from pathlib import Path
from typing import List, Optional

import lightning as pl
import numpy as np
import torch
import typer
from aim.pytorch_lightning import AimLogger
from functorch.compile import compiled_function, draw_graph
from lightning import Trainer
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from PIL import Image
from torchinfo import summary
from torchviz import make_dot

from .data import MNISTDataModule
from .has_black_patches_or_not import has_black_patch
from .loss_landscape import plot_3d, simulate_loss_curvature
from .model import MNISTModel

__all__: List[str] = ["app"]

resources_dir = pathlib.Path(__file__).parent.parent.resolve() / "resources"

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(has_black_patch, name="has_black_patch")


@app.callback()
def set_seed(seed: int = typer.Option(42, help="Seed")):
    """
    ORM APP
    """
    pl.seed_everything(seed, workers=True)


@app.command()
def train(
    name: str = typer.Option("chapter_2", help="Name of the run"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(5, help="Total number of epochs"),
    learning_rate: float = typer.Option(
        2e-4, help="The learning rate to be used in training"
    ),
    include_test: bool = typer.Option(True, help="Whether to include test or not"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
    refresh_rate: int = typer.Option(20, help="Refresh rates"),
    seed: Optional[int] = typer.Option(None, callback=set_seed, help="Seed"),
):
    exp_name = f"chapter_2/train/{name}"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

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
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", dirpath=result_dir, filename="{epoch:02d}"
    )
    model = MNISTModel(learning_rate=learning_rate)
    datamodule = MNISTDataModule(data_dir, num_workers=num_workers)
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=refresh_rate),
            checkpoint_callback,
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


@app.command()
def convert_to_torch_script(
    model_checkpoint_fn: Path = typer.Argument(None, help="Location of model weights"),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
):
    model = MNISTModel.load_from_checkpoint(model_checkpoint_fn)
    datamodule = MNISTDataModule(data_dir)
    datamodule.setup("test")
    for x, y in datamodule.test_dataloader():
        break

    script = model.to_torchscript(method="script", example_inputs=(x))
    out_loc = model_checkpoint_fn.parent / f"{model_checkpoint_fn.stem}_scripted.pt"
    torch.jit.save(script, out_loc)
    logger.debug("✅ Torch conversion was successful, file saved {}", out_loc)


@app.command()
def infere(
    model_checkpoint_fn: Path = typer.Argument(
        ".tmp/output/chapter_2/train/chapter_2/epoch=04_scripted.pt",
        exists=True,
        help="Location of model weights",
    ),
    use_torch_script: bool = typer.Option(
        True, help="Whether to use torch script or not"
    ),
    # url: str = typer.Option(
    #     "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpNWA0_VH9l1JK0WUJ5Se0DORU47-2dk_PPQ&usqp=CAU",
    #     help="HTTP Url to number image for inference",
    # ),
    image_fn: Path = typer.Argument(
        resources_dir / "test.png",
        help="Location of image to test",
        exists=True,
    ),
):
    if use_torch_script:
        model = torch.jit.load(model_checkpoint_fn)
    else:
        model = MNISTModel.load_from_checkpoint(model_checkpoint_fn)
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(image_fn)
    x = torch.Tensor(np.asarray(image.resize((28, 28)).convert("L")) / 255.0)[None, ...]
    # model.eval()
    # logits = model(x)
    with torch.no_grad():
        logits = model(x)
    result = torch.argmax(logits, dim=1)
    logger.debug("Image has digit, {}", result.numpy()[0])
    return


@app.command()
def simulate_loss_curve(
    model_checkpoint_fn: Path = typer.Argument(
        Path(".tmp/output/chapter_2/train/chapter_2/epoch=04.ckpt"),
        help="Location of model weights",
    ),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_loc: Path = typer.Option(
        Path(".tmp/output/loss_curve.pdf"), help="Path where to store output"
    ),
    batch_size: int = typer.Option(32, help="Batch size"),
    resolution: int = typer.Option(20, help="resolution"),
    seed: Optional[bool] = typer.Option(None, callback=set_seed, help="Seed"),
):
    model = MNISTModel.load_from_checkpoint(model_checkpoint_fn)
    datamodule = MNISTDataModule(data_dir, batch_size=batch_size)
    alpha, beta, loss_surface = simulate_loss_curvature(
        model, datamodule, resolution=resolution
    )
    plot_3d(alpha, beta, loss_surface, out_loc)


@app.command()
def viz_model(
    model_checkpoint_fn: Path = typer.Argument(
        Path(".tmp/output/chapter_2/train/chapter_2/epoch=04.ckpt"),
        help="Location of model weights",
        exists=True,
    ),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
    out_loc: Path = typer.Option(
        Path(".tmp/output/"), help="Path where to store output"
    ),
    seed: Optional[bool] = typer.Option(None, callback=set_seed, help="Seed"),
):
    model = MNISTModel.load_from_checkpoint(model_checkpoint_fn)
    datamodule = MNISTDataModule(data_dir)
    datamodule.setup("test")
    for x, y in datamodule.test_dataloader():
        y = model(x)
        break
    fig = make_dot(
        y.mean(),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    fig.render(out_loc / "graph_viz.jpg", format="jpg")
    logger.debug(
        "🚀 Visualisation has been created and saved at {}", out_loc / "graph_viz.jpg"
    )

    def f(x):
        return x.cos().cos()

    def fw(f, x):
        draw_graph(f, out_loc / "forward-pass.svg", clear_meta=False)
        return f

    compiled_function(model, fw_compiler=fw)(x)
    logger.debug(
        """🚀 Visualisation of forward pass computation graph has 
        been created and saved at {}""",
        out_loc / "forward-pass.svg",
    )


@app.command()
def inspect_model(
    model_checkpoint_fn: Path = typer.Argument(
        Path(".tmp/output/chapter_2/train/chapter_2/epoch=04.ckpt"),
        help="Location of model weights",
        exists=True,
    ),
    data_dir: Path = typer.Option(Path(".tmp"), help="Path where to store data"),
):
    model = MNISTModel.load_from_checkpoint(model_checkpoint_fn)
    datamodule = MNISTDataModule(data_dir)
    datamodule.setup("test")
    for x, _ in datamodule.test_dataloader():
        break

    summary(
        model,
        input_size=x.shape,
        depth=5,
        verbose=2,
        col_width=16,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        row_settings=["var_names"],
    )
