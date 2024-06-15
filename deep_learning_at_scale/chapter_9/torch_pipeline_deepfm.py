from pathlib import Path
from typing import List

import torch
import typer
from torchmetrics.functional import mean_squared_error

from deep_learning_at_scale.chapter_9.deepfm import MovieLensModule
from deep_learning_at_scale.chapter_9.torch_model_parallel_deepfm import (
    ParallelDeepFactorizationMachineModel,
)

__all__ = ["app"]

"""
    deep-learning-at-scale chapter_9 pt-pipe-deepfm train  
"""


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class PipeDeepFactorizationMachineModel(ParallelDeepFactorizationMachineModel):
    def __init__(
        self,
        in_features: List[int],
        num_gpus: int = 2,
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
        micro_batch_chunks: int = 8,
    ):
        super().__init__(
            in_features=in_features,
            num_gpus=num_gpus,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        )
        self.micro_batch_chunks = micro_batch_chunks

    def forward(self, x: torch.Tensor):
        micro_outputs = []
        for xs in iter(x.split(self.micro_batch_chunks, dim=0)):
            micro_outputs.append(super().forward(xs))
        return torch.cat(micro_outputs)


@app.command()
def train(
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(240, help="Size of batch for the run"),
    num_workers: int = typer.Option(0, help="Refresh rates"),
):
    result_dir = out_dir / "chapter_9/deepfm/pt/pipeline_parallel"
    result_dir.mkdir(exist_ok=True, parents=True)

    datamodule = MovieLensModule(
        data_dir=result_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_genere=False,
    )
    datamodule.setup(None)

    model = PipeDeepFactorizationMachineModel(
        in_features=datamodule.train.features,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        eps=1e-8,
        weight_decay=1e-2,
    )

    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(datamodule.train_dataloader()):
            optimizer.zero_grad()
            outputs = model(x)
            y = y.to(outputs.device)
            loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
            mse = mean_squared_error(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == (99):
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    with torch.no_grad():
        val_loss = []
        val_mse = []
        for i, (x, y) in enumerate(datamodule.val_dataloader()):
            outputs = model(x)
            y = y.to(outputs.device)
            loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
            mse = mean_squared_error(outputs, y)

            val_loss.append(loss)
            val_mse.append(mse)

        print(
            f"""validation stats:
            rating mse: {torch.stack(val_mse).mean()},  
            loss: {torch.stack(val_loss).mean()}"""
        )


if __name__ == "__main__":
    app()
