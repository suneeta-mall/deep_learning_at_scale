from pathlib import Path

import torch
import typer
from torchmetrics.functional import mean_squared_error

from deep_learning_at_scale.chapter_9.deepfm import (
    DeepFactorizationMachineModel,
    MovieLensModule,
)

__all__ = ["app"]

"""
    python deep_learning_at_scale/chapter_9/simple_model_parallel_deepfm.py
    deep-learning-at-scale chapter_9 pt-baseline-deepfm train  
"""


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def train(
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(240, help="Size of batch for the run"),
    num_workers: int = typer.Option(5, help="Refresh rates"),
):
    result_dir = out_dir / "chapter_9/deepfm/pt/baseline"
    result_dir.mkdir(exist_ok=True, parents=True)

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
