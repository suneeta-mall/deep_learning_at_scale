from pathlib import Path
from typing import List

import timm
import torch
import typer
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error

from deep_learning_at_scale.chapter_9.deepfm import (
    FactorizationMachine,
    FeaturesEmbedding,
    LinearFeatureEmbedding,
    MovieLensDataset,
)

__all__ = ["app"]

"""
    python deep_learning_at_scale/chapter_9/simple_model_parallel_deepfm.py
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


@app.command()
def train(
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(240, help="Size of batch for the run"),
    num_workers: int = typer.Option(0, help="Refresh rates"),
):
    result_dir = out_dir / "chapter_9/deepfm/"
    result_dir.mkdir(exist_ok=True, parents=True)

    train = MovieLensDataset(result_dir, train=True)
    val = MovieLensDataset(result_dir, train=False)

    train_loader, val_loader = [
        DataLoader(
            _set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        for _set in [train, val]
    ]

    model = ParallelDeepFactorizationMachineModel(in_features=train.features)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        eps=1e-8,
        weight_decay=1e-2,
    )

    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            y = y.to(outputs.device)
            loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
            mse = mean_squared_error(outputs, y)
            loss.backward()

            running_loss += loss.item()
            if i % 100 == (99):
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    with torch.no_grad():
        val_loss = []
        val_mse = []
        for i, (x, y) in enumerate(val_loader):
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
