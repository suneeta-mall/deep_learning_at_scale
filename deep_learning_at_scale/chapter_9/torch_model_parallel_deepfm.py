from pathlib import Path
from typing import List

import torch
import typer
from torchmetrics.functional import mean_squared_error

from .deepfm import (
    MLP,
    FactorizationMachine,
    FeaturesEmbedding,
    LinearFeatureEmbedding,
    MovieLensModule,
)

__all__ = ["app"]

"""
    deep-learning-at-scale chapter_9 pt-mp-deepfm train  
"""


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
        self.linear = LinearFeatureEmbedding(in_features).to(device=self.devices[1])

        self.fm = FactorizationMachine().to(device=self.devices[1])

        self.deep_fm_embedding_size = len(in_features) * hidden_features
        self.mlp = MLP(
            embedding_size=self.deep_fm_embedding_size,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        ).to(device=self.devices[1])
        self.sigmoid = torch.nn.Sigmoid().to(device=self.devices[1])

    def forward(self, x):
        embeddings = self.embedding(x.to(self.devices[0]))
        linear_features = self.linear(x.to(self.devices[1]))

        embeddings = embeddings.to(self.devices[1])

        linear_features += self.fm(embeddings)
        linear_features += self.mlp(embeddings.view(-1, self.deep_fm_embedding_size))
        return self.sigmoid(linear_features.squeeze(1))


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def train(
    out_dir: Path = typer.Option(
        Path(".tmp/output"), help="Path where to store output"
    ),
    max_epochs: int = typer.Option(10, help="Total number of epochs"),
    batch_size: int = typer.Option(240, help="Size of batch for the run"),
    num_workers: int = typer.Option(0, help="Refresh rates"),
):
    result_dir = out_dir / "chapter_9/deepfm/pt/model_parallel"
    result_dir.mkdir(exist_ok=True, parents=True)

    datamodule = MovieLensModule(
        data_dir=result_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_genere=False,
    )
    datamodule.setup(None)

    model = ParallelDeepFactorizationMachineModel(
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
