from pathlib import Path
from typing import List

import lightning as pl
import numpy as np
import pandas as pd
import timm
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import mean_squared_error
from torchvision.datasets.utils import download_and_extract_archive

validation_split_percentage: int = 5


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, in_features: List[int], out_features: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(in_features), out_features)
        self.offsets = np.array((0, *np.cumsum(in_features)[:-1]), dtype=np.int64)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class LinearFeatureEmbedding(torch.nn.Module):
    def __init__(self, in_features: List[int], out_features: int = 1):
        super().__init__()
        self.fc = torch.nn.Embedding(
            num_embeddings=sum(in_features), embedding_dim=out_features
        )
        self.bias = torch.nn.Parameter(torch.zeros((out_features,)))
        self.offsets = np.array((0, *np.cumsum(in_features)[:-1]), dtype=np.int64)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lamba = 0.5

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return self.lamba * ix


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        embedding_size: int,
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__(
            timm.layers.Mlp(
                in_features=embedding_size,
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


class DeepFactorizationMachineModel(torch.nn.Module):
    def __init__(
        self,
        in_features: List[int],
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.embedding = FeaturesEmbedding(in_features, hidden_features)

        self.linear = LinearFeatureEmbedding(in_features)
        self.fm = FactorizationMachine()

        self.deep_fm_embedding_size = len(in_features) * hidden_features
        self.mlp = MLP(
            embedding_size=self.deep_fm_embedding_size,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        embeddings = self.embedding(x)
        linear_features = self.linear(x)
        linear_features += self.fm(embeddings)
        linear_features += self.mlp(embeddings.view(-1, self.deep_fm_embedding_size))
        return self.sigmoid(linear_features.squeeze(1))


class LinearFMHead(torch.nn.Module):
    def __init__(
        self,
        in_features: List[int],
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.linear = LinearFeatureEmbedding(in_features=in_features, out_features=1)
        self.fm = FactorizationMachine()
        self.deep_fm_embedding_size = len(in_features) * hidden_features
        self.mlp = MLP(
            embedding_size=self.deep_fm_embedding_size,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor):
        linear_features = self.linear(x)
        linear_features += self.fm(embeddings)
        linear_features += self.mlp(embeddings.view(-1, self.deep_fm_embedding_size))
        return self.sigmoid(linear_features.squeeze(1))


class DeepFactorizationMachineModelV2(torch.nn.Module):
    def __init__(
        self,
        in_features: List[int],
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.embedding = FeaturesEmbedding(in_features, hidden_features)

        self.linear = LinearFMHead(
            in_features=in_features,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        linear_features = self.linear(x, embeddings)
        return linear_features


class MovieLensDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool = True, use_genere: bool = False):
        # download moview lens data from source
        movielens_data_url = (
            "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        )

        download_and_extract_archive(movielens_data_url, data_dir)
        rating_df = pd.read_csv(data_dir / "ml-latest-small" / "ratings.csv")
        movies_df = pd.read_csv(data_dir / "ml-latest-small" / "movies.csv")

        self.user2id = {
            x: i for i, x in enumerate(rating_df["userId"].unique().tolist())
        }
        self.movie2id = {
            x: i for i, x in enumerate(rating_df["movieId"].unique().tolist())
        }
        self.id2movie = {i: x for x, i in self.movie2id.items()}

        self.generes = []
        if use_genere:
            for _a in movies_df["genres"].tolist():
                self.generes.extend(_a.split("|"))
            self.generes = list(set(self.generes))
            self.generes.sort()

            def generate_genere(row):
                movie_id = row["movieId"]
                _genere_str = movies_df["genres"][
                    movies_df["movieId"] == movie_id
                ].values[0]
                for g in _genere_str.split("|"):
                    row[g] = 1
                return row

            rating_df[self.generes] = [0] * self.num_generes
            rating_df = rating_df.apply(generate_genere, axis=1)

        rating_df["user"] = rating_df["userId"].map(self.user2id)
        rating_df["movie"] = rating_df["movieId"].map(self.movie2id)
        rating_df["rating"] = rating_df["rating"].values.astype(np.float32)
        min_rating, max_rating = min(rating_df["rating"]), max(rating_df["rating"])

        rating_df = rating_df.sample(frac=1, random_state=42)

        x = rating_df[["user", "movie", *self.generes]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = (
            rating_df["rating"]
            .apply(lambda x: (x - min_rating) / (max_rating - min_rating))
            .values
        )
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * rating_df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )
        self.x = x_train if train else x_val
        self.y = y_train if train else y_val

    @property
    def num_users(self) -> int:
        return len(self.user2id)

    @property
    def num_movies(self) -> int:
        return len(self.id2movie)

    @property
    def num_generes(self) -> int:
        return len(self.generes)

    @property
    def features(self) -> List[int]:
        return (
            [
                self.num_users,
                self.num_movies,
                self.num_generes,
            ]
            if self.generes
            else [
                self.num_users,
                self.num_movies,
            ]
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MovieLensModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 8,
        num_workers: int = 0,
        use_genere: bool = False,
    ):
        super().__init__()
        super().save_hyperparameters()

    def setup(self, stage: str):
        self.train = MovieLensDataset(
            self.hparams.data_dir, train=True, use_genere=self.hparams.use_genere
        )
        self.val = MovieLensDataset(
            self.hparams.data_dir, train=False, use_genere=self.hparams.use_genere
        )

    def prepare_data(self): ...

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


class DeepFMModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-2,
        use_compile: bool = False,
        use_channel_last: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def _common_step(self, batch, batch_idx, key: str):
        x, y = batch

        outputs = self(x)
        loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
        mse = mean_squared_error(outputs, y)

        self.log(f"{key}/loss", loss, prog_bar=True, sync_dist=key != "train")
        self.log(f"{key}/mse", mse, prog_bar=True, sync_dist=key != "train")

        return {"loss": loss, "preds": outputs, "labels": y, "mse": mse}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer]


def to_parallel_data_loader(dataset, batch_size, rank, world_size):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size - 1, rank=rank
    )
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
    )
