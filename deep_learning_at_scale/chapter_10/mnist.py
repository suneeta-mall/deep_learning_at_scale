from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from deep_learning_at_scale.chapter_2.data import MNISTDataModule
from deep_learning_at_scale.chapter_2.model import MNISTModel

from ..utils import export

__all__: List[str] = []


class IndexedMNIST(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target


@export
class IndexedMNISTDataModule(MNISTDataModule):
    def __init__(
        self, data_dir: Path = Path("."), batch_size: int = 32, num_workers: int = 4
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
        )

    def prepare_data(self):
        IndexedMNIST(self.data_dir, train=True, download=True)
        IndexedMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = IndexedMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = IndexedMNIST(
                self.data_dir, train=False, transform=self.transform
            )


@export
class IndexedMNISTModel(MNISTModel):
    def __init__(
        self,
        hidden_size=64,
        learning_rate=2e-4,
        input_shape=(1, 28, 28),  #: Tuple[int, int, int]
        num_classes: int = 10,
    ):
        super().__init__(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            input_shape=input_shape,
            num_classes=num_classes,
        )

    def training_step(self, batch, batch_idx):
        idx, x, y = batch
        return super().training_step((x, y), batch_idx)

    def validation_step(self, batch, batch_idx):
        idx, x, y = batch
        return super().validation_step((x, y), batch_idx)

    def test_step(self, batch, batch_idx):
        idx, x, y = batch
        return super().test_step((x, y), batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        idx, x, y = batch
        return super().predict_step((x, y), batch_idx)


class SampleWiseStatsLogger(Callback):
    def __init__(self, metric_save_fn: Path = Path("samplewise_stats.csv")):
        self.metric_save_fn = metric_save_fn
        self.metric_save_fn.parent.mkdir(exist_ok=True, parents=True)
        self.df = pd.DataFrame(
            {
                "sample_index": [],
                "epoch": [],
                "loss": [],
                "preds": [],
            }
        )

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: Dict,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # {"loss": loss, "logits": logits, "preds": preds}
        idx, x, y = batch
        logits = outputs["logits"]
        with torch.no_grad():
            loss = (
                F.nll_loss(
                    logits,
                    y,
                    reduction="none",
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        idx = idx.detach().cpu().numpy().tolist()
        step_df = pd.DataFrame(
            {
                "sample_index": idx,
                "epoch": [trainer.current_epoch] * len(idx),
                "loss": loss,
                "preds": preds,
            }
        )
        self.df = pd.concat([self.df, step_df])

    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.df.to_csv(self.metric_save_fn)

    def on_train_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self.df.to_csv(self.metric_save_fn)


class DataDietLogger(Callback):
    def __init__(self, metric_save_fn: Path = Path("data_diet.csv")):
        self.metric_save_fn = metric_save_fn
        self.metric_save_fn.parent.mkdir(exist_ok=True, parents=True)
        self.df = pd.DataFrame(
            {
                "sample_index": [],
                "epoch": [],
                "grad_score": [],
            }
        )

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: Dict,
        batch: Any,
        batch_idx: int,
    ) -> None:
        idx, x, y = batch

        def model_fn(params, x):
            return torch.func.functional_call(pl_module.model, params, x)

        def sample_wise_nll_loss(x, y):
            return torch.nn.functional.nll_loss(x, y, reduction="none")

        def sample_wise_loss_fn(params, x, y):
            return torch.vmap(sample_wise_nll_loss)(model_fn(params, x), y)

        pl_module.eval()

        params = dict(pl_module.model.named_parameters())

        # https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html
        rev_jaccobian_dict = torch.func.jacrev(sample_wise_loss_fn)(params, x, y)
        flat_jaccobian = torch.utils._pytree.tree_flatten(
            torch.utils._pytree.tree_map(torch.vmap(torch.ravel), rev_jaccobian_dict)
        )[0]
        loss_grads = torch.concatenate(flat_jaccobian, axis=1)
        grad_score_per_sample = (
            torch.linalg.norm(loss_grads, axis=-1).detach().cpu().numpy().tolist()
        )

        idx = idx.detach().cpu().numpy().tolist()

        pl_module.train()

        step_df = pd.DataFrame(
            {
                "sample_index": idx,
                "epoch": [trainer.current_epoch] * len(idx),
                "grad_score": grad_score_per_sample,
            }
        )
        self.df = pd.concat([self.df, step_df])

    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.df.to_csv(self.metric_save_fn)

    def on_train_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self.df.to_csv(self.metric_save_fn)
