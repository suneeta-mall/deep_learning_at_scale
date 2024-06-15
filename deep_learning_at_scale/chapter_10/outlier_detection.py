from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import typer
from pyod.models.ecod import ECOD
from torch.utils.data import DataLoader

from deep_learning_at_scale.chapter_4.dataset import SceneParsingDataset

od = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

__all__ = ["od"]


class SceneParsingSourceDataset(SceneParsingDataset):
    def __init__(self, set: str = "train", image_size: int = 128):
        super().__init__(
            set=set,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(
                        (image_size, image_size), antialias=True
                    ),
                    torchvision.transforms.Grayscale(),
                ]
            ),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image.reshape(-1)


@od.command()
def detect(
    name: str = typer.Option("chapter_104_outlier_det", help="Name of the run"),
    out_fn: Path = typer.Option(
        Path(".tmp/output/chapter_11/od/outlier.pdf"), help="Path where to store output"
    ),
    image_size: int = typer.Option(128, help="Image Size"),
    num_workers: int = typer.Option(4, help="Refresh rates"),
):
    out_fn.parent.mkdir(exist_ok=True, parents=True)

    full_res_dataset = SceneParsingDataset(set="train")
    dataset = SceneParsingSourceDataset(set="train", image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=20210,
        collate_fn=torch.utils.data.default_collate,
        num_workers=num_workers,
    )

    clf = ECOD(contamination=0.001, n_jobs=num_workers)
    for x in loader:
        clf.fit(x)

    images = [full_res_dataset[x][0] for x in np.where(clf.labels_ == 1)[0]]
    images[0].save(out_fn, save_all=True, append_images=images)

    most_outlier_sample = clf.decision_scores_.argmax(axis=0)
    full_res_dataset[most_outlier_sample][0].save(
        out_fn.parent / f"Most_oulier_image_{most_outlier_sample}.jpg"
    )

    pd.DataFrame(clf.decision_scores_).plot(kind="hist", bins=1000)
    plt.savefig(out_fn.parent / "outlier_score_histogram.jpg")
