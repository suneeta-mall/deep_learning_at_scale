from pathlib import Path
from typing import List, Tuple

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from tqdm.auto import trange

from ..utils import export

__all__: List[str] = []


@export
def plot_3d(
    alpha: np.ndarray,
    beta: np.ndarray,
    loss_surface: np.ndarray,
    out_loc: Path = None,
    use_log: bool = False,
):
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_axis_off()
    _modified_surface = np.log(loss_surface) if use_log else loss_surface
    ax.plot_surface(
        alpha,
        beta,
        _modified_surface,
        edgecolor="royalblue",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
        cmap=cm.coolwarm,  # linewidth=1, antialiased=False
    )
    ax.contourf(
        alpha, beta, _modified_surface, 10, lw=3, linestyles="solid", cmap=cm.coolwarm
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    if out_loc:
        plt.savefig(out_loc)  # format="jpg", try with pdf
    else:
        plt.show()
    plt.close()


def _delta_nu_for_param(param: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = torch.normal(0.0, 1.0, size=param.size())
    nu = torch.normal(0.0, 1.0, size=param.size())

    param_norm = torch.norm(param)

    delta_norm = torch.norm(delta)
    delta = (delta / delta_norm) * param_norm

    nu_norm = torch.norm(nu)
    nu = (nu / nu_norm) * param_norm
    return delta, nu


def init_nosiy_directions(model):
    noises = [_delta_nu_for_param(param) for _, param in model.named_parameters()]
    return noises


def init_network(model, alpha, beta):
    with torch.no_grad():
        for param in model.parameters():
            delta, nu = _delta_nu_for_param(param)
            param.copy_(param + alpha * delta + beta * nu)
    return model


@export
def simulate_loss_curvature(
    model: pl.LightningModule,
    dataloader: pl.LightningDataModule,
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    resolution: int = 20,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha, beta = np.meshgrid(
        np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution), indexing="ij"
    )
    loss_surface = np.empty_like(alpha)
    dataloader.setup("fit")
    for i in trange(resolution, desc="ith resolution"):
        for j in trange(resolution, desc="jth resolution"):
            total_loss = 0.0
            n_batch = 0
            network = init_network(model, alpha[i, j], beta[i, j])
            for images, labels in dataloader.train_dataloader():
                with torch.no_grad():
                    preds = network(images)
                    loss = criterion(preds, labels)
                    total_loss += loss.item()
                    n_batch += 1
            loss_surface[i, j] = total_loss / (n_batch * batch_size)
    return alpha, beta, loss_surface
