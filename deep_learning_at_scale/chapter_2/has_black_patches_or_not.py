import math
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import umap.umap_ as umap

__all__: List[str] = ["has_black_patch"]

"""
deep-learning-at-scale chapter_2 has_back_patch feature-embedding

"""
has_black_patch = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

np.random.seed(42)


def bce(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0 + term_1, axis=0)


def bce_prime(y_pred, y):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -y / (y_pred + 1e-7) + (1 - y) / (1 - y_pred + 1e-7)


def random_x_y(size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    x = np.full((size * size), 0.996, dtype=np.float32)
    x = x.reshape((size, size))
    y = np.zeros((1), dtype=np.float32)
    if np.random.uniform() < 0.5:
        idx = np.random.randint(size // 4, size - size // 4)
        x[idx - 1 : idx + 2, idx - 1 : idx + 2] = 0
        y[0] = 1
    x = x.reshape((1, size * size))
    x += np.abs(np.random.normal(-1e-4, 1e-4, (size * size)))
    yield x, y


def generate_data(
    sample_size: int, input_size: int
) -> Iterable[Tuple[np.ndarray, int]]:
    for _ in range(sample_size):
        yield from random_x_y(input_size)


def prediction_outcome(y_pred, y) -> str:
    decision = (y_pred >= 0.5) * 1
    if y == decision:
        return "TP" if y[0] == 1.0 else "TN"

    return "FN" if y[0] == 1.0 else "FP"


def metrics_from_outcome(outcome: List[str]):
    outcome = np.array(outcome)

    def occurances(key):
        return len(np.where(outcome == key)[0])

    tp, fp, tn, fn = (
        occurances("TP"),
        occurances("FP"),
        occurances("TN"),
        occurances("FN"),
    )
    # Calculate metrics and add an small number for numerical stability.
    accuracy = (tp + tn) / len(outcome)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-7)
    return (
        np.round_(accuracy),
        np.round_(precision),
        np.round_(recall),
        np.round_(f1_score),
    )


class LinearLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Random initialisation of layer parameters
        self.weights = np.random.uniform(
            low=-0.01, high=0.01, size=(input_size, output_size)
        )
        self.bias = np.random.uniform(low=-0.01, high=0.01, size=(1, output_size))

    def forward(self, x) -> np.ndarray:
        self.x = x
        # y = weights * x + bias
        return np.matmul(x, self.weights) + self.bias

    def backward(self, error, learning_rate) -> np.ndarray:
        i_error = np.matmul(error, self.weights.T)

        # Estimate contribution of error by parameteres
        weights_error = np.matmul(self.x.T, error).reshape(self.weights.shape)
        bias_error = error
        # Adjust the weights and bias to account for error in controlled manner
        # (guided by learning rate)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return i_error


class ReluActivationLayer:
    def __init__(self): ...

    def forward(self, x) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, error, learning_rate) -> np.ndarray:
        return error * np.array(self.x >= 0).astype("int")


class SigmoidLayer:
    def __init__(self, input_size):
        self.input_size = input_size

    def forward(self, x) -> np.ndarray:
        self.x = x
        self.y = 1 / (1 + math.exp(-x))
        return self.y

    def backward(self, error, learning_rate) -> np.ndarray:
        sigmoid_prime = self.y * (1 - self.y)  # differential of sigmoid equation
        return error * sigmoid_prime


def _plot_embedding(x, y, n_neighbors: int = 5, label: str = "Patch Dataset"):
    embedding = umap.UMAP(n_neighbors=n_neighbors).fit_transform(x)
    _, ax = plt.subplots(1, figsize=(9, 7))
    plt.scatter(*embedding.T, s=0.5, c=y, cmap="Spectral", alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(2))
    cbar.set_ticklabels(np.unique(y).astype(np.bool_).tolist())
    plt.title(f"UMAP Embedding of {label} using {n_neighbors=}")
    plt.show()


def _visualise_model(model: List):
    def ordinal(n):
        return "%d%s" % (
            n,
            "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 5))
    min_, max_ = model[0].weights.min(), model[0].weights.max()
    min_, max_ = (
        np.round(min(min_, model[2].weights.min())),
        np.round(max(max_, model[2].weights.max())),
    )
    for i, (ax, data) in enumerate(
        zip(axes.flat, [model[0].weights, model[2].weights])
    ):
        ax.set_axis_off()
        im = ax.imshow(data, cmap="viridis", vmin=min_, vmax=max_)
        ax.title.set_text(
            f"Weights of the {ordinal(i+1)} perceptron {'89x56' if i == 0 else '56x1'}"
        )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

    ticks = np.arange(min_, max_, 0.1).round(decimals=2)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)

    fig.suptitle("Visualising model weights", y=0.1)
    plt.show()


@has_black_patch.command()
def train(
    input_size: int = typer.Option(9),
    output_size: int = typer.Option(1),
    epoches: int = typer.Option(50),
    learning_rate: float = typer.Option(1e-3),
    model_output_fn: Path = typer.Option(Path(".tmp/model_fn.pkl")),
):
    samples = list(generate_data(2000, input_size))

    # # with learning_rate = 1e-2 #
    # model = [
    #     LinearLayer(input_size=input_size * input_size, output_size=56),
    #     ReluActivationLayer(),
    #     LinearLayer(input_size=56, output_size=48),
    #     ReluActivationLayer(),
    #     LinearLayer(input_size=48, output_size=output_size),
    #     SigmoidLayer(input_size=output_size),
    # ]
    # # For other lrs like 1e-3, ie-6 it gets struck in local minima

    model = [
        LinearLayer(input_size=input_size * input_size, output_size=56),
        ReluActivationLayer(),
        LinearLayer(input_size=56, output_size=output_size),
        SigmoidLayer(input_size=output_size),
    ]

    loss_by_epoch = []
    for epoch in range(epoches):
        loss = []
        outcome = []
        for x, y in samples:
            output = x
            for layer in model:
                output = layer.forward(output)
            y_pred = output

            loss.append(bce(y_pred, y))
            outcome.append(prediction_outcome(y_pred, y))

            # Backprop
            de_dx = bce_prime(y_pred, y)
            error_gradient = de_dx
            for layer in reversed(model):
                error_gradient = layer.backward(error_gradient, learning_rate)

        accuracy, precision, recall, f1_score = metrics_from_outcome(outcome)

        loss_by_epoch.append(
            {
                "epoch": epoch,
                "loss": np.mean(loss),
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score,
            }
        )
        if epoch % 5 == 0:
            print(f"Result: epoch:{epoch}  loss:{loss_by_epoch[-1]['loss']}")

    df = pd.json_normalize(loss_by_epoch)
    df.plot.line(
        x="epoch",
    )
    plt.show()

    ## Test
    test_samples = list(generate_data(600, input_size))
    loss = []
    outcome = []
    for x, y in test_samples:
        output = x
        for layer in model:
            output = layer.forward(output)
        y_pred = output

        loss.append(bce(y_pred, y))
        outcome.append(prediction_outcome(y_pred, y))

    accuracy, precision, recall, f1_score = metrics_from_outcome(outcome)

    print(
        f"""Result: test:  loss:{np.mean(loss)}, {accuracy=}, 
        {precision=}, {recall=}, {f1_score=}"""
    )

    with open(model_output_fn, "wb") as f:
        pickle.dump(model, f)


@has_black_patch.command()
def embedding(input_size: int = typer.Option(9), neighbors: int = typer.Option(5)):
    samples = list(generate_data(2000, input_size))
    x, y = list(map(list, zip(*[(x[0, ...], y[0]) for x, y in samples])))

    _plot_embedding(x, y, n_neighbors=neighbors)


@has_black_patch.command()
def feature_embedding(
    input_size: int = typer.Option(9),
    neighbors: int = typer.Option(150),
    model_fn: Path = typer.Option(Path(".tmp/model_fn.pkl")),
):
    with open(model_fn, "rb") as f:
        model = pickle.load(f)

    _visualise_model(model)

    test_samples = list(generate_data(2000, input_size))
    layer1_fm, layer2_fm = [], []
    for x, y in test_samples:
        output = x
        for i, layer in enumerate(model):
            output = layer.forward(output)
            if i == 0:
                layer1_fm.append(output.reshape(-1))
            elif i == 2:
                layer2_fm.append(output.reshape(-1))
            else:
                ...
    x, y = list(map(list, zip(*[(x[0, ...], y[0]) for x, y in test_samples])))

    _plot_embedding(x, y, n_neighbors=neighbors, label="Test Set")
    _plot_embedding(
        layer1_fm, y, n_neighbors=neighbors, label="Features from 1st Perceptron"
    )
    _plot_embedding(
        layer2_fm, y, n_neighbors=neighbors, label="Features from 2nd Perceptron"
    )


if __name__ == "__main__":
    has_black_patch()
