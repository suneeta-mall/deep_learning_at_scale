import os
from typing import Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer
from torchvision import datasets, transforms

__all__ = ["hogwild"]

"""
time deep-learning-at-scale chapter_7 hogwild train --world-size 4       
ps axo pid,ppid,user,command | egrep 93959    
"""


class SimpleNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
    ):
        super().__init__()
        channels, width, height = input_shape

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


"""
    time deep-learning-at-scale chapter_7 hogwild train --world-size 3 
    ps axo pid,ppid,user,command | egrep <your pid of above process>
"""

hogwild = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def get_loader(rank, world_size):
    train_dataset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    subset_data = train_dataset
    if world_size > 1:
        data_size = len(train_dataset)
        indices = [x for x in range(data_size)]
        start_idx = int(data_size * rank / world_size)
        end_idx = int(data_size * (rank + 1) / world_size)
        subset_data = torch.utils.data.Subset(
            train_dataset,
            indices[start_idx:end_idx],
        )
    return torch.utils.data.DataLoader(
        subset_data,
        batch_size=64,
        shuffle=True,
    )


def run_trainer(rank: int, world_size: int, model: nn.Module):
    # Setup distributed optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
    )

    for epoch in range(2):
        train_loader = get_loader(rank, world_size)
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            if i % 5 == 0:
                print(f"Rank {rank} {epoch=} training batch {i} loss {loss.item()}")
            loss.backward(retain_graph=True)
            optimizer.step()
        print("Training done for epoch {}".format(epoch))


@hogwild.command()
def train(
    world_size: int = typer.Option(
        2,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""",
    ),
):
    mp.set_start_method("spawn", force=True)
    print(f"==== Main process id is {os.getpid()} =====")
    model = SimpleNetwork()
    model.share_memory()

    mp.spawn(
        run_trainer,
        args=(world_size, model),
        nprocs=world_size,
        join=True,
    )
