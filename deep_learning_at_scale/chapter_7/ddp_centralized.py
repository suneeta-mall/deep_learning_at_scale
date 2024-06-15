import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import typer
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

from .rpc_centralized import SimpleNetwork

__all__ = ["ddp_centralized"]

"""
    time deep-learning-at-scale chapter_7 ddp-centralized train --world-size 3 
    time deep-learning-at-scale chapter_7 ddp-centralized train --world-size 6  
"""

ddp_centralized = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class SimpleDDPNetwork(torch.nn.Module):
    def __init__(self):
        super(SimpleDDPNetwork, self).__init__()
        self.network = DDP(SimpleNetwork())

    def forward(self, x):
        return self.network(x)


def _run_trainer(rank, world_size):
    model = SimpleDDPNetwork()

    def get_loader(rank, world_size):
        train_dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        # this will shuffle the indices so shuffle on loader is disabled
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=64,
        )

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

            loss.backward()
            optimizer.step()
        print(f"Rank {rank} training done for epoch {epoch}")


def run_worker(rank: int, world_size: int, master_address: str, master_port: int):
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_address}:{master_port}",
    )

    _run_trainer(rank, world_size)
    dist.destroy_process_group()


@ddp_centralized.command()
def train(
    world_size: int = typer.Option(
        2,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""",
    ),
    master_address: str = typer.Option(
        "localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""",
        envvar="MASTER_ADDR",
    ),
    master_port: int = typer.Option(
        29501,
        help="""Port that master is listening on, will default to 29501 if not
        provided. Master must be able to accept network traffic on the 
        host and port.""",
        envvar="MASTER_PORT",
    ),
):
    print(f"==== Main process id is {os.getpid()} =====")
    mp.spawn(
        run_worker,
        args=(world_size, master_address, master_port),
        nprocs=world_size,
        join=True,
    )
