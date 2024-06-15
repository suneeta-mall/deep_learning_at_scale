import itertools
from typing import Optional

import torch
import torch.multiprocessing as mp
import typer

__all__ = ["distribute_iterable"]

"""
    deep-learning-at-scale chapter_7 rpc-ddp train --world-size 3 
    time deep-learning-at-scale chapter_7 rpc-ddp train --world-size 6  
"""

distribute_iterable = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def fibonacci(max: Optional[int] = 1000):
    a, b = 0, 1
    while True:
        if max < a:
            break
        yield a
        a, b = b, a + b


class FibonacciDataset(torch.utils.data.IterableDataset):
    """
    A simple iterable dataset that creates fibonacci sequence
    """

    def __init__(self):
        super().__init__()
        self.f = fibonacci()

    def __iter__(self):
        return iter(self.f)


class DistributedFibonacciDataset(torch.utils.data.IterableDataset):
    """
    A distributed version of iterable dataset that creates fibonacci sequence
    and uniquely allocates it to each worker as per round robin allocation
    """

    def __init__(self, rank: int, world_size: int, max: Optional[int] = 100):
        super().__init__()
        self.f = fibonacci(max=max)
        self.sequence = itertools.cycle(range(0, world_size))
        self.rank = rank

    def __iter__(self):
        for shard, value in zip(self.sequence, self.f):
            if shard == self.rank:
                yield value


def _run_trainer(rank: int, world_size):
    iterable_dataset = DistributedFibonacciDataset(rank=rank, world_size=world_size)
    train_loader = torch.utils.data.DataLoader(
        iterable_dataset,
        batch_size=64,
    )
    for data in train_loader:
        print(rank, data)


@distribute_iterable.command()
def start_worker(
    world_size: int = typer.Option(
        2,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""",
    ),
):
    mp.spawn(
        _run_trainer,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
