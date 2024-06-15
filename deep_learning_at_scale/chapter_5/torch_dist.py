import os
from typing import Callable, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import typer

torch_dist = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def run(world_rank: int, group: List[int], use_async: bool) -> None:
    """
    4. Each rank process has a local tensor of size = group size.
       Sum operation is conducted on this collective tensor across the group
       resulting in each process with final reduced sum
    """
    tensor = torch.arange(len(group)) + 2 * world_rank
    print(f"PreReduce: Data on rank {world_rank} is {tensor.numpy()}")
    req = dist.all_reduce(
        tensor, op=dist.ReduceOp.SUM, group=dist.new_group(group), async_op=use_async
    )

    if use_async:
        req.wait()
    print(f"PostReduce: Data on rank {world_rank} is {tensor.numpy()}")
    return


def init_process(
    world_rank: int,
    world_size: int,
    use_sync: bool,
    fn: Callable,
    backend="gloo",
):
    """
    3. Initialize the distributed backend for a collective group.
    Creates a group of total process of size world_size. Each process in the group is
    initialised to coordinate with master at local host and a TCP port.
    Each process executes task given by function fn
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=world_rank, world_size=world_size)
    group = list(range(world_size))
    fn(world_rank, group, use_sync)


@torch_dist.command()
def run_distributed(
    world_size: int = typer.Option(2),
    use_async: bool = typer.Option(False),
):
    # 1. Start a group of process, number given by  world_size,
    #  in spawn multiprocessing context
    mp.set_start_method("spawn")
    processes = []
    for world_rank in range(world_size):
        # 2. Initialise each process for their rank,
        # group discovery as defined in `init_process``
        p = mp.Process(
            target=init_process,
            args=(world_rank, world_size, use_async, run),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    torch_dist()
