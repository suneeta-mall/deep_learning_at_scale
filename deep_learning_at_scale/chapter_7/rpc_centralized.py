import os
from typing import Tuple

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms

torch.autograd.set_detect_anomaly(True)

__all__ = ["rpc_centralized"]

rpc_centralized = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

""""
deep-learning-at-scale chapter_7 rpc train --world-size 2      
"""

param_server = None
global_lock = mp.Lock()


# A simple network used for demonstration of rpc
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


class ParameterServer(nn.Module):
    def __init__(self):
        super().__init__()
        model = SimpleNetwork()
        self.model = model

    def forward(self, input):
        input = input
        output = self.model(input)
        return output

    # Wraps the parameters into RPC remote references i.e. RRef.
    # These referenced parameters are used by distributed optimisation
    # algorithm to take optimisations steps (for convergence) based on remote data
    def get_param_remote_references(self):
        param_remote_references = [rpc.RRef(param) for param in self.model.parameters()]
        return param_remote_references


# A thin wrapper over nn.Module that wraps RPC remote invocations to
# call forward() over the network to the parameter server.
class Worker(nn.Module):
    def __init__(self, sync: bool):
        super().__init__()
        self.param_server_rref = rpc.remote("parameter_server", get_parameter_server)
        self.sync = sync

    def get_parameter_remote_reference(self):
        remote_params = self.remote_method(
            ParameterServer.get_param_remote_references, self.param_server_rref
        )
        return remote_params

    def forward(self, x):
        model_output = self.remote_method(
            ParameterServer.forward, self.param_server_rref, x
        )

        return model_output

    # A helper method to call a method with local value held in the RRef
    @staticmethod
    def call_method(method, rref, *args, **kwargs):
        return method(rref.local_value(), *args, **kwargs)

    # Given an remote reference, return the result of calling the remote method
    def remote_method(self, method, rref, *args, **kwargs):
        args = [method, rref] + list(args)
        sync_fn = rpc.rpc_sync if self.sync else rpc.rpc_async
        result = sync_fn(rref.owner(), Worker.call_method, args=args, kwargs=kwargs)

        if self.sync:
            return result

        result.wait()
        return result.value()


def get_parameter_server():
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer()
        return param_server


def run_training_loop(rank: int, train_loader: torch.utils.data.DataLoader, sync: bool):
    # Runs the training loop for SimpleNetwork i.e. forward, backward loop
    # followed by optimization steps but in a distributed fashion.
    worker = Worker(sync=sync)
    param_rrefs = worker.get_parameter_remote_reference()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.01)
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as context_id:
            model_output = worker(data)
            loss = F.nll_loss(model_output, target)
            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}")
            dist_autograd.backward(context_id, [loss], retain_graph=False)
            opt.step(context_id)

    print("Training complete!")


# Main loop for trainers.
def run_worker(
    rank: int, world_size: int, master_address: str, master_port: int, sync: bool
):
    backend_init = f"tcp://{master_address}:{master_port}"
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=20,
        init_method=backend_init,
        _transports=["uv"],
    )

    ps_rank = world_size - 1

    rpc.init_rpc(
        "parameter_server" if rank == ps_rank else f"trainer_{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options,
    )

    print(f"Worker {rank} initialized with RPC @ {backend_init}")

    if rank != ps_rank:
        train_dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        subset_data = train_dataset
        num_workers = world_size - 1
        if num_workers > 1:
            data_size = len(train_dataset)
            indices = list(range(data_size))
            start_idx = int(data_size * rank / num_workers)
            end_idx = int(data_size * (rank + 1) / num_workers)
            subset_data = torch.utils.data.Subset(
                train_dataset,
                indices[start_idx:end_idx],
            )
        train_loader = torch.utils.data.DataLoader(
            subset_data,
            batch_size=64,
            shuffle=True,
        )
        run_training_loop(rank, train_loader, sync=sync)

    print("RPC initialized!")
    rpc.shutdown()
    print("RPC shutdown")


@rpc_centralized.command()
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
    sync: bool = typer.Option(
        False, help="""If rpc communication should happen sync or async"""
    ),
):
    print(f"==== Main process id is {os.getpid()} =====")
    mp.spawn(
        run_worker,
        args=(world_size, master_address, master_port, sync),
        nprocs=world_size,
        join=True,
    )
