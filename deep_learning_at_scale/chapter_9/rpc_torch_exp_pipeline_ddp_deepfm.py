import os
from pathlib import Path

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import typer
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.functional import mean_squared_error

from deep_learning_at_scale.chapter_9.deepfm import (
    MovieLensModule,
    to_parallel_data_loader,
)
from deep_learning_at_scale.chapter_9.rpc_torch_exp_pipeline_deepfm import (
    create_pipe_model,
)

torch.autograd.set_detect_anomaly(True)

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

""""
    deep-learning-at-scale chapter_9 rpc-pt-exp-pipeline-hybrid-deepfm train  
    see https://pytorch.org/tutorials/advanced/ddp_pipeline.html
"""


def run_training_loop(
    rank: int,
    world_size: int,
    micro_batch_chunks: int = 2,
    out_dir: Path = Path(".tmp/output"),
    batch_size: int = 240,
    num_workers: int = 5,
    max_epoch: int = 10,
):
    result_dir = out_dir / "chapter_9/rpc_pipeline_deepfm/"
    result_dir.mkdir(exist_ok=True, parents=True)

    datamodule = MovieLensModule(
        data_dir=result_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_genere=False,
    )
    datamodule.setup(None)
    data_loader = to_parallel_data_loader(
        datamodule.train, batch_size, rank, world_size
    )
    devices = [torch.device(f"cuda:{i}") for i in range(world_size - 1)]
    pipe_model = create_pipe_model(
        in_features=datamodule.train.features,
        micro_batch_chunks=micro_batch_chunks,
        # worker_ids=[f"trainer_{i}" for i in range(world_size - 1)],
        devices=devices,
    )
    pipe_model = DDP(pipe_model)
    optimizer = optim.AdamW(
        pipe_model.parameters(),
        lr=1e-3,
        eps=1e-8,
        weight_decay=1e-2,
    )
    for epoch in range(max_epoch):
        loss_sum, mse_sum = torch.tensor([0.0]), torch.tensor([0.0])
        for i, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()

            outputs = pipe_model(x.to(device=devices[0]))
            outputs = outputs.local_value().to(y.device)

            loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
            mse = mean_squared_error(outputs, y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            mse_sum += mse.item()
            if i % 20 == 0:
                print(
                    f"""Rank {rank}, epoch {epoch} training batch {i} 
                    loss {loss_sum/(i+1)}, mse {mse_sum/(i+1)}"""
                )

    print("Training complete!")


def run_worker(
    rank: int,
    world_size: int,
    master_address: str,
    master_port: int,
):
    backend_init = f"tcp://{master_address}:{master_port}"

    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=120,
        init_method=backend_init,
        _transports=["uv"],
    )

    leaders_rank = world_size - 1
    rpc.init_rpc(
        "leaders" if rank == leaders_rank else f"trainer_{rank}",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc_backend_options,
    )

    print(f"Worker {rank} initialized with RPC @ {backend_init}")

    if rank == leaders_rank:
        run_training_loop(
            rank,
            world_size,
            micro_batch_chunks=8,
            out_dir=Path(".tmp/output"),
            batch_size=240,
            num_workers=5,
            max_epoch=10,
        )

    print("RPC initialized!")
    rpc.shutdown()
    print("RPC shutdown")


@app.command()
def train(
    world_size: int = typer.Option(
        3,
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


if __name__ == "__main__":
    app()
