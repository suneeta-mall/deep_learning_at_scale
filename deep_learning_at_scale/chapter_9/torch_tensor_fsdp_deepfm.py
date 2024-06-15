from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import typer
from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torchmetrics.functional import mean_squared_error


from deep_learning_at_scale.chapter_9.deepfm import (
    MovieLensModule,
    to_parallel_data_loader,
)

__all__ = ["app"]

"""
deep-learning-at-scale chapter_9 pt-tensor-ddp-deepfm train
"""


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=10334, embedding_dim=32)
        self.bias = torch.nn.Parameter(torch.zeros((1, 32)))
        self.offsets = [0, 610]

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.embedding(x), dim=1) + self.bias


class RecoHead(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        return torch.nn.functional.sigmoid(self.linear(x).squeeze(1))


def run_worker(
    rank: int,
    world_size: int,
    master_address: str,
    master_port: int,
    tp_parallelism: int,
):
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_address}:{master_port}",
    )
    torch.cuda.set_device(rank)

    _run_trainer(rank, world_size, tp_parallelism)

    dist.destroy_process_group()


def _run_trainer(
    rank: int,
    world_size: int,
    tp_parallelism: int,
    out_dir: Path = Path(".tmp/output"),
    max_epochs: int = 10,
    batch_size: int = 240,
    num_workers: int = 3,
):
    result_dir = out_dir / "chapter_9/deepfm/pt/tensor_parallel"
    result_dir.mkdir(exist_ok=True, parents=True)

    datamodule = MovieLensModule(
        data_dir=result_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_genere=False,
    )
    datamodule.setup(None)

    device_mesh = DeviceMesh(
        "cuda", torch.arange(0, world_size).view(-1, tp_parallelism)
    )

    rank_device = torch.device(f"cuda:{rank}")

    model = torch.nn.Sequential(EmbeddingLayer(), RecoHead()).to(rank_device)
    model = parallelize_module(
        model,
        device_mesh=device_mesh,
        parallelize_plan=ColwiseParallel(),
        tp_mesh_dim=1,
    )
    dp_pg = device_mesh.get_dim_groups()[0]

    model = FSDP(model, process_group=dp_pg)

    # fsdp_auto_wrap_policy can be used to shard models by number of params. For ex:
    # model = FSDP(
    #     model,
    #     fsdp_auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=50),
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        eps=1e-8,
        weight_decay=1e-2,
    )
    dp_rank = dist.get_rank(dp_pg)
    dp_train_loader = to_parallel_data_loader(
        datamodule.train, batch_size, dp_rank, dist.get_world_size()
    )
    dp_val_loader = to_parallel_data_loader(
        datamodule.val, batch_size, dp_rank, dist.get_world_size()
    )
    for epoch in range(max_epochs):
        metrics_loss = torch.zeros(3).to(rank_device)

        for i, (x, y) in enumerate(dp_train_loader):
            x, y = x.to(rank_device), y.to(rank_device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = torch.nn.functional.binary_cross_entropy(
                outputs, y.float(), reduction="sum"
            )
            mse = mean_squared_error(outputs, y)
            loss.sum().backward()
            optimizer.step()

            metrics_loss[0] += loss.item()
            metrics_loss[1] += mse.item()
            metrics_loss[2] += len(x)

            if rank == 0:
                if i % 100 == (99):
                    print(
                        f"""Train Epoch: {epoch} 
                        \tLoss: {metrics_loss[0] /metrics_loss[2] :.6f} 
                        \tMSE: {metrics_loss[1]/metrics_loss[2]:.6f}"""
                    )

    with torch.no_grad():
        metrics_loss = torch.zeros(3).to(rank)
        for i, (x, y) in enumerate(dp_val_loader):
            x, y = x.to(rank_device), y.to(rank_device)
            outputs = model(x)
            y = y.to(outputs.device)
            loss = torch.nn.functional.binary_cross_entropy(
                outputs, y.float(), reduction="sum"
            )
            mse = mean_squared_error(outputs, y)

            metrics_loss[0] += loss.item()
            metrics_loss[1] += mse.item()
            metrics_loss[2] += len(x)

        if rank == 0:
            print(
                f"""Validation Epoch: {epoch} 
                \tLoss: {metrics_loss[0]/metrics_loss[2]:.6f} 
                \tMSE: {metrics_loss[1]/metrics_loss[2]:.6f}"""
            )

    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(states, "model_checkpoint.pt")


@app.command()
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
    tp_parallelism: int = typer.Option(
        2,
        help="""The level of tensor parallism
        """,
    ),
):
    if world_size > torch.cuda.device_count():
        raise AssertionError("not enough GPU")

    mp.spawn(
        run_worker,
        args=(
            world_size,
            master_address,
            master_port,
            tp_parallelism,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    app()
