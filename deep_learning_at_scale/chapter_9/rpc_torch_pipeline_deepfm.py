import os
import threading
from pathlib import Path
from typing import List

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import typer
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torchmetrics.functional import mean_squared_error

from deep_learning_at_scale.chapter_9.deepfm import (
    MLP,
    FactorizationMachine,
    FeaturesEmbedding,
    LinearFeatureEmbedding,
    MovieLensModule,
)

torch.autograd.set_detect_anomaly(True)

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

""""
    deep-learning-at-scale chapter_9 rpc-pt-pipeline-deepfm train  
"""


class RPCFeaturesEmbedding(torch.nn.Module):
    def __init__(self, device: torch.device, in_features: List[int], out_features: int):
        super().__init__()
        self.device = device
        self._lock = threading.Lock()

        self.embedding = FeaturesEmbedding(
            in_features=in_features, out_features=out_features
        ).to(device=device)

    def forward(self, x_rref: RRef):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.embedding(x)
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class RPCLinearHead(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        in_features: List[int],
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.device = device
        self._lock = threading.Lock()

        self.linear = LinearFeatureEmbedding(
            in_features=in_features, out_features=1
        ).to(device=device)
        self.fm = FactorizationMachine().to(device=device)
        self.deep_fm_embedding_size = len(in_features) * hidden_features
        self.mlp = MLP(
            embedding_size=self.deep_fm_embedding_size,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
        ).to(device=device)
        self.sigmoid = torch.nn.Sigmoid().to(device=device)

    def forward(self, x_rref: RRef, embeddings_rref: RRef):
        x = x_rref.to_here().to(self.device)
        embeddings = embeddings_rref.to_here().to(self.device)
        with self._lock:
            linear_features = self.linear(x)
            linear_features += self.fm(embeddings)
            linear_features += self.mlp(
                embeddings.view(-1, self.deep_fm_embedding_size)
            )
            out = self.sigmoid(linear_features.squeeze(1))
        return out.cpu()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class RPCDeepFactorizationMachineModel(torch.nn.Module):
    def __init__(
        self,
        in_features: List[int],
        devices: List[torch.device],
        micro_batch_chunks: int,
        worker_ids: List[str],
        hidden_features: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.micro_batch_chunks = micro_batch_chunks

        self.embedding = rpc.remote(
            worker_ids[0],
            RPCFeaturesEmbedding,
            args=(devices[0], in_features, hidden_features),
        )

        self.linear_head = rpc.remote(
            worker_ids[1],
            RPCLinearHead,
            args=(
                devices[1],
                in_features,
                hidden_features,
                dropout_rate,
            ),
        )

    def _forward_micro(self, x: RRef) -> torch.Tensor:
        xs_rref = RRef(x)
        embeddings = self.embedding.remote().forward(xs_rref)
        # result = self.linear_head.rpc_async().forward(xs_rref, embeddings)
        result = self.linear_head.rpc_sync().forward(xs_rref, embeddings)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        micro_outputs = []
        for xs in iter(x.split(self.micro_batch_chunks, dim=0)):
            micro_outputs.append(self._forward_micro(xs))
        # return torch.cat(torch.futures.wait_all(micro_outputs))
        return torch.cat(micro_outputs)

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.embedding.remote().parameter_rrefs().to_here())
        remote_params.extend(self.linear_head.remote().parameter_rrefs().to_here())
        return remote_params


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
    data_loader = datamodule.train_dataloader()

    dist_model = RPCDeepFactorizationMachineModel(
        in_features=datamodule.train.features,
        micro_batch_chunks=micro_batch_chunks,
        worker_ids=[f"trainer_{i}" for i in range(world_size - 1)],
        devices=[torch.device(f"cuda:{i}") for i in range(world_size - 1)],
    )
    optimizer = DistributedOptimizer(
        optim.AdamW,
        dist_model.parameter_rrefs(),
        lr=1e-3,
        eps=1e-8,
        weight_decay=1e-2,
    )
    for epoch in range(max_epoch):
        loss_sum, mse_sum = torch.tensor([0.0]), torch.tensor([0.0])
        for i, (x, y) in enumerate(data_loader):
            with dist_autograd.context() as context_id:
                outputs = dist_model(x)
                loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
                mse = mean_squared_error(outputs, y)
                dist_autograd.backward(context_id, [loss], retain_graph=False)
                optimizer.step(context_id)
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
