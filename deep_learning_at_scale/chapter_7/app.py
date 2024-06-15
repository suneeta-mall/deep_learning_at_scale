import typer

from .complete_ddp import ddp
from .ddp_centralized import ddp_centralized
from .ddp_centralized_ray import centralized_ddp_ray
from .distributed_dataset import distribute_iterable
from .efficient_ffcv_ddp import efficient_ddp
from .hogwild import hogwild
from .rpc_centralized import rpc_centralized
from .sharded_ddp import ddp as sharded_ddp

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(distribute_iterable, name="distribute-iterable")

app.add_typer(rpc_centralized, name="rpc")
app.add_typer(ddp_centralized, name="ddp-centralized")
app.add_typer(centralized_ddp_ray, name="ddp-centralized-ray")
app.add_typer(hogwild, name="hogwild")

app.add_typer(ddp, name="ddp")
app.add_typer(sharded_ddp, name="sharded-ddp")
app.add_typer(efficient_ddp, name="efficient-ddp")


@app.command()
def init():
    print("echo")
    return 0
