import typer

from .contrastive_learning import app as cl
from .distillation import app as distller
from .hpo import hpo
from .mnist_moe_baseline import app as mnist_baseline
from .moe import app as moe

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(hpo, name="hpo")
app.add_typer(distller, name="distil")
app.add_typer(moe, name="moe")
app.add_typer(mnist_baseline, name="mnist-baseline")
app.add_typer(cl, name="cl")


@app.command()
def init():
    print("echo")
    return 0
