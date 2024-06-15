import typer

from .data_diet import data_diet
from .outlier_detection import od

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(data_diet, name="data-diet")
app.add_typer(od, name="od")


@app.command()
def init():
    print("echo")
    return 0
