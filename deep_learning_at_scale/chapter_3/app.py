import typer

from .crawler import crawler

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(crawler, name="crawler")


@app.command()
def init():
    print("echo")
    return 0
