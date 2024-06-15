import typer

from .qlora_llama_2 import app as qlora_app

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(qlora_app, name="qlora")


@app.command()
def init():
    print("echo")
    return 0
