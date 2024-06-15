#!/usr/bin/env python3

import typer

from deep_learning_at_scale.chapter_2 import app as chapter_2_app
from deep_learning_at_scale.chapter_3 import app as chapter_3_app
from deep_learning_at_scale.chapter_4 import app as chapter_4_app
from deep_learning_at_scale.chapter_5 import app as chapter_5_app
from deep_learning_at_scale.chapter_7 import app as chapter_7_app
from deep_learning_at_scale.chapter_9 import app as chapter_9_app
from deep_learning_at_scale.chapter_10 import app as chapter_10_app
from deep_learning_at_scale.chapter_11 import app as chapter_11_app
from deep_learning_at_scale.chapter_12 import app as chapter_12_app

uni_app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

uni_app.add_typer(chapter_2_app, name="chapter_2")
uni_app.add_typer(chapter_3_app, name="chapter_3")
uni_app.add_typer(chapter_4_app, name="chapter_4")
uni_app.add_typer(chapter_5_app, name="chapter_5")
uni_app.add_typer(chapter_7_app, name="chapter_7")
uni_app.add_typer(chapter_9_app, name="chapter_9")
uni_app.add_typer(chapter_10_app, name="chapter_10")
uni_app.add_typer(chapter_11_app, name="chapter_11")
uni_app.add_typer(chapter_12_app, name="chapter_12")


def main():
    uni_app()


if __name__ == "__main__":
    main()
