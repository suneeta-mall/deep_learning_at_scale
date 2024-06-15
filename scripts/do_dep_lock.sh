#!/bin/bash

set -euo pipefail

src_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo "🤓 Running locking ${src_dir} to create requirements.txt"

python -m pip install pip-tools

pip-compile -v -o ${src_dir}/../requirements.txt --upgrade --no-emit-index-url \
    --resolver backtracking --generate-hashes --allow-unsafe \
    --extra dev \
    pyproject.toml
