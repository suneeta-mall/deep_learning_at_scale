#!/bin/bash

set -euo pipefail

src_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo "🤓 Running lint"

LOC="."
FMT_ARGS="--check"
CHK_ARGS=""    


while getopts ":l:f" opt; do
  case $opt in
    l) LOC="$OPTARG"
    ;;
    f) FMT_ARGS=""; CHK_ARGS="--fix"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
  
done

ruff format ${FMT_ARGS} ${LOC}
ruff check ${CHK_ARGS} ${LOC} 
