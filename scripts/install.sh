#!/bin/bash

set -euo pipefail

src_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

base_dir=${src_dir}/../

if [ $# -eq 0 ]
  then
    export REQ_FN="requirements.txt"
  else
    export REQ_FN=$1
fi


echo "🤓 Running install @ ${base_dir}"
    
pip install -Uq pip pip-tools
pip install --no-deps -r ${REQ_FN}
pip install --no-deps -qe .
pip check 