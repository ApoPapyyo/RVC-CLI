#!/bin/bash

path=$(dirname `realpath "$0"`)
if ! [ -d "$path/venv" ]; then
    wd=$(pwd)
    cd "$path"
    "$path/setup.sh"
    cd "$wd"
fi
. "$path/venv/bin/activate"
python3.8 "$path/infer/modules/train/train.py" "$@"
