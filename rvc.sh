#!/bin/bash

path=$(dirname `realpath "$0"`)
if ! [ -d "$path/venv" ]; then
    "$path/setup.sh"
fi
. "$path/venv/bin/activate"
python3.8 "$path/tools/infer_cli.py" "$@"
