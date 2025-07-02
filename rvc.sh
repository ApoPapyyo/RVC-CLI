#!/bin/bash

path=$(dirname `realpath "$0"`)
if ! [ -d "$path/venv" ]; then
    "$path/setup.sh"
fi
. "$path/venv/bin/activate"
if [ "$weight_root" = '' ]; then
    export weight_root="$path/models/weights"
fi
wd=$(pwd)
cd "$path"
python3.8 "$path/tools/infer_cli.py" "$wd" "$@"
