#!/usr/bin/env bash
set -euo pipefail

# Run from the interactive GPU worker shell after `mlx worker login`.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/activate_anime.sh"
cd "$project_root"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
: "${HF_TOKEN:?Set HF_TOKEN in the worker shell before running this script}"
export HF_TOKEN
export PYTHONPATH="$project_root/.runtime/host:$project_root${PYTHONPATH:+:$PYTHONPATH}"

python -m rgba_pipeline isolate --run runs/template_pbr_demo --output runs/single_object_demo --resolution 1024 --samples 512 --seed 42 --framing object --max-workers 2 --resume
