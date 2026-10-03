#!/usr/bin/env bash
set -euo pipefail

# Run in the interactive GPU worker shell after `mlx worker login`.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/activate_anime.sh"
cd "$project_root"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
: "${HF_TOKEN:?Set HF_TOKEN in the worker shell before running this script}"
export HF_TOKEN
export PYTHONPATH="$project_root/.runtime/host:$project_root${PYTHONPATH:+:$PYTHONPATH}"

# Source meshes, material partitions, UVs and PBR maps remain in the source GLB.
python -m rgba_pipeline assets fetch --candidate-file assets/texverse_pbr_16.json --count 16 --max-file-mib 256
python -m rgba_pipeline assets prepare --candidate-file assets/texverse_pbr_16.json --count 16 --preserve-source
python scripts/fetch_scene_templates.py --scenes blue_wall pawn_shop the_shed
python scripts/fetch_material_presets.py

python -m rgba_pipeline generate --suite template_pbr_v1 --run-name template_pbr_demo --resolution 1200 --samples 1024 --seed 42
python -m rgba_pipeline batch --manifest runs/template_pbr_demo/acquisition_manifest.jsonl --max-workers 5 --resume
python -m rgba_pipeline noise --run runs/template_pbr_demo --max-workers 5
python -m rgba_pipeline effect-qa --run runs/template_pbr_demo --max-workers 5
python -m rgba_pipeline calibrate --run runs/template_pbr_demo
python -m rgba_pipeline report --run runs/template_pbr_demo
