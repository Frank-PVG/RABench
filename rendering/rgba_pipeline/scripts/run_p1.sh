#!/usr/bin/env bash
set -euo pipefail

# Enter an interactive GPU worker with `mlx worker login` first.
# Image QA calls Luna for only the three IDs in configs/p1/suite.json.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/scripts/activate_anime.sh"
cd "$project_root"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
: "${HF_TOKEN:?Set HF_TOKEN in the worker shell before running this script}"
export HF_TOKEN
export PYTHONPATH="$project_root/.runtime/host:$project_root${PYTHONPATH:+:$PYTHONPATH}"

python -m rgba_pipeline generate --suite p1_pairs_v1 --run-name p1_small_batch --resolution 1024 --samples 512 --seed 42
python -m rgba_pipeline p1-render --run runs/p1_small_batch --max-workers 2 --resume
python -m rgba_pipeline luna-annotate --run runs/p1_small_batch --limit 3 --workers 3 --model-id gpt-5.6-luna --reasoning-effort high --max-tokens 4096 --resume
python -m rgba_pipeline report --run runs/p1_small_batch
