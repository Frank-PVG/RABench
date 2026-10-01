from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets"
RAW_ASSETS = ASSET_ROOT / "raw"
PREPARED_ASSETS = ASSET_ROOT / "prepared"
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIG_ROOT = PROJECT_ROOT / "configs"
BLENDER_BIN = Path(os.environ.get("RGBA_BLENDER_BIN", "/data/gaoshaohan/blender/blender"))
BLENDER_SCRIPT = PROJECT_ROOT / "blender" / "render_scene.py"


def ensure_layout() -> None:
    for path in (ASSET_ROOT, RAW_ASSETS, PREPARED_ASSETS, RUNS_ROOT, CONFIG_ROOT):
        path.mkdir(parents=True, exist_ok=True)
