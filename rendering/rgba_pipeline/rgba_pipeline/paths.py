from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = next((p for p in PROJECT_ROOT.parents if (p / "runtime" / "bpy_runtime").is_dir()), PROJECT_ROOT.parents[1])
# Disposable staging only. Persistent inputs, configuration and results live below PROJECT_ROOT.
# Every caller creates its temporary directories; no pre-existing file here is required.
TEMP_ROOT = WORKSPACE_ROOT / "codex_tmp" / "rgba_pipeline"
ASSET_ROOT = PROJECT_ROOT / "assets"
RAW_ASSETS = ASSET_ROOT / "raw"
PREPARED_ASSETS = ASSET_ROOT / "prepared"
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIG_ROOT = PROJECT_ROOT / "configs"
BLENDER_BIN = Path(os.environ.get("RGBA_BLENDER_BIN", shutil.which("blender") or "blender"))
BLENDER_SCRIPT = PROJECT_ROOT / "blender" / "render_scene.py"
BPY_PYTHON = Path(os.environ.get("RGBA_BPY_PYTHON", "/usr/bin/python3.11"))
BPY_RUNTIME = Path(os.environ.get("RGBA_BPY_RUNTIME", str(WORKSPACE_ROOT / "runtime" / "bpy_runtime")))
BPY_LIBRARIES = Path(os.environ.get("RGBA_BPY_LIBRARIES", str(WORKSPACE_ROOT / "runtime" / "bpy_x11_runtime" / "lib")))


def ensure_layout() -> None:
    for path in (ASSET_ROOT, RAW_ASSETS, PREPARED_ASSETS, RUNS_ROOT, CONFIG_ROOT, TEMP_ROOT):
        path.mkdir(parents=True, exist_ok=True)
