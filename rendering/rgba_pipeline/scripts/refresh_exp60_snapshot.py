from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("README.md", "pyproject.toml", "uv.lock", "environment.json",
           "blender", "rgba_pipeline", "scripts", "tests")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "*.pyc"}


def _files() -> list[Path]:
    result: list[Path] = []
    for name in INCLUDE:
        source = PROJECT_ROOT / name
        if source.is_file():
            result.append(source)
        elif source.is_dir():
            result.extend(path for path in source.rglob("*") if path.is_file()
                          and not (set(path.parts) & EXCLUDE_PARTS)
                          and path.suffix != ".pyc")
    return sorted(set(result))


def refresh(run: Path) -> dict[str, Any]:
    run = run.resolve()
    snapshot = run / "source_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True)
    inventory = []
    for source in _files():
        relative = source.relative_to(PROJECT_ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        inventory.append({"path": relative.as_posix(), "sha256": digest, "bytes": source.stat().st_size})
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snapshot_doc = {"captured_at": captured, "file_count": len(inventory), "inventory": inventory}
    (snapshot / "inventory.json").write_text(json.dumps(snapshot_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    experiment_path = run / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment["source_snapshot"] = snapshot_doc
    experiment_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"snapshot": str(snapshot), "captured_at": captured, "file_count": len(inventory)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/refresh_exp60_snapshot.py RUN_DIRECTORY")
    print(json.dumps(refresh(Path(sys.argv[1])), ensure_ascii=False, indent=2))
