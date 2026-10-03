from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_reference(value: dict[str, Any]) -> str:
    identity = value.get('identity', value)
    return identity.get('recipe_id') or identity['recipe_hash']


def reference_field(recipe: dict[str, Any], prefix: str = '') -> str:
    return prefix + ('recipe_id' if recipe.get('schema_version') == '1.2' else 'recipe_hash')


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
