from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.0"
EXPANSION_SCHEMA_VERSION = "1.1"
GENERATOR_VERSION = "0.2.0"


@dataclass(frozen=True)
class Asset:
    asset_id: str
    uid: str
    local_path: str
    source_url: str
    license: str
    lineage_id: str


def required_recipe_fields() -> set[str]:
    return {
        "schema_version", "generator_version", "identity", "sets", "objects", "relations",
        "camera", "environment", "background_protocol", "render", "seeds", "diagnostics", "semantics",
    }


def object_by_id(recipe: dict[str, Any], object_id: str) -> dict[str, Any]:
    for obj in recipe["objects"]:
        if obj["id"] == object_id:
            return obj
    raise KeyError(f"Unknown object id: {object_id}")
