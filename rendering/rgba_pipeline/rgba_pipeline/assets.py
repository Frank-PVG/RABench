from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import append_jsonl, directory_size, iter_jsonl, sha256, write_json
from .paths import ASSET_ROOT, PREPARED_ASSETS, RAW_ASSETS, ensure_layout

GIB = 1024**3
DEFAULT_DOWNLOAD_BUDGET = GIB
DEFAULT_CACHE_BUDGET = 2 * GIB
HARD_LIMIT = 10 * GIB
LEDGER = ASSET_ROOT / "download_ledger.jsonl"
MANIFEST = ASSET_ROOT / "asset_manifest.jsonl"


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    asset_id: str
    uid: str
    url: str
    license: str
    source_page: str


def _downloaded_bytes() -> int:
    return sum(int(row.get("received_bytes", 0)) for row in iter_jsonl(LEDGER) if row.get("event") == "attempt")


def budget_status() -> dict[str, int]:
    ensure_layout()
    return {
        "downloaded_bytes": _downloaded_bytes(),
        "asset_cache_bytes": directory_size(ASSET_ROOT),
        "download_budget_bytes": DEFAULT_DOWNLOAD_BUDGET,
        "cache_budget_bytes": DEFAULT_CACHE_BUDGET,
        "hard_limit_bytes": HARD_LIMIT,
    }


def _already_complete(candidate: Candidate) -> Path | None:
    target = RAW_ASSETS / f"{candidate.asset_id}_{candidate.uid}.glb"
    return target if target.exists() and target.stat().st_size else None


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "rgba-pipeline/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except (urllib.error.URLError, ValueError):
        return None


def download_candidate(candidate: Candidate, *, retries: int = 2) -> Path:
    """Download one file serially with a persistent, non-resettable byte ledger."""
    ensure_layout()
    existing = _already_complete(candidate)
    if existing:
        return existing
    target = RAW_ASSETS / f"{candidate.asset_id}_{candidate.uid}.glb"
    known_length = _content_length(candidate.url)
    status = budget_status()
    if known_length is not None and known_length > 64 * 1024 * 1024:
        raise BudgetExceeded(f"{candidate.asset_id}: file exceeds 64 MiB candidate cap")
    if known_length is not None and status["downloaded_bytes"] + known_length > DEFAULT_DOWNLOAD_BUDGET:
        raise BudgetExceeded("download budget would be exceeded")
    if known_length is not None and status["asset_cache_bytes"] + known_length > DEFAULT_CACHE_BUDGET:
        raise BudgetExceeded("asset-cache budget would be exceeded")

    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        temporary = target.with_suffix(f".attempt{attempt}.tmp")
        received = 0
        try:
            request = urllib.request.Request(candidate.url, headers={"User-Agent": "rgba-pipeline/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    now = budget_status()
                    if now["downloaded_bytes"] + received + len(chunk) > DEFAULT_DOWNLOAD_BUDGET:
                        raise BudgetExceeded("stream would exceed 1 GiB download budget")
                    if directory_size(ASSET_ROOT) + len(chunk) > DEFAULT_CACHE_BUDGET:
                        raise BudgetExceeded("stream would exceed 2 GiB asset-cache budget")
                    output.write(chunk)
                    received += len(chunk)
            if received == 0:
                raise RuntimeError("empty download")
            temporary.replace(target)
            append_jsonl(LEDGER, {"event": "attempt", "asset_id": candidate.asset_id, "uid": candidate.uid,
                                  "url": candidate.url, "attempt": attempt, "status": "complete",
                                  "received_bytes": received, "sha256": sha256(target), "timestamp": time.time()})
            return target
        except Exception as exc:
            last_error = str(exc)
            append_jsonl(LEDGER, {"event": "attempt", "asset_id": candidate.asset_id, "uid": candidate.uid,
                                  "url": candidate.url, "attempt": attempt, "status": "failed",
                                  "received_bytes": received, "error": last_error, "timestamp": time.time()})
            if temporary.exists():
                temporary.unlink()
            if isinstance(exc, BudgetExceeded):
                raise
    raise RuntimeError(f"Failed to download {candidate.asset_id}: {last_error}")


def load_candidates(path: Path | None) -> list[Candidate]:
    """Load explicit provenance-rich candidates; no implicit full-dataset crawl."""
    if path is None:
        path = ASSET_ROOT / "candidates.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate list is required at {path}. It must contain up to 12 individually selected Objaverse GLB URLs."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("candidates.json must contain 1 to 12 candidates")
    result = [Candidate(**row) for row in rows]
    if len({item.asset_id for item in result}) != len(result):
        raise ValueError("candidate asset_id values must be unique")
    return result


def fetch_selected(candidate_file: Path | None = None) -> list[Path]:
    """Download until three valid raw assets are present, never exceeding budgets."""
    candidates = load_candidates(candidate_file)
    selected: list[Path] = []
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        if len(selected) == 3:
            break
        try:
            selected.append(download_candidate(candidate))
        except Exception as exc:
            rejected.append({"asset_id": candidate.asset_id, "reason": str(exc)})
    write_json(ASSET_ROOT / "fetch_result.json", {"selected": [str(p) for p in selected], "rejected": rejected,
                                                     "budget": budget_status()})
    if len(selected) < 3:
        raise RuntimeError(f"Only {len(selected)} usable downloads; see assets/fetch_result.json")
    return selected


def raw_assets() -> Iterable[Path]:
    return sorted(RAW_ASSETS.glob("*.glb"))


def write_prepared_manifest(records: list[dict[str, Any]]) -> None:
    MANIFEST.unlink(missing_ok=True)
    for record in records:
        append_jsonl(MANIFEST, record)


def prepared_manifest() -> list[dict[str, Any]]:
    return list(iter_jsonl(MANIFEST))
