from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .budget import BudgetedDownloader, _safe_error
from .experiment import EXPERIMENT_ID, SCENES
from .io import sha256, write_json

OBJAVERSE_BASE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
CACHE = Path.home() / ".objaverse" / "hf-objaverse-v1"
MAX_REVIEW_CANDIDATES = 3
MAX_ASSET_BYTES = 512 * 1024 * 1024


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _category_uids(lvis: dict[str, list[str]], paths: dict[str, str], query: str) -> list[tuple[str, str]]:
    q = norm(query)
    exact = [(name, uids) for name, uids in lvis.items() if norm(name) == q]
    if not exact:
        exact = [(name, uids) for name, uids in lvis.items()
                 if q and (norm(name).startswith(q + "_") or norm(name).endswith("_" + q))]
    result: list[tuple[str, str]] = []
    for category, uids in sorted(exact, key=lambda x: (norm(x[0]) != q, norm(x[0]))):
        for uid in sorted(set(uids), key=lambda u: hashlib.sha256(f"{query}:{u}".encode()).hexdigest()):
            if uid in paths:
                result.append((category, uid))
    return result


def discover_candidates(run: Path, downloader: BudgetedDownloader) -> dict[str, Any]:
    """Index LVIS locally, then fetch only shards for three selected UIDs per root."""
    lvis_path = CACHE / "lvis-annotations.json.gz"
    paths_path = CACHE / "object-paths.json.gz"
    if not lvis_path.exists() or not paths_path.exists():
        raise FileNotFoundError("Objaverse UID/category/path index is missing from the existing local cache")
    lvis = _read_gzip_json(lvis_path)
    object_paths = _read_gzip_json(paths_path)
    selected: dict[str, list[dict[str, Any]]] = {}
    used_primary: set[str] = set()
    no_match: list[dict[str, str]] = []

    for task in SCENES:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for query in task.primary_search:
            matches = _category_uids(lvis, object_paths, query)
            for category, uid in matches:
                if uid in seen or uid in used_primary:
                    continue
                relative = object_paths[uid]
                shard = Path(relative).parent.name
                candidates.append({
                    "experiment_id": EXPERIMENT_ID,
                    "scene_id": task.scene_id,
                    "role": "unique_primary",
                    "query": query,
                    "lvis_category": category,
                    "dataset": "Objaverse 1.0",
                    "uid": uid,
                    "object_path": relative,
                    "metadata_shard": shard,
                    "selection_status": "metadata_review_pending",
                })
                seen.add(uid)
                if len(candidates) >= MAX_REVIEW_CANDIDATES:
                    break
            if len(candidates) >= MAX_REVIEW_CANDIDATES:
                break
        if candidates:
            used_primary.add(candidates[0]["uid"])
            selected[task.scene_id] = candidates
        else:
            no_match.append({"scene_id": task.scene_id,
                             "reason": "No distinct Objaverse LVIS UID in declared search terms"})

    shard_files: dict[str, Path] = {}
    metadata_root = run / "assets" / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for rows in selected.values():
        for row in rows:
            shard = row["metadata_shard"]
            if shard in shard_files:
                continue
            cached = CACHE / "metadata" / f"{shard}.json.gz"
            if cached.exists():
                shard_files[shard] = cached
                continue
            destination = metadata_root / f"{shard}.json.gz"
            record = downloader.fetch(
                f"{OBJAVERSE_BASE}/metadata/{shard}.json.gz", destination,
                purpose="selected_objaverse_metadata_shard", asset_id=f"metadata:{shard}",
                max_file_bytes=16 * 1024 * 1024,
            )
            shard_files[shard] = Path(record["path"])

    metadata_by_shard = {shard: _read_gzip_json(path) for shard, path in shard_files.items()}
    preview_root = run / "selection" / "thumbnails"
    preview_root.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict[str, Any]] = []
    for scene_id, rows in selected.items():
        for position, row in enumerate(rows):
            annotation = metadata_by_shard[row["metadata_shard"]].get(row["uid"])
            if not annotation:
                row["selection_status"] = "reject_metadata_missing"
                row["rejection_reason"] = "selected UID absent from its indexed metadata shard"
                review_rows.append(row)
                continue
            archives = annotation.get("archives") or {}
            glb = archives.get("glb") or {}
            license_id = str(annotation.get("license", "")).lower()
            glb_size = int(glb.get("size") or 0)
            faces = int(annotation.get("faceCount") or glb.get("faceCount") or 0)
            row.update({
                "name": annotation.get("name", ""),
                "tags": [str(x.get("name", "")) for x in annotation.get("tags", [])],
                "viewer_url": annotation.get("viewerUrl", ""),
                "source_url": annotation.get("viewerUrl", ""),
                "license": license_id,
                "author": (annotation.get("user") or {}).get("displayName", ""),
                "glb_size": glb_size,
                "face_count": faces,
                "is_downloadable": bool(annotation.get("isDownloadable")),
                "metadata_sha256": sha256(shard_files[row["metadata_shard"]]),
            })
            valid_license = license_id in {"cc0", "by", "cc0-1.0", "cc-by-4.0"}
            if not row["is_downloadable"]:
                row["selection_status"] = "reject_not_downloadable"
                row["rejection_reason"] = "Objaverse marks the model as not downloadable"
            elif not valid_license:
                row["selection_status"] = "reject_license"
                row["rejection_reason"] = f"license {license_id!r} is outside this run's CC0/CC-BY allowlist"
            elif not glb_size or glb_size > MAX_ASSET_BYTES:
                row["selection_status"] = "reject_file_size"
                row["rejection_reason"] = f"GLB size absent or exceeds {MAX_ASSET_BYTES} bytes"
            elif faces > 1_000_000:
                row["selection_status"] = "reject_face_count"
                row["rejection_reason"] = f"metadata reports {faces} faces"
            else:
                row["selection_status"] = "eligible_thumbnail_review"

            images = ((annotation.get("thumbnails") or {}).get("images") or [])
            suitable = [i for i in images if int(i.get("width") or 0) <= 720 and i.get("url")]
            if not suitable:
                suitable = [i for i in images if i.get("url")]
            thumb = min(suitable, key=lambda i: abs(int(i.get("width") or 720) - 512)) if suitable else None
            thumb = min(suitable, key=lambda i: abs(int(i.get("width") or 720) - 512)) if suitable else None
            thumb_candidates = sorted(suitable, key=lambda i: abs(int(i.get("width") or 720) - 512))
            row["thumbnail"] = None
            preview_errors: list[str] = []
            for thumb_index, image in enumerate(thumb_candidates[:3]):
                thumb_path = preview_root / f"{scene_id}_{position + 1}_{row['uid']}_{thumb_index + 1}.jpg"
                try:
                    thumb_record = downloader.fetch(
                        str(image["url"]), thumb_path, purpose="selected_asset_thumbnail",
                        asset_id=f"thumbnail:{scene_id}:{row['uid']}", max_file_bytes=8 * 1024 * 1024,
                    )
                except Exception as exc:
                    preview_errors.append(_safe_error(str(exc)) or str(exc))
                    continue
                row["thumbnail"] = thumb_record["path"]
                row["thumbnail_sha256"] = thumb_record["sha256"]
                row["thumbnail_width"] = int(image.get("width") or 0)
                row["thumbnail_height"] = int(image.get("height") or 0)
                break
            if row["thumbnail"] is None and row["selection_status"] == "eligible_thumbnail_review":
                row["selection_status"] = "reject_missing_preview"
                row["rejection_reason"] = "selected object has no downloadable preview thumbnail"
            if preview_errors:
                row["preview_errors"] = preview_errors
            review_rows.append(row)

    write_json(run / "selection" / "objaverse_candidates.json", {
        "experiment_id": EXPERIMENT_ID,
        "selection_rule": "UID/category/path index first; per-UID metadata and preview before any GLB download",
        "max_review_candidates_per_scene": MAX_REVIEW_CANDIDATES,
        "selected_primary_uids": sorted(used_primary),
        "no_match": no_match,
        "candidates": review_rows,
        "download_budget": downloader.status(),
    })
    downloader.export_jsonl(run / "asset_download_ledger.jsonl")
    return {"selected_roots": len(selected), "no_match": no_match, "reviewed": len(review_rows),
            "candidates": review_rows, "download_budget": downloader.status()}


def make_contact_sheets(run: Path) -> list[Path]:
    from PIL import Image, ImageDraw

    data = json.loads((run / "selection" / "objaverse_candidates.json").read_text())
    rows = data.get("candidates", [])
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    scene_ids = [x.scene_id for x in SCENES]
    output: list[Path] = []
    tile_w, tile_h, label_h = 210, 150, 38
    for batch in range(10):
        subset = scene_ids[batch * 6:(batch + 1) * 6]
        sheet = Image.new("RGB", (tile_w * MAX_REVIEW_CANDIDATES, (tile_h + label_h) * len(subset)), "white")
        draw = ImageDraw.Draw(sheet)
        for line, scene_id in enumerate(subset):
            for col, row in enumerate(by_scene.get(scene_id, [])[:MAX_REVIEW_CANDIDATES]):
                x, y = col * tile_w, line * (tile_h + label_h)
                p = row.get("thumbnail")
                if p and Path(p).exists():
                    try:
                        im = Image.open(p).convert("RGB")
                        im.thumbnail((tile_w - 8, tile_h - 8))
                        sheet.paste(im, (x + (tile_w - im.width) // 2, y + (tile_h - im.height) // 2))
                    except Exception:
                        pass
                draw.text((x + 3, y + tile_h + 1), f"{scene_id} C{col+1} {row.get('name','')[:18]}", fill="black")
                status = row.get("selection_status", "")
                draw.text((x + 3, y + tile_h + 17), f"{row['uid'][:10]} {status[:19]}", fill="#555555")
        out = run / "selection" / f"objaverse_contact_sheet_{batch+1:02d}.jpg"
        sheet.save(out, quality=86)
        output.append(out)
    return output
