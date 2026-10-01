from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .budget import BudgetedDownloader, _safe_error
from .discovery import CACHE, MAX_ASSET_BYTES, OBJAVERSE_BASE, _category_uids, _read_gzip_json
from .experiment import SCENES
from .io import sha256, write_json

MAX_SOURCE_CANDIDATES = 30
MAX_PREVIEW_CANDIDATES = 6


def _specific_terms(scene_id: str) -> tuple[str, ...]:
    return {
        "CT07": ("block", "brick", "building block", "cube", "toy block"),
        "CT08": ("chess", "chessman", "pawn", "rook", "bishop", "knight", "king", "queen"),
        "CT12": ("wooden", "block", "toy part", "construction", "brick"),
        "RF02": ("bust", "torso", "bust sculpture"),
        "RF04": ("pitcher", "jug", "ewer", "water jug"),
        "RF05": ("animal", "bird", "cat", "dog", "elephant", "figurine"),
        "RF09": ("watering can", "watering", "water can", "can"),
        "EN02": ("vase", "pottery", "urn", "vessel"),
        "EN04": ("cookware", "pot", "pan", "saucepan", "kettle"),
        "EN06": ("ornament", "decorative", "metal", "sculpture", "figurine"),
        "TR11": ("castle", "fortress", "toy castle"),
        "TR12": ("rocket", "space rocket", "toy rocket"),
        "TR13": ("tripod", "tripod stand"),
        "TR14": ("display stand", "display", "stand", "pedestal", "rack"),
        "TR15": ("dollhouse", "toy house", "miniature house", "house"),
        "OC01": ("chain", "chain link", "chainlink", "necklace"),
        "OC02": ("chain", "chain link", "oval link", "necklace"),
        "OC03": ("rope", "cord", "cable", "wire", "loop"),
        "OC04": ("branch", "tree branch", "twig", "wood branch"),
        "OC05": ("branch", "leafy", "foliage", "tree branch"),
        "OC06": ("fern", "frond", "leaf", "fern frond"),
        "OC07": ("wire basket", "basket", "metal basket"),
        "OC08": ("net", "mesh", "fishing net", "folded net", "basket"),
        "OC09": ("cable", "wire", "cord", "rope", "loop"),
    }.get(scene_id, ())


def _score(scene_id: str, name: str, tags: list[str], category: str) -> int:
    corpus = " ".join([name.lower(), category.lower(), *(str(x).lower() for x in tags)])
    score = 0
    for term in _specific_terms(scene_id):
        if term in corpus:
            score += 5 if " " in term else 3
    task = next(x for x in SCENES if x.scene_id == scene_id)
    for term in (task.primary_name.lower(),):
        significant = [x for x in term.split() if len(x) > 3]
        score += sum(1 for token in significant if token in corpus)
    return score


def discover_expanded_objaverse(run: Path, downloader: BudgetedDownloader,
                                 scene_ids: set[str]) -> dict[str, Any]:
    lvis = _read_gzip_json(CACHE / "lvis-annotations.json.gz")
    paths = _read_gzip_json(CACHE / "object-paths.json.gz")
    existing_path = run / "selection" / "objaverse_candidates.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {"candidates": []}
    existing_by_scene: dict[str, list[dict[str, Any]]] = {}
    preview_by_uid: dict[str, str] = {}
    for row in existing.get("candidates", []):
        existing_by_scene.setdefault(row["scene_id"], []).append(row)
        if row.get("uid") and row.get("thumbnail") and Path(row["thumbnail"]).exists():
            preview_by_uid[row["uid"]] = row["thumbnail"]

    selected: dict[str, list[dict[str, Any]]] = {}
    for task in SCENES:
        if task.scene_id not in scene_ids:
            continue
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in existing_by_scene.get(task.scene_id, []):
            uid = row.get("uid")
            if uid and uid not in seen:
                rows.append({**row, "expanded_candidate": True})
                seen.add(uid)
        for query in task.primary_search:
            for category, uid in _category_uids(lvis, paths, query):
                if uid in seen:
                    continue
                rows.append({"experiment_id": existing.get("experiment_id"), "scene_id": task.scene_id,
                             "role": "unique_primary", "query": query, "lvis_category": category,
                             "dataset": "Objaverse 1.0", "uid": uid, "object_path": paths[uid],
                             "metadata_shard": Path(paths[uid]).parent.name,
                             "selection_status": "metadata_review_pending", "expanded_candidate": True})
                seen.add(uid)
                if len(rows) >= MAX_SOURCE_CANDIDATES:
                    break
            if len(rows) >= MAX_SOURCE_CANDIDATES:
                break
        selected[task.scene_id] = rows[:MAX_SOURCE_CANDIDATES]

    shard_files: dict[str, Path] = {}
    metadata_root = run / "assets" / "metadata"
    for rows in selected.values():
        for row in rows:
            shard = row["metadata_shard"]
            if shard in shard_files:
                continue
            local = metadata_root / f"{shard}.json.gz"
            cached = CACHE / "metadata" / f"{shard}.json.gz"
            if local.exists(): shard_files[shard] = local
            elif cached.exists(): shard_files[shard] = cached
            else:
                record = downloader.fetch(f"{OBJAVERSE_BASE}/metadata/{shard}.json.gz", local,
                    purpose="selected_objaverse_candidate_metadata", asset_id=f"metadata:{shard}",
                    max_file_bytes=16 * 1024 * 1024)
                shard_files[shard] = Path(record["path"])
    metadata_by_shard = {key: _read_gzip_json(path) for key, path in shard_files.items()}
    thumb_root = run / "selection" / "expanded_thumbnails"; thumb_root.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for scene_id, rows in selected.items():
        rated: list[dict[str, Any]] = []
        for row in rows:
            annotation = metadata_by_shard[row["metadata_shard"]].get(row["uid"])
            if not annotation: continue
            license_id = str(annotation.get("license", "")).lower()
            glb = (annotation.get("archives") or {}).get("glb") or {}
            name = str(annotation.get("name", ""))
            tags = [str(x.get("name", "")) for x in annotation.get("tags", [])]
            faces = int(annotation.get("faceCount") or glb.get("faceCount") or 0)
            row.update({"name": name, "tags": tags, "license": license_id,
                        "glb_size": int(glb.get("size") or 0), "face_count": faces,
                        "is_downloadable": bool(annotation.get("isDownloadable")),
                        "source_url": annotation.get("viewerUrl", ""),
                        "author": (annotation.get("user") or {}).get("displayName", ""),
                        "metadata_sha256": sha256(shard_files[row["metadata_shard"]]),
                        "semantic_score": _score(scene_id, name, tags, row.get("lvis_category", ""))})
            row["selection_status"] = "eligible_thumbnail_review"
            if not row["is_downloadable"]: row["selection_status"] = "reject_not_downloadable"
            elif license_id not in {"cc0", "by", "cc0-1.0", "cc-by-4.0"}: row["selection_status"] = "reject_license"
            elif not row["glb_size"] or row["glb_size"] > MAX_ASSET_BYTES: row["selection_status"] = "reject_file_size"
            elif faces > 1_000_000: row["selection_status"] = "reject_face_count"
            rated.append(row)
        rated.sort(key=lambda x: (-int(x.get("semantic_score", 0)), int(x.get("glb_size") or 0), x["uid"]))
        eligible = [x for x in rated if x["selection_status"] == "eligible_thumbnail_review"][:MAX_PREVIEW_CANDIDATES]
        for rank, row in enumerate(eligible, 1):
            images = ((metadata_by_shard[row["metadata_shard"]].get(row["uid"]) or {}).get("thumbnails") or {}).get("images") or []
            images = [x for x in images if x.get("url")]
            images.sort(key=lambda x: abs(int(x.get("width") or 512) - 512))
            row["thumbnail"] = preview_by_uid.get(row["uid"])
            if row["thumbnail"]:
                continue
            for index, image in enumerate(images[:3], 1):
                path = thumb_root / f"{scene_id}_{rank}_{row['uid']}_{index}.jpg"
                try:
                    transfer = downloader.fetch(str(image["url"]), path, purpose="selected_asset_thumbnail",
                        asset_id=f"thumbnail:{scene_id}:{row['uid']}", max_file_bytes=8 * 1024 * 1024)
                except Exception as exc:
                    row.setdefault("preview_errors", []).append(_safe_error(str(exc)) or str(exc)); continue
                row["thumbnail"] = transfer["path"]
                row["thumbnail_sha256"] = transfer["sha256"]
                row["thumbnail_width"] = int(image.get("width") or 0)
                row["thumbnail_height"] = int(image.get("height") or 0)
                break
            if not row["thumbnail"]:
                row["selection_status"] = "reject_missing_preview"
                row["rejection_reason"] = "no available Objaverse thumbnail"
        output.extend(rated)

    result = {"experiment_id": existing.get("experiment_id"),
              "selection_rule": "at most 30 targeted Objaverse UIDs per missing semantic role; rank metadata names/tags before preview; no GLB download",
              "max_candidates_per_scene": MAX_SOURCE_CANDIDATES,
              "max_previews_per_scene": MAX_PREVIEW_CANDIDATES,
              "scenes": sorted(scene_ids), "candidates": output,
              "download_budget": downloader.status()}
    target = run / "selection" / "objaverse_expanded_candidates.json"
    write_json(target, result)
    from PIL import Image, ImageDraw
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in output:
        if row.get("thumbnail"):
            by_scene.setdefault(row["scene_id"], []).append(row)
    for batch in range((len(scene_ids) + 5) // 6):
        ids = sorted(scene_ids)[batch * 6:(batch + 1) * 6]
        tile_w, tile_h, label_h = 210, 150, 38
        sheet = Image.new("RGB", (tile_w * MAX_PREVIEW_CANDIDATES, (tile_h + label_h) * len(ids)), "white")
        draw = ImageDraw.Draw(sheet)
        for line, sid in enumerate(ids):
            for col, row in enumerate(by_scene.get(sid, [])[:MAX_PREVIEW_CANDIDATES]):
                x, y = col * tile_w, line * (tile_h + label_h)
                im = Image.open(row["thumbnail"]).convert("RGB"); im.thumbnail((tile_w - 8, tile_h - 8))
                sheet.paste(im, (x + (tile_w - im.width) // 2, y + (tile_h - im.height) // 2))
                draw.text((x + 3, y + tile_h + 1), f"{sid} {row.get('name','')[:20]}", fill="black")
                draw.text((x + 3, y + tile_h + 17), f"{row['uid'][:10]} s={row.get('semantic_score',0)} {row['selection_status'][:12]}", fill="#555555")
        sheet.save(run / "selection" / f"objaverse_expanded_contact_sheet_{batch+1:02d}.jpg", quality=88)
    return result
