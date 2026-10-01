from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

from .budget import BudgetedDownloader, _safe_error
from .experiment import SCENES
from .io import write_json

API = "https://api.polyhaven.com"
MAX_SEARCH_RESULTS = 30
MAX_INFO_PER_SCENE = 10
MAX_PREVIEWS_PER_SCENE = 6

# Keep this fallback search list restricted to the semantic gaps observed in
# Objaverse LVIS results. No category or asset archive is enumerated or fetched.
SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "CT07": ("wooden building blocks", "toy brick block"),
    "CT12": ("wooden toy building block", "stacking wooden blocks"),
    "TR11": ("toy castle miniature", "castle model"),
    "TR12": ("toy rocket miniature", "rocket model"),
    "OC01": ("round linked chain", "metal chain links"),
    "OC02": ("oval chain links", "linked oval chain"),
    "OC03": ("rope loop", "coiled rope cable"),
    "OC04": ("bare tree branch", "tree branch twig"),
    "OC05": ("leafy branch", "branch with leaves"),
    "OC06": ("fern frond", "fern leaf"),
    "OC07": ("wire basket", "metal mesh basket"),
    "OC08": ("folded fishing net", "net cage mesh"),
    "OC09": ("curved cable assembly", "wire rope loop"),
}


def _lexical_score(scene_id: str, text: str) -> int:
    tokens = {
        "CT07": ("building block", "toy brick", "wooden block", "block", "brick"),
        "CT12": ("wooden block", "toy part", "building block", "stacking block", "block"),
        "TR11": ("toy castle", "castle", "fortress", "miniature"),
        "TR12": ("toy rocket", "space rocket", "rocket", "spaceship"),
        "OC01": ("chain link", "round chain", "chain", "link"),
        "OC02": ("oval chain", "chain link", "chain", "link"),
        "OC03": ("rope loop", "coiled rope", "rope", "cord", "cable"),
        "OC04": ("tree branch", "branch", "twig", "tree"),
        "OC05": ("leafy branch", "branch with leaves", "branch", "foliage", "leaves"),
        "OC06": ("fern frond", "fern", "frond", "leaf"),
        "OC07": ("wire basket", "metal mesh basket", "basket", "wire", "mesh"),
        "OC08": ("folded net", "fishing net", "net cage", "net", "mesh"),
        "OC09": ("cable assembly", "wire rope", "cable", "wire", "rope", "loop"),
    }.get(scene_id, ())
    lowered = text.lower()
    return sum(5 if " " in term and term in lowered else 0 if " " in term else 1 if term in lowered else 0 for term in tokens)


def discover_polyhaven(run: Path, downloader: BudgetedDownloader,
                       scene_ids: set[str]) -> dict[str, Any]:
    """Look up only named semantic gaps; download metadata and individual previews.

    The function never downloads models. A model may only be fetched in a later
    stage after visual review and an individual `/files/{slug}` manifest check.
    """
    run = Path(run)
    root = run / "selection" / "polyhaven"
    root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    seen_global: set[str] = set()
    for task in SCENES:
        sid = task.scene_id
        if sid not in scene_ids:
            continue
        by_slug: dict[str, dict[str, Any]] = {}
        for query_index, query in enumerate(SEARCH_TERMS.get(sid, ()), 1):
            qs = urllib.parse.urlencode({"q": query.lower(), "t": "models", "limit": MAX_SEARCH_RESULTS})
            url = f"{API}/search?{qs}"
            search_path = root / "search" / f"{sid}_{query_index}.json"
            try:
                transfer = downloader.fetch(url, search_path,
                    purpose="targeted_polyhaven_search_metadata",
                    asset_id=f"polyhaven-search:{sid}:{query_index}", max_file_bytes=2 * 1024 * 1024)
                doc = json.loads(Path(transfer["path"]).read_text())
            except Exception as exc:
                candidates.append({"scene_id": sid, "dataset": "Poly Haven", "query": query,
                    "selection_status": "search_failed", "error": _safe_error(str(exc))})
                continue
            for rank, result in enumerate(doc.get("results", [])[:MAX_SEARCH_RESULTS], 1):
                slug = str(result.get("slug") or "").strip()
                if not slug:
                    continue
                record = by_slug.setdefault(slug, {
                    "experiment_id": "exp60_20260930_v1", "scene_id": sid,
                    "role": "targeted_fallback_candidate", "dataset": "Poly Haven",
                    "uid": slug, "slug": slug, "license": "CC0",
                    "search_score": float(result.get("score") or 0),
                    "queries": [], "ranks": []})
                record["queries"].append(query)
                record["ranks"].append(rank)
                record["search_score"] = max(record["search_score"], float(result.get("score") or 0))
        search_rows = sorted(by_slug.values(), key=lambda x: (-x["search_score"], x["slug"]))
        rows = []
        for row in search_rows[:MAX_INFO_PER_SCENE]:
            slug = row["slug"]
            safe_slug = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in slug)
            info_path = root / "info" / f"{safe_slug}.json"
            try:
                transfer = downloader.fetch(f"{API}/info/{urllib.parse.quote(slug, safe='')}", info_path,
                    purpose="targeted_polyhaven_asset_metadata",
                    asset_id=f"polyhaven-info:{slug}", max_file_bytes=2 * 1024 * 1024)
                info = json.loads(Path(transfer["path"]).read_text())
            except Exception as exc:
                row.update({"selection_status": "info_failed", "info_error": _safe_error(str(exc))})
                rows.append(row); continue
            if info.get("type") != 2:
                row["selection_status"] = "reject_not_model"
                rows.append(row); continue
            row.update({"asset_type": "model", "name": info.get("name", slug),
                "description": info.get("description", ""), "authors": info.get("authors", {}),
                "tags": info.get("tags", []), "category": info.get("category", ""),
                "thumbnail_url": info.get("thumbnail_url"), "polycount": info.get("polycount"),
                "max_resolution": info.get("max_resolution"), "files_hash": info.get("files_hash"),
                "metadata_path": str(info_path)})
            corpus = " ".join([row["name"], row["description"], row["category"], *row["tags"]])
            row["lexical_score"] = _lexical_score(sid, corpus)
            row["selection_status"] = "metadata_review_complete"
            rows.append(row)
        rows.sort(key=lambda x: (-int(x.get("lexical_score", -1)), -x.get("search_score", 0), x["slug"]))
        preview_rank = 0
        for row in rows:
            if row.get("asset_type") != "model" or preview_rank >= MAX_PREVIEWS_PER_SCENE:
                continue
            if not row.get("thumbnail_url"):
                row["selection_status"] = "reject_no_preview_url"
                continue
            preview_rank += 1
            safe_slug = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in row["slug"])
            preview_path = root / "thumbnails" / f"{sid}_{preview_rank}_{safe_slug}.webp"
            try:
                transfer = downloader.fetch(row["thumbnail_url"], preview_path,
                    purpose="targeted_polyhaven_asset_thumbnail",
                    asset_id=f"polyhaven-thumbnail:{row['slug']}", max_file_bytes=8 * 1024 * 1024)
                row.update({"thumbnail": transfer["path"], "thumbnail_sha256": transfer["sha256"],
                            "selection_status": "eligible_thumbnail_review"})
            except Exception as exc:
                row["selection_status"] = "reject_thumbnail_unavailable"
                row["preview_error"] = _safe_error(str(exc))
        for row in rows:
            row["duplicate_candidate_slug_across_scenes"] = row["slug"] in seen_global
            seen_global.add(row["slug"])
            candidates.append(row)

    from PIL import Image, ImageDraw
    ids = sorted(scene_ids)
    for batch in range((len(ids) + 5) // 6):
        subset = ids[batch * 6:(batch + 1) * 6]
        tile_w, tile_h, label_h = 210, 150, 38
        sheet = Image.new("RGB", (tile_w * MAX_PREVIEWS_PER_SCENE, (tile_h + label_h) * len(subset)), "white")
        draw = ImageDraw.Draw(sheet)
        for line, sid in enumerate(subset):
            rows = [x for x in candidates if x.get("scene_id") == sid and x.get("thumbnail")]
            rows.sort(key=lambda x: (-x.get("semantic_score", 0), x["slug"]))
            for col, row in enumerate(rows[:MAX_PREVIEWS_PER_SCENE]):
                x, y = col * tile_w, line * (tile_h + label_h)
                im = Image.open(row["thumbnail"]).convert("RGB"); im.thumbnail((tile_w - 8, tile_h - 8))
                sheet.paste(im, (x + (tile_w - im.width) // 2, y + (tile_h - im.height) // 2))
                draw.text((x + 3, y + tile_h + 1), f"{sid} {row.get('name', '')[:20]}", fill="black")
                draw.text((x + 3, y + tile_h + 17), f"{row['slug'][:14]} s={row.get('semantic_score', 0):.2f}", fill="#555555")
        sheet.save(root / f"contact_sheet_{batch + 1:02d}.jpg", quality=88)

    report = {"experiment_id": "exp60_20260930_v1", "dataset": "Poly Haven",
        "license_policy": "CC0 assets only", "selection_rule": "query only semantic gaps; metadata and thumbnail review before individual model manifest/download",
        "search_limit_per_query": MAX_SEARCH_RESULTS, "info_limit_per_scene": MAX_INFO_PER_SCENE, "preview_limit_per_scene": MAX_PREVIEWS_PER_SCENE,
        "scenes": sorted(scene_ids), "candidates": candidates,
        "download_budget": downloader.status()}
    write_json(root / "candidates.json", report)
    return report
