from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from .budget import BudgetedDownloader, _safe_error
from .experiment import SCENES
from .io import write_json

FUEL = "https://fuel.gazebosim.org"
API_VERSION = "1.0"
COLLECTION = "Scanned Objects by Google Research"
OWNER = "GoogleResearch"
PAGE_SIZE = 100
MAX_PREVIEWS_PER_SCENE = 6
MAX_CATALOG_PAGES = 20


def _queries(scene_id: str) -> tuple[str, ...]:
    return {
        "CT07": ("building blocks", "toy bricks", "wood blocks"),
        "CT12": ("wooden toy blocks", "stacking blocks", "building blocks"),
        "TR11": ("toy castle", "castle model", "fortress toy"),
        "TR12": ("toy rocket", "space rocket", "rocket"),
        "OC01": ("round linked chain", "chain links", "necklace chain"),
        "OC02": ("oval chain links", "linked chain", "chain"),
        "OC03": ("rope loop", "rope", "cable"),
        "OC04": ("tree branch", "branch", "twig"),
        "OC05": ("leafy branch", "branch with leaves", "foliage"),
        "OC06": ("fern frond", "fern leaf", "fern"),
        "OC07": ("wire basket", "metal basket", "basket"),
        "OC08": ("folded net", "fishing net", "mesh"),
        "OC09": ("curved cable", "wire rope", "cable"),
    }.get(scene_id, ())


def _score(scene_id: str, text: str) -> int:
    rules = {
        "CT07": (("building blocks", 8), ("wood blocks", 8), ("blocks", 4), ("block", 4), ("brick", 3), ("jenga", 1)),
        "CT12": (("wooden toy blocks", 8), ("stacking blocks", 8), ("building blocks", 7), ("blocks", 4), ("block", 4)),
        "TR11": (("toy castle", 10), ("castle", 8), ("fortress", 5)),
        "TR12": (("toy rocket", 10), ("space rocket", 9), ("rocket", 8), ("spacecraft", 5)),
        "OC01": (("round chain", 10), ("chain links", 9), ("necklace chain", 7), ("chain", 5)),
        "OC02": (("oval chain", 10), ("chain links", 9), ("linked chain", 8), ("chain", 5)),
        "OC03": (("rope loop", 10), ("rope", 8), ("cables", 6), ("cable", 5), ("cord", 4)),
        "OC04": (("tree branch", 10), ("branches", 8), ("branch", 8), ("twig", 7), ("trees", 2), ("tree", 2)),
        "OC05": (("leafy branch", 10), ("branch with leaves", 10), ("branches", 8), ("branch", 8), ("foliage", 4), ("leaves", 4)),
        "OC06": (("fern frond", 10), ("ferns", 8), ("fern", 8), ("frond", 7)),
        "OC07": (("wire basket", 10), ("metal basket", 9), ("basket", 7), ("mesh", 5)),
        "OC08": (("fishing net", 10), ("folded net", 10), ("nets", 8), ("net", 8), ("mesh", 5)),
        "OC09": (("curved cable", 10), ("wire rope", 10), ("cables", 8), ("cable", 8), ("wires", 6), ("wire", 6), ("rope", 4)),
    }
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    padded = f" {normalized} "
    return sum(weight for term, weight in rules.get(scene_id, ())
               if f" {term.lower()} " in padded)


def _catalog(run: Path, downloader: BudgetedDownloader) -> list[dict[str, Any]]:
    root = run / "selection" / "gso" / "catalog"
    root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    collection_query = f"collections:{COLLECTION}"
    for page in range(1, MAX_CATALOG_PAGES + 1):
        path = root / f"page_{page:03d}.json"
        qs = urllib.parse.urlencode({"page": page, "per_page": PAGE_SIZE, "q": collection_query})
        url = f"{FUEL}/{API_VERSION}/models?{qs}"
        transfer = downloader.fetch(url, path, purpose="targeted_gso_catalog_manifest",
            asset_id=f"gso-catalog:page:{page}", max_file_bytes=4 * 1024 * 1024)
        rows = json.loads(Path(transfer["path"]).read_text())
        if not isinstance(rows, list):
            raise ValueError(f"unexpected GSO catalog response on page {page}")
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
    else:
        raise RuntimeError(f"GSO catalog exceeded the bounded {MAX_CATALOG_PAGES}-page lookup")
    return all_rows


def discover_gso(run: Path, downloader: BudgetedDownloader,
                 scene_ids: set[str]) -> dict[str, Any]:
    """Search the GSO online catalog and download only previews, never model ZIPs."""
    run = Path(run)
    root = run / "selection" / "gso"
    root.mkdir(parents=True, exist_ok=True)
    catalog = _catalog(run, downloader)
    output: list[dict[str, Any]] = []
    globally_seen: set[str] = set()
    for sid in sorted(scene_ids):
        ranked = []
        for item in catalog:
            model_name = str(item.get("name") or "")
            description = str(item.get("description") or "")
            text = f"{model_name} {description} {' '.join(item.get('categories') or [])}"
            score = _score(sid, text)
            if score <= 0:
                continue
            ranked.append({"experiment_id": "exp60_20260930_v1", "scene_id": sid,
                "role": "targeted_fallback_candidate", "dataset": "Google Scanned Objects",
                "uid": model_name, "name": model_name, "description": description,
                "owner": item.get("owner", OWNER), "license": item.get("license_name"),
                "license_url": item.get("license_url"), "filesize": int(item.get("filesize") or 0),
                "category": item.get("categories", []), "semantic_score": score,
                "thumbnail_url": f"{FUEL}/{API_VERSION}/{str(item.get('thumbnail_url') or '').lstrip('/')}",
                "source_url": f"{FUEL}/{API_VERSION}/{OWNER}/models/{urllib.parse.quote(model_name, safe='')}",
                "model_url": f"{FUEL}/{API_VERSION}/{OWNER}/models/{urllib.parse.quote(model_name, safe='')}.zip",
                "collection": COLLECTION, "queries": list(_queries(sid)),
                "selection_status": "metadata_review_pending"})
        ranked.sort(key=lambda x: (-x["semantic_score"], x["name"].lower()))
        for row in ranked[:MAX_PREVIEWS_PER_SCENE]:
            raw_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in row["name"])
            preview = root / "thumbnails" / f"{sid}_{raw_name}.jpg"
            try:
                transfer = downloader.fetch(row["thumbnail_url"], preview,
                    purpose="targeted_gso_asset_thumbnail",
                    asset_id=f"gso-thumbnail:{row['name']}", max_file_bytes=8 * 1024 * 1024)
                row.update({"thumbnail": transfer["path"], "thumbnail_sha256": transfer["sha256"],
                            "selection_status": "eligible_thumbnail_review"})
            except Exception as exc:
                row.update({"selection_status": "reject_thumbnail_unavailable",
                            "preview_error": _safe_error(str(exc))})
        for row in ranked:
            row["duplicate_candidate_name_across_scenes"] = row["name"] in globally_seen
            globally_seen.add(row["name"])
            output.append(row)

    from PIL import Image, ImageDraw
    ids = sorted(scene_ids)
    for batch in range((len(ids) + 5) // 6):
        subset = ids[batch * 6:(batch + 1) * 6]
        tile_w, tile_h, label_h = 210, 150, 38
        sheet = Image.new("RGB", (tile_w * MAX_PREVIEWS_PER_SCENE, (tile_h + label_h) * len(subset)), "white")
        draw = ImageDraw.Draw(sheet)
        for line, sid in enumerate(subset):
            rows = [x for x in output if x["scene_id"] == sid and x.get("thumbnail")]
            for col, row in enumerate(rows[:MAX_PREVIEWS_PER_SCENE]):
                x, y = col * tile_w, line * (tile_h + label_h)
                im = Image.open(row["thumbnail"]).convert("RGB"); im.thumbnail((tile_w - 8, tile_h - 8))
                sheet.paste(im, (x + (tile_w - im.width) // 2, y + (tile_h - im.height) // 2))
                draw.text((x + 3, y + tile_h + 1), f"{sid} {row['name'][:20]}", fill="black")
                draw.text((x + 3, y + tile_h + 17), f"{row['name'][:16]} score={row['semantic_score']}", fill="#555555")
        sheet.save(root / f"contact_sheet_{batch + 1:02d}.jpg", quality=88)

    report = {"experiment_id": "exp60_20260930_v1", "dataset": "Google Scanned Objects",
        "catalog_scope": COLLECTION, "catalog_items_indexed": len(catalog),
        "catalog_page_size": PAGE_SIZE, "model_files_downloaded": 0,
        "selection_rule": "search catalog manifest and individual model preview first; fetch only exact ZIP URLs for chosen entries",
        "scenes": sorted(scene_ids), "candidates": output,
        "download_budget": downloader.status()}
    write_json(root / "candidates.json", report)
    return report
