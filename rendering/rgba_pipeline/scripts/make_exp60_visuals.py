from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COMBOS = ("L1_C1", "L1_C2", "L2_C1", "L2_C2")
TILE = 512
LABEL_H = 36


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tile_image(path: Path, size: int, fallback: str = "missing") -> Image.Image:
    if path.is_file():
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    image = Image.new("RGB", (size, size), (70, 32, 32))
    ImageDraw.Draw(image).text((10, 10), fallback, fill=(255, 255, 255))
    return image


def make_four_up(run: Path, scene_id: str, combo_rows: dict[str, str], folder: str,
                 source_suffix: str, output: Path, *, monochrome: bool = False) -> None:
    canvas = Image.new("RGB", (TILE * 2, (TILE + LABEL_H) * 2), (24, 27, 32))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, combo in enumerate(COMBOS):
        sample_id = combo_rows.get(combo)
        x = (index % 2) * TILE
        y = (index // 2) * (TILE + LABEL_H)
        draw.text((x + 10, y + 10), f"{scene_id}  {combo}", font=font, fill=(240, 244, 248))
        if sample_id:
            path = run / "samples" / sample_id / folder / source_suffix
            if monochrome and path.exists():
                with Image.open(path) as image:
                    image.load()
                    tile = image.convert("L").convert("RGB").resize((TILE, TILE), Image.Resampling.NEAREST)
            else:
                tile = tile_image(path, TILE)
        else:
            tile = tile_image(Path("/missing"), TILE, "sample missing")
        canvas.paste(tile, (x, y + LABEL_H))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def create_visuals(run: Path) -> dict[str, Any]:
    plan = load_jsonl(run / "scene_plan.jsonl")
    manifest = load_jsonl(run / "acquisition_manifest.jsonl")
    by_scene: dict[str, dict[str, str]] = {}
    for row in manifest:
        by_scene.setdefault(row["scene_id"], {})[f"{row['light_id']}_{row['camera_id']}"] = row["sample_id"]

    visual_root = run / "visuals"
    for name in ("four_up", "alpha_four_up", "effect_roi_four_up"):
        (visual_root / name).mkdir(parents=True, exist_ok=True)

    overview_cols = 6
    overview_cell_w = 256
    overview_cell_h = 292
    overview_rows = (len(plan) + overview_cols - 1) // overview_cols
    overview = Image.new("RGB", (overview_cols * overview_cell_w, overview_rows * overview_cell_h), (24, 27, 32))
    draw = ImageDraw.Draw(overview)
    font = ImageFont.load_default()

    for index, scene in enumerate(plan):
        sid = scene["scene_id"]
        combos = by_scene.get(sid, {})
        make_four_up(run, sid, combos, "joint", "preview_checker.png", visual_root / "four_up" / f"{sid}.png")
        make_four_up(run, sid, combos, "joint", "alpha_preview.png", visual_root / "alpha_four_up" / f"{sid}.png", monochrome=True)
        make_four_up(run, sid, combos, "diagnostics", "effect_roi.png", visual_root / "effect_roi_four_up" / f"{sid}.png", monochrome=True)

        first = combos.get("L1_C1")
        image_path = run / "samples" / first / "joint" / "preview_checker.png" if first else Path("/missing")
        thumb = tile_image(image_path, overview_cell_w - 8)
        x = (index % overview_cols) * overview_cell_w
        y = (index // overview_cols) * overview_cell_h
        overview.paste(thumb, (x + 4, y + 30))
        label = f"{sid} · {scene.get('category', '')}"
        draw.text((x + 5, y + 8), label, font=font, fill=(240, 244, 248))

    overview_path = visual_root / "scene_overview.png"
    overview.save(overview_path, format="PNG", optimize=True)
    readme = """# exp60 visuals

- `scene_overview.png`: one representative L1/C1 checker preview for each planned scene.
- `four_up/SCxx.png`: the four official light/camera combinations, rendered checker previews.
- `alpha_four_up/SCxx.png`: matching grayscale alpha previews.
- `effect_roi_four_up/SCxx.png`: interaction-region diagnostics from the QA pass. These are diagnostic masks, not alpha targets.

Each tile caption identifies its exact light and camera combination. Source files remain in the corresponding sample bundle.
"""
    (visual_root / "README.md").write_text(readme, encoding="utf-8")
    return {"scenes": len(plan), "overview": str(overview_path),
            "four_up_count": len(list((visual_root / "four_up").glob("*.png"))),
            "alpha_four_up_count": len(list((visual_root / "alpha_four_up").glob("*.png"))),
            "effect_roi_four_up_count": len(list((visual_root / "effect_roi_four_up").glob("*.png")))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scene overview and four-combination visual contact sheets.")
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(create_visuals(args.run.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
