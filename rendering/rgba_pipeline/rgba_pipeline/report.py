from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

from .io import directory_size, read_json, write_json


CATEGORY_ORDER = (
    "simple_control",
    "contact_shadow",
    "reflection_color_spill",
    "environmental_influence",
    "translucent_overlap",
    "interleaved_occlusion",
)
CATEGORY_LABELS = {
    "simple_control": "简单对照",
    "contact_shadow": "接触与相互投影",
    "reflection_color_spill": "反射与颜色溢出",
    "environmental_influence": "环境影响",
    "translucent_overlap": "半透明相互覆盖",
    "interleaved_occlusion": "交错遮挡与细结构",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _download_totals(run: Path) -> dict[str, int]:
    db = run / "download_budget.sqlite"
    if db.exists():
        try:
            with sqlite3.connect(db) as connection:
                row = connection.execute(
                    "SELECT COALESCE(SUM(received_bytes),0), COALESCE(SUM(reserved_bytes),0), COUNT(*) FROM transfers"
                ).fetchone()
            return {"downloaded_bytes": int(row[0]), "reserved_bytes": int(row[1]), "transfer_count": int(row[2])}
        except sqlite3.Error:
            pass
    selection = run / "selection" / "selected_assets.json"
    if selection.exists():
        budget = read_json(selection).get("download_budget", {})
        return {"downloaded_bytes": int(budget.get("received_bytes", 0)),
                "reserved_bytes": int(budget.get("reserved_bytes", 0)), "transfer_count": 0}
    return {"downloaded_bytes": 0, "reserved_bytes": 0, "transfer_count": 0}


def make_report(run: Path) -> Path:
    run = Path(run).resolve()
    experiment = read_json(run / "experiment.json") if (run / "experiment.json").exists() else {}
    if experiment.get('suite') == 'p1_pairs_v1':
        from .p1_report import make_p1_report
        return make_p1_report(run, experiment)
    if experiment.get('suite') == 'isolated_objects_v1':
        return make_isolation_report(run)
    if experiment.get('suite') == 'template_pbr_v1':
        return make_template_report(run, experiment)
    plan = _jsonl(run / "scene_plan.jsonl")
    manifest = _jsonl(run / "acquisition_manifest.jsonl")
    qa_by_id: dict[str, dict[str, Any]] = {}
    for root in sorted((run / "samples").glob("*")):
        qa_path = root / "qa.json"
        if not qa_path.exists():
            continue
        item = read_json(qa_path)
        sid = item.get("sample_id", root.name)
        noise_path = root / "noise_estimate" / "noise_statistics.json"
        if noise_path.exists():
            item["noise_statistics"] = read_json(noise_path)
        effect_path = root / "diagnostics" / "category_effect_qa.json"
        if effect_path.exists():
            item["category_effect_qa"] = read_json(effect_path)
        intervention_path = root / "diagnostics" / "intervention_metrics.json"
        if intervention_path.exists():
            item["intervention_metrics"] = read_json(intervention_path)
        qa_by_id[sid] = item

    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        rows_by_scene[row.get("scene_id", "")].append(row)
    plan_by_scene = {row["scene_id"]: row for row in plan if row.get("scene_id")}
    scene_ids = list(plan_by_scene) or sorted({x.get("scene_id", "") for x in manifest if x.get("scene_id")})
    scene_records = []
    for scene_id in scene_ids:
        planned = plan_by_scene.get(scene_id, {})
        rows = rows_by_scene.get(scene_id, [])
        qa = [qa_by_id.get(row.get("sample_id", "")) for row in rows]
        category_qa = [item.get("category_effect_qa") for item in qa if item]
        category_failed = any(item and item.get("status") == "fail" for item in category_qa)
        category_missing = len(category_qa) != 4 or any(item is None for item in category_qa)
        if category_failed:
            status = "paused_quality_failed"
        elif len(rows) == 4 and all(row.get("status") == "succeeded" for row in rows) and all(x and x.get("qa_status") == "pass" for x in qa) and not category_missing:
            status = "completed"
        elif str(planned.get("status", "")).startswith("paused") or any(str(x.get("status", "")).startswith("paused") for x in rows):
            status = "paused_asset_missing"
        elif any(row.get("status") in {"failed", "qa_failed", "blocked_storage", "aborted_download_budget"} for row in rows):
            status = "failed"
        else:
            status = "incomplete"
        scene_records.append({"scene_id": scene_id, "category": planned.get("category", "unknown"),
                              "status": status, "primary_name": planned.get("primary_name"),
                              "composition": planned.get("composition"),
                              "sample_ids": sorted(row.get("sample_id", "") for row in rows),
                              "errors": ([error for item in qa if item for error in item.get("errors", [])] +
                                         [error for item in category_qa if item for error in item.get("errors", [])])})

    dataset_counts: Counter[str] = Counter()
    source_appearance = "unspecified"
    selected_path = run / "selection" / "selected_assets.json"
    selected_data: dict[str, Any] = {}
    if selected_path.exists():
        selected_data = read_json(selected_path)
        selected = selected_data.get("selected", {})
        for record in selected.values():
            dataset_counts[record.get("dataset", "unknown")] += 1
        source_appearance = selected_data.get("source_appearance_policy", source_appearance)
    budget = _download_totals(run)
    budget["limit_bytes"] = int(experiment.get("total_download_budget_bytes", 100 * 1024**3))
    budget["percent_of_limit"] = (budget["downloaded_bytes"] / budget["limit_bytes"] * 100) if budget["limit_bytes"] else 0.0

    calibration = read_json(run / "calibration.json") if (run / "calibration.json").exists() else None
    noise_summary = read_json(run / "noise_summary.json") if (run / "noise_summary.json").exists() else None
    qa_counts = Counter(item.get("qa_status", "missing") for item in qa_by_id.values())
    scene_counts = Counter(item["status"] for item in scene_records)
    category_counts = Counter(item["category"] for item in scene_records)
    output_summary = {
        "experiment_id": experiment.get("experiment_id", run.name),
        "run": str(run),
        "planned_scenes": len(scene_records),
        "scene_status_counts": dict(scene_counts),
        "scene_category_counts": dict(category_counts),
        "expected_combinations": len(manifest),
        "combination_status_counts": dict(Counter(row.get("status", "missing") for row in manifest)),
        "sample_qa_counts": dict(qa_counts),
        "samples": len(qa_by_id),
        "passed": sum(item.get("qa_status") == "pass" for item in qa_by_id.values()),
        "failed": sum(item.get("qa_status") != "pass" for item in qa_by_id.values()),
        "paused_scenes": [item for item in scene_records if item["status"].startswith("paused")],
        "category_effect_qa_counts": dict(Counter(
            item.get("category_effect_qa", {}).get("status", "missing")
            for item in qa_by_id.values())),
        "asset_sources": dict(dataset_counts),
        "source_appearance_policy": source_appearance,
        "download_budget": budget,
        "asset_cache_bytes": directory_size(run / "assets") if (run / "assets").exists() else 0,
        "sample_output_bytes": directory_size(run / "samples") if (run / "samples").exists() else 0,
        "run_output_bytes": directory_size(run),
        "calibration": calibration,
        "noise_estimation": noise_summary,
        "scene_statuses": scene_records,
        "items": list(qa_by_id.values()),
    }
    write_json(run / "summary.json", output_summary)

    by_id = {row.get("sample_id"): row for row in manifest}
    category_sections = []
    ordered_categories = list(CATEGORY_ORDER) + sorted({x["category"] for x in scene_records} - set(CATEGORY_ORDER))
    for category in ordered_categories:
        scenes = [x for x in scene_records if x["category"] == category]
        if not scenes:
            continue
        cards = []
        for scene in scenes:
            sid = scene["scene_id"]
            four_up = run / "visuals" / "four_up" / f"{sid}.png"
            four_up_html = f"<img class='fourup' src='visuals/four_up/{escape(sid)}.png' alt='{escape(sid)} four-up'>" if four_up.exists() else ""
            samples = []
            for sample_id in sorted(scene["sample_ids"]):
                item = qa_by_id.get(sample_id)
                row = by_id.get(sample_id, {})
                if item:
                    image_root = f"samples/{escape(sample_id)}/joint"
                    noise = item.get("noise_statistics", {})
                    noise_text = json.dumps(noise.get("statistics", {}), ensure_ascii=False, indent=2) if noise else "未生成"
                    metrics = json.dumps(item.get("metrics", {}), ensure_ascii=False, indent=2)
                    effect_qa = item.get("category_effect_qa", {})
                    effect_errors = "\n".join(effect_qa.get("errors", [])) or "无"
                    errors = "\n".join(item.get("errors", [])) or "无"
                    samples.append(
                        f"<article class='sample'><h4>{escape(sample_id)} · 基础 QA {escape(item.get('qa_status','missing'))} · 效果 QA {escape(effect_qa.get('status','missing'))}</h4>"
                        f"<p>light={escape(str(row.get('light_id','?')))} · camera={escape(str(row.get('camera_id','?')))} · GPU={escape(str(item.get('gpu',{}).get('index','logged in metadata')))}</p>"
                        f"<img src='{image_root}/preview_checker.png' alt='checker preview'>"
                        f"<img src='{image_root}/preview_white.png' alt='white preview'>"
                        f"<img src='{image_root}/alpha_preview.png' alt='alpha preview'>"
                        f"<details><summary>基础 QA 指标与错误</summary><pre>{escape(metrics)}\n{escape(errors)}</pre></details>"
                        f"<details><summary>分类效果证据</summary><pre>{escape(json.dumps(effect_qa,ensure_ascii=False,indent=2))}\n{escape(effect_errors)}</pre></details>"
                        f"<details><summary>Second-seed noise estimate</summary><pre>{escape(noise_text)}</pre></details></article>"
                    )
                else:
                    samples.append(f"<article class='sample missing'><h4>{escape(sample_id)} · {escape(str(row.get('status','not scheduled')))}</h4><p>QA artifact missing</p></article>")
            errors = "; ".join(scene["errors"]) or "无"
            cards.append(
                f"<section class='scene' id='{escape(sid)}'><h3>{escape(sid)} · {escape(scene['status'])} · {escape(scene.get('primary_name') or '')}</h3>"
                f"<p>{escape(scene.get('composition') or '')}</p>{four_up_html}"
                f"<div class='samples'>{''.join(samples)}</div><p class='errors'>场景问题：{escape(errors)}</p></section>"
            )
        category_sections.append(
            f"<section class='category'><h2>{escape(CATEGORY_LABELS.get(category, category))} · {len(scenes)} scenes</h2>{''.join(cards)}</section>"
        )

    budget_text = (f"下载 {budget['downloaded_bytes']:,} / {budget['limit_bytes']:,} bytes "
                   f"({budget['percent_of_limit']:.3f}%) · reserved {budget['reserved_bytes']:,} bytes")
    cal_text = calibration.get("status", "missing") if calibration else "missing"
    noise_text = (f"{noise_summary.get('completed', 0)}/{noise_summary.get('submitted', 0)} samples"
                  if noise_summary else "missing")
    overview = "<img class='overview' src='visuals/scene_overview.png' alt='60 scene overview'>" if (run / "visuals" / "scene_overview.png").exists() else ""
    html = (
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>Exp60 RGBA 渲染报告</title>"
        "<style>body{font:14px system-ui;margin:1.5rem;background:#17191d;color:#eceff4}"
        ".category{margin:2rem 0}.scene{padding:1rem;margin:1rem 0;background:#242831;border-radius:8px}"
        ".samples{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:.7rem}"
        ".sample{padding:.7rem;background:#303641;border-radius:6px}.sample img{width:30%;background:#888;margin:.2rem}"
        ".fourup{width:min(720px,100%);display:block;background:#888}.overview{width:min(1000px,100%)}"
        "pre{white-space:pre-wrap;max-height:20rem;overflow:auto}details{margin:.4rem}.errors{color:#ffb4ab}</style>"
        f"<h1>{escape(output_summary['experiment_id'])} · RGBA 实验报告</h1>"
        f"<p>场景 {scene_counts.get('completed',0)}/{len(scene_records)} 全项完成；基础样本 QA {output_summary['passed']}/{output_summary['samples']} 通过；分类效果 QA：{escape(json.dumps(output_summary['category_effect_qa_counts'],ensure_ascii=False))}；校准 {escape(str(cal_text))}；噪声估计 {escape(noise_text)}。</p>"
        f"<p>{escape(budget_text)} · 源外观策略：{escape(source_appearance)} · 样本输出 {output_summary['sample_output_bytes']:,} B。</p>"
        f"<p>资产来源：{escape(json.dumps(dict(dataset_counts),ensure_ascii=False))}</p>{overview}"
        f"{''.join(category_sections)}</html>"
    )
    path = run / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def make_template_report(run: Path, experiment: dict[str, Any]) -> Path:
    """Display paired outputs without treating unrendered QA variants as passed."""
    from PIL import Image, ImageDraw, ImageFont
    cards = []
    records = []
    thumbnails = []
    for row in _jsonl(run/'acquisition_manifest.jsonl'):
        sample = run/'samples'/row['sample_id']
        if not (sample/'scene_recipe.json').exists():
            continue
        recipe = read_json(sample/'scene_recipe.json')
        qa = read_json(sample/'qa.json') if (sample/'qa.json').exists() else {'qa_status':'not_run'}
        effect_path = sample/'diagnostics/category_effect_qa.json'
        effect = read_json(effect_path) if effect_path.exists() else {'status': 'not_run' if recipe['background_protocol']['mode']=='core' else 'not_applicable'}
        files = [('Scene RGB','variants/scene_full/rgb.png'),
                 ('Target RGBA','joint/preview_checker.png'),
                 ('Without target','variants/scene_without_target/rgb.png')]
        figures = []
        for label, relative in files:
            path = sample/relative
            if path.exists():
                url = str(path.relative_to(run))
                figures.append(f'<figure><a href="{escape(url)}"><img src="{escape(url)}"></a><figcaption>{label}</figcaption></figure>')
        scene = recipe['scene_template']
        objects = ', '.join(obj.get('semantic_object',obj['id']) for obj in recipe['objects'])
        mode = recipe['background_protocol']['mode']
        sid = row['sample_id']
        cards.append(f'<article><h2>{escape(sid)} · {escape(scene["id"])}</h2>'
                     f'<p>{escape(objects)}</p><p>{escape(mode)} · {escape(scene["material_preset"]["id"])} · QA: {escape(qa["qa_status"])} · Effect QA: {escape(effect["status"])}</p>'
                     f'<p>{escape("; ".join(qa.get("errors", []) + effect.get("errors", [])))}</p>'
                     f'<div class="images">{"".join(figures)}</div></article>')
        records.append({'sample_id':sid,'scene_template':scene['id'],'mode':mode,'qa_status':qa['qa_status'],
                        'effect_qa_status':effect['status'],'material_preset':scene['material_preset']['id'],'render':recipe['render']})
        rgb = sample/'variants/scene_full/rgb.png'
        rgba = sample/'joint/preview_checker.png'
        if rgb.exists() and rgba.exists(): thumbnails.append((sid,rgb,rgba))
    summary = {'suite':experiment['suite'],'asset_count':experiment['asset_count'],
               'scene_template_count':len(experiment['scene_templates']),'rendered_samples':len(records),
               'qa_counts':dict(Counter(r['qa_status'] for r in records)),
               'effect_qa_counts':dict(Counter(r['effect_qa_status'] for r in records)),'samples':records}
    for name in ('calibration', 'noise_summary', 'effect_summary'):
        if (run/(name+'.json')).exists(): summary[name] = read_json(run/(name+'.json'))
    write_json(run/'report.json',summary)
    selection=read_json(run/'selection/selected_assets.json')['selected']
    sources=''.join(f'<li><a href="{escape(a["source_page"])}">{escape(a["name"])}</a> — {escape(a.get("author", ""))} · {escape(a["license"])}</li>' for a in selection.values())
    document=('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>Template PBR renders</title><style>'
              'body{margin:32px auto;padding:0 24px;max-width:1600px;background:#eeeae3;color:#252622;font:16px system-ui}'
              'h1{font-size:34px}article{background:#faf9f5;padding:20px;margin:28px 0;border-radius:8px}'
              '.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;display:block}figcaption{padding:8px 0}a{color:#245449}'
              '@media(max-width:800px){.images{grid-template-columns:1fr}}</style>'
              f'<h1>场景 RGB 与目标 RGBA</h1><p>{experiment["asset_count"]} 个 TexVerse PBR 物体 · {len(experiment["scene_templates"])} 个场景模板 · {len(records)} 个已渲染样本。</p>'
              f'<p>Export calibration: {escape(summary.get("calibration",{}).get("status","not_run"))} · '
              f'<a href="report.json">Quantitative results</a></p>'
              '<p>图像来自 Blender/Cycles。Physical 样本的 alpha 为渲染器 alpha；未执行的评测显示为 not_run。</p>'
              +''.join(cards)+f'<details><summary>物体来源与署名</summary><ul>{sources}</ul></details></html>')
    (run/'report.html').write_text(document,encoding='utf-8')
    if thumbnails:
        width,height=600,400
        sheet=Image.new('RGB',(width*2,(height+36)*len(thumbnails)), '#eeeae3')
        draw=ImageDraw.Draw(sheet)
        try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        except OSError: font=ImageFont.load_default()
        for index,(sid,rgb,rgba) in enumerate(thumbnails):
            y=index*(height+36)
            for column,(path,label) in enumerate([(rgb,'SCENE RGB'),(rgba,'TARGET RGBA')]):
                img=Image.open(path).convert('RGB'); img.thumbnail((width,height),Image.Resampling.LANCZOS)
                sheet.paste(img,(column*width+(width-img.width)//2,y+36+(height-img.height)//2))
                draw.text((column*width+12,y+8),f'{sid}  |  {label}',font=font,fill='#252622')
        sheet.save(run/'comparison.jpg',quality=95)
    return run/'report.html'


def make_isolation_report(run: Path) -> Path:
    """Gallery of separate single-object files; no target-group render is produced."""
    from PIL import Image, ImageDraw, ImageFont
    cards, images = [], []
    summary = read_json(run/'isolation_summary.json')
    for row in summary['objects']:
        folder = run/'samples'/row['sample_id']
        if not (folder/'scene_recipe.json').exists(): continue
        recipe = read_json(folder/'scene_recipe.json')
        obj = recipe['objects'][0]
        title = obj.get('semantic_object', obj['id'])
        relative = str(folder.relative_to(run))
        preview = folder/'preview_checker.png'
        if not preview.exists(): continue
        cards.append(f'<article><h2>{escape(title)}</h2>'
                     f'<p>{escape(recipe["identity"]["source_sample_id"])} / {escape(obj["id"])}</p>'
                     f'<a href="{escape(relative)}/straight_rgba.png"><img src="{escape(relative)}/preview_checker.png"></a>'
                     f'<p><a href="{escape(relative)}/straight_rgba.png">RGBA PNG</a> · '
                     f'<a href="{escape(relative)}/linear_premult.exr">Linear EXR</a> · '
                     f'<a href="{escape(relative)}/scene.blend">Blender scene</a> · '
                     f'<a href="{escape(relative)}/isolation.json">Render settings</a></p>'
                     f'<p>Status: {escape(row["status"])}</p></article>')
        images.append((recipe['identity']['source_sample_id']+' / '+obj['id'], title, preview))
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Single-object RGBA</title><style>body{max-width:1600px;margin:32px auto;padding:0 24px;background:#eeeae3;color:#252622;font:16px system-ui}'
                '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{padding:16px;background:#faf9f5;border-radius:8px}'
                'h2{font-size:20px}img{width:100%;display:block}a{color:#245449}</style>'
                f'<h1>单物体 RGBA · 中性白光</h1><p>{summary["completed"]} / {summary["object_count"]} 个独立物体输出。</p>'
                '<p>每次渲染只包含一个语义物体，并使用配方中记录的中性白光。新版默认全方向均匀白光；旧版配方保留三盏面积灯。原场景几何、其他物体、HDRI 与原场景灯光均不参与渲染。'
                '默认单独居中，保留原观察方向。棋盘格仅用于预览，下载的 PNG 背景透明。</p>'
                '<p>原生 PBR 材质与自身遮挡、反射仍保留。玻璃使用透明胶片，其 alpha 为当前照明下的渲染器 alpha。</p>'
                '<div class="grid">'+''.join(cards)+'</div></html>')
    (run/'report.html').write_text(document, encoding='utf-8')
    if images:
        tile, caption, columns = 384, 56, 3
        sheet = Image.new('RGB', (tile*columns, (tile+caption)*((len(images)+columns-1)//columns)), '#eeeae3')
        draw = ImageDraw.Draw(sheet)
        try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        except OSError: font = ImageFont.load_default()
        for index, (sid, title, path) in enumerate(images):
            x, y = (index % columns)*tile, (index // columns)*(tile+caption)
            preview = Image.open(path).convert('RGB')
            preview.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(preview, (x+(tile-preview.width)//2, y+caption+(tile-preview.height)//2))
            draw.text((x+10, y+7), sid, font=font, fill='#252622')
            draw.text((x+10, y+30), title[:38], font=font, fill='#52544e')
        sheet.save(run/'single_objects.jpg', quality=95)
    return run/'report.html'
