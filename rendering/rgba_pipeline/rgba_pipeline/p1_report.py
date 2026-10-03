"""Inspectable paired-image gallery for the fixed P1 dataset and Luna pilot."""
from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .io import read_json, write_json


def contact_sheet(run: Path, rows: list[dict], destination: str):
    if not rows:
        return
    width, height, gap, label = 280, 210, 12, 48
    cell_width, cell_height = width*2+gap, height+label
    sheet = Image.new('RGB', (cell_width*2+gap*3, ((len(rows)+1)//2)*(cell_height+gap)+gap), '#eeeae3')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = small = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = gap+(index % 2)*(cell_width+gap)
        y = gap+(index // 2)*(cell_height+gap)
        draw.text((x, y+2), row['id'], font=font, fill='#252622')
        draw.text((x, y+26), 'SCENE RGB', font=small, fill='#52544e')
        draw.text((x+width+gap, y+26), 'NEUTRAL RGBA / SAME CAMERA', font=small, fill='#52544e')
        for column, path in enumerate((run/row['condition_image'], run/'samples'/row['id']/'target/preview_checker.png')):
            with Image.open(path) as source:
                image = source.convert('RGB')
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
                sheet.paste(image, (x+column*(width+gap)+(width-image.width)//2, y+label+(height-image.height)//2))
    sheet.save(run/destination, quality=95)


def make_p1_report(run: Path, experiment: dict) -> Path:
    rows = [json.loads(line) for line in (run/'dataset.jsonl').read_text().splitlines() if line.strip()]
    cards = []
    for row in rows:
        sid = row['id']
        sample = f'samples/{sid}'
        quality = {}
        result_path = run/'luna'/sid/'result.json'
        if result_path.exists() and row.get('luna_status') != 'not_run':
            quality = read_json(result_path).get('image_quality', {})
        annotation = ''.join(f'<dt>{escape(key)}</dt><dd>{escape(row[key])}</dd>'
                             for key in ('short_prompt','detailed_prompt','extraction_prompt') if key in row)
        figures = []
        for label, preview, download in (
            ('场景 RGB', row['condition_image'], row['condition_image']),
            ('均匀白光 RGBA · 相同位置与尺度', f'{sample}/target/preview_checker.png', row['target_rgba']),
        ):
            figures.append(f'<figure><a href="{escape(download)}"><img loading="lazy" src="{escape(preview)}" alt="{escape(sid+" "+label)}"></a>'
                           f'<figcaption>{label}</figcaption></figure>')
        presets = row['presets']
        caption = ('<dl>'+annotation+'</dl>') if annotation else '<p class="muted">此样本未生成 Luna 标注。</p>'
        quality_details = (f'<details><summary>Luna 图像质检依据 · {escape(quality["decision"])}</summary>'
                           f'<pre>{escape(json.dumps(quality,ensure_ascii=False,indent=2))}</pre></details>') if quality else ''
        cards.append(f'<article id="{escape(sid)}"><h2>{escape(sid)} · {escape(row["semantic_object"])}</h2>'
                     f'<p>{escape(presets["scene"])} / {escape(presets["scene_material"])} / {escape(presets["camera"])}'
                     f' · 1 个语义物体，{row["component_count"]} 个注册组件</p>'
                     f'<p>配对与文件检查：{escape(row["qa_status"])} · Luna：{escape(row.get("luna_status","not_run"))}</p>'
                     f'<div class="images">{"".join(figures)}</div>'
                     f'<p><a href="{escape(row["target_rgba"])}">RGBA PNG</a> · '
                     f'<a href="{escape(row["target_linear_exr"])}">Linear EXR</a> · '
                     f'<a href="{sample}/target/alpha_preview.png">Alpha</a> · '
                     f'<a href="recipes/{sid}.json">场景配方</a> · '
                     f'<a href="recipes/{sid}.target.json">物体配方</a> · '
                     f'<a href="{sample}/qa.json">配对检查</a></p>{caption}{quality_details}</article>')
    summary = {
        'suite': experiment['suite'], 'planned': experiment['count'], 'rendered': len(rows),
        'semantic_units': len({row['presets']['object'] for row in rows}),
        'scene_templates': sorted({row['presets']['scene'] for row in rows}),
        'pair_qa_counts': dict(Counter(row['qa_status'] for row in rows)),
        'luna_status_counts': dict(Counter(row.get('luna_status','not_run') for row in rows)),
        'luna_image_quality_counts': dict(Counter(row.get('image_quality_decision','not_run') for row in rows)),
        'lighting': 'constant neutral white world, strength 1, no directional lamps or original environment',
        'pair_contract': 'identical camera, geometry, canvas coordinates and image dimensions; target relit',
        'luna_contract': 'image quality first, then three direct English annotations for kept samples; no rewrite or text-quality grading',
        'samples': rows,
    }
    write_json(run/'report.json', summary)
    counts = summary['luna_status_counts']
    document = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>P1 · 场景 RGB 与透明物体</title><style>'
        'body{max-width:1440px;margin:36px auto;padding:0 24px;background:#eeeae3;color:#252622;font:16px/1.5 system-ui}'
        'h1{font-size:32px}h2{font-size:22px}article{background:#faf9f5;padding:20px;margin:28px 0;border-radius:8px}'
        '.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{display:block;width:100%}'
        'figcaption{padding:8px 0}.muted{color:#69716d}a{color:#245449}dt{font-weight:600;margin-top:12px}'
        'dd{margin:4px 0;max-width:100ch}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}'
        'details{margin:16px 0}@media(max-width:750px){.images{grid-template-columns:1fr}body{padding:0 12px}h1{font-size:27px}}</style>'
        f'<h1>P1 · 场景 RGB 与透明物体</h1><p>{len(rows)} / {experiment["count"]} 组配对 · '
        f'{len(summary["scene_templates"])} 个场景 · {summary["semantic_units"]} 种语义物体。</p>'
        f'<p>配对与文件检查通过 {summary["pair_qa_counts"].get("pass",0)} 组；'
        f'Luna 已标注 {counts.get("annotated",0)} 组、图像质检淘汰 {counts.get("discarded",0)} 组、'
        f'未调用 {counts.get("not_run",0)} 组、调用错误 {counts.get("error",0)} 组。</p>'
        '<p>每组使用相同相机、画布位置和尺度。RGBA 使用全方向均匀白光；玻璃杯、液体与吸管整体保留。'
        '棋盘格仅为预览，PNG 背景透明。纹理、自身遮挡和材质反射仍保留。</p>'
        '<p>Luna 先检查图像质量，仅为保留样本生成三种英文描述或指令；没有独立重写，也不对标注内容打分。'
        '透明材质的 alpha 是当前照明下的渲染器 alpha。</p>'
        f'<p><a href="dataset.jsonl">全部数据</a> · <a href="annotated.jsonl">{counts.get("annotated",0)} 条试标数据</a> · '
        '<a href="pairs_overview.jpg">全部配对预览</a> · <a href="report.json">检查结果</a> · '
        '<a href="preset_snapshot/suite.json">固定预设清单</a></p>'
        +''.join(cards)+'</html>'
    )
    (run/'report.html').write_text(document, encoding='utf-8')
    contact_sheet(run, rows, 'pairs_overview.jpg')
    preview_ids = {'P01_boot_a','P03_camera_a','P09_orange_a','P16_moka_b','P19_latte_a','P24_drink_b'}
    contact_sheet(run, [row for row in rows if row['id'] in preview_ids], 'pairs_preview.jpg')
    return run/'report.html'
