"""Small Luna pilot using the same client/config interface as AlphaLift's curator."""
from __future__ import annotations
import base64
import importlib.util
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps

from .io import read_json, write_json
from .paths import CONFIG_ROOT

CHECKS = ('geometry_integrity','alpha_quality','clean_isolation','neutral_lighting','pair_correspondence','material_quality')


def image_inputs(run, row):
    files = [run/row['condition_image'], run/row['target_rgba']]
    return [{'path':str(path.relative_to(run)), 'bytes':path.stat().st_size,
             'mtime_ns':path.stat().st_mtime_ns} for path in files]


def save_dataset(run, rows):
    """Attach cached results only to the exact image files they describe."""
    annotated = []
    annotation_fields = ('short_prompt','detailed_prompt','extraction_prompt',
                         'annotation_model','image_quality_decision')
    for row in rows:
        for key in annotation_fields: row.pop(key, None)
        row['luna_status'] = 'not_run'
        path = run/'luna'/row['id']/'result.json'
        if path.exists():
            item = read_json(path)
            if item.get('inputs') == image_inputs(run, row):
                row['luna_status'] = item['status']
                if item.get('image_quality'):
                    row['image_quality_decision'] = item['image_quality']['decision']
                if item['status'] == 'annotated' and row['qa_status'] == 'pass':
                    row.update(item['annotation'])
                    row['annotation_model'] = item['model']
                    annotated.append(row)
    (run/'dataset.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    (run/'annotated.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in annotated))


def validate_quality(value):
    if set(value) != {'decision','reason','checks'} or value['decision'] not in {'keep','discard'}:
        raise ValueError('invalid image-quality response fields')
    if not isinstance(value['reason'],str) or not value['reason'].strip() or set(value['checks']) != set(CHECKS):
        raise ValueError('missing image-quality evidence')
    for check in value['checks'].values():
        if set(check) != {'status','evidence'} or check['status'] not in {'pass','fail','uncertain'}:
            raise ValueError('invalid image-quality check')
        if not isinstance(check['evidence'],str) or not check['evidence'].strip(): raise ValueError('missing visual evidence')
    if (value['decision']=='keep') != all(check['status']=='pass' for check in value['checks'].values()):
        raise ValueError('quality decision disagrees with checks')


def validate_annotation(value):
    if set(value) != {'short_prompt','detailed_prompt','extraction_prompt'}:
        raise ValueError('invalid annotation JSON fields')
    if any(not isinstance(text,str) or not text.strip() for text in value.values()):
        raise ValueError('annotation fields must be nonempty strings')


def load_client(client_module, teacher_config, model_id, reasoning_effort, max_tokens):
    spec=importlib.util.spec_from_file_location('_rabench_alphalift_vlm',client_module)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    config=replace(module.VLMClientConfig.from_yaml(teacher_config),model_id=model_id,
                   reasoning_effort=reasoning_effort,max_tokens=max_tokens,temperature=0.0)
    return module.VLMClient(config)


def diagnostic_views(run,row,max_side=1024):
    folder=run/'samples'/row['id']/'target'
    with Image.open(run/row['condition_image']) as image: source=image.convert('RGB')
    with Image.open(folder/'preview_checker.png') as image: checker=image.convert('RGB')
    with Image.open(folder/'alpha_preview.png') as image: alpha=image.convert('L')
    bbox=alpha.getbbox() or (0,0,alpha.width,alpha.height)
    x0,y0,x1,y1=bbox
    box=(max(0,x0-8),max(0,y0-8),min(alpha.width,x1+8),min(alpha.height,y1+8))
    panels=[]
    for name in ('preview_black.png','preview_white.png'):
        with Image.open(folder/name) as image:
            panels.append(ImageOps.contain(image.convert('RGB').crop(box),(max_side//2,max_side//2)))
    strip=Image.new('RGB',(panels[0].width+panels[1].width,max(p.height for p in panels)))
    strip.paste(panels[0],(0,0));strip.paste(panels[1],(panels[0].width,0))
    return [ImageOps.contain(image,(max_side,max_side),Image.Resampling.LANCZOS) for image in (source,checker,strip,alpha)]


def data_url(image):
    buffer=io.BytesIO();image.save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def annotate_pilot(run, *, client_module, teacher_config, model_id='gpt-5.6-luna', reasoning_effort='high',
                   max_tokens=4096, workers=3, limit=3, sample_ids=None, resume=True):
    run=Path(run).resolve()
    if limit < 1 or workers < 1: raise ValueError('Luna limit and workers must be positive')
    rows=[json.loads(line) for line in (run/'dataset.jsonl').read_text().splitlines() if line.strip()]
    selected_ids=set(sample_ids) if sample_ids is not None else set(read_json(run/'experiment.json')['luna_pilot'][:limit])
    selected=[row for row in rows if row['id'] in selected_ids and row['qa_status']=='pass'][:limit]
    if not selected: raise ValueError('no fully rendered selected samples for Luna')
    if selected_ids-set(row['id'] for row in selected): raise ValueError('selected Luna samples are missing, failed, or exceed the limit')
    prompts={name:(CONFIG_ROOT/'p1/prompts'/filename).read_text() for name,filename in
             [('image_quality','image_quality.txt'),('annotation','annotate.txt')]}
    client=load_client(client_module,teacher_config,model_id,reasoning_effort,max_tokens)
    protocol={'version':'luna-image-qa-caption-v1','client':client.config.safe_summary,'max_tokens':max_tokens,
              'stages':['image_quality','annotation'],'prompts':prompts,'rewrite_stage':False,
              'quality_scope':'images only; annotation content is not scored'}
    write_json(run/'luna/protocol.json',protocol)
    def work(row):
        output=run/'luna'/row['id'];path=output/'result.json'
        inputs=image_inputs(run, row)
        if resume and path.exists():
            previous=read_json(path)
            if previous.get('inputs')==inputs and previous.get('protocol')==protocol and previous.get('status') in {'annotated','discarded'}:
                print('SUCCESS Luna resumed',row['id'],flush=True);return previous
        started=time.monotonic()
        result={'id':row['id'],'inputs':inputs,'protocol':protocol,'model':model_id,'status':'error'}
        try:
            views=diagnostic_views(run,row)
            preview_dir=output/'input_views';preview_dir.mkdir(parents=True,exist_ok=True)
            for name,image in zip(('scene','target_checker','target_black_white','alpha'),views): image.save(preview_dir/(name+'.png'))
            urls=[data_url(image) for image in views]
            quality=client.complete_json(system_prompt=prompts['image_quality'],
                user_prompt='Inspect the final rendered images and return image-quality JSON only.',
                image_data_urls=urls,request_retries=2,json_retries=1,response_validator=validate_quality)
            write_json(output/'image_quality.json',quality)
            result['image_quality']=quality
            if quality['decision']=='keep':
                annotation=client.complete_json(system_prompt=prompts['annotation'],
                    user_prompt='Generate the three English annotations from the images.',
                    image_data_urls=urls,request_retries=2,json_retries=1,response_validator=validate_annotation)
                write_json(output/'annotation.json',annotation)
                result.update(status='annotated',annotation=annotation)
            else:
                result.update(status='discarded',annotation=None)
        except Exception as exc:
            # The reference client deliberately redacts transport errors.
            result['error']=str(exc) if type(exc).__name__ in {'VLMRequestError','ValueError'} else type(exc).__name__
        result['elapsed_seconds']=round(time.monotonic()-started,3)
        write_json(path,result)
        print('ERROR' if result['status']=='error' else 'SUCCESS', 'Luna',row['id'],result['status'],flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool: results=list(pool.map(work,selected))
    save_dataset(run, rows)
    summary={'selected':len(results),'annotated':sum(row['status']=='annotated' for row in results),
             'discarded':sum(row['status']=='discarded' for row in results),
             'errors':[{'id':row['id'],'error':row.get('error')} for row in results if row['status']=='error'],
             'model':model_id,'quality_scope':'image quality only','rewrite_stage':False}
    write_json(run/'luna_summary.json',summary)
    return summary
