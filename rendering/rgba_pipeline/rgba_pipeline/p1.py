"""Generate and render a fixed, small paired RGB / neutral RGBA dataset."""
from __future__ import annotations
import copy
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import prepared_manifest
from .io import read_json, write_json
from .isolation import OUTPUTS as ISOLATED_OUTPUTS, object_recipe, validate_isolated_output
from .paths import PROJECT_ROOT, RUNS_ROOT
from .presets import PresetBook
from .recipes import validate_recipe
from .render import run_variant
from .runner import eligible_gpus
from .validate import read_exr


def generate_p1(run_name, *, samples=None, resolution=None, seed=42):
    book = PresetBook()
    assets = {row['asset_id']:row for row in prepared_manifest()}
    run = RUNS_ROOT/run_name
    (run/'recipes').mkdir(parents=True,exist_ok=True)
    rows = []
    for case in book.documents['suite']['cases']:
        sid = case['id']; scene = book.get('scenes',case['scene'])
        export = book.get('export',case['export'])
        if samples is not None: export['samples'] = samples
        if resolution is not None: export['resolution'] = [resolution,round(resolution*3/4)]
        obj,size,height = book.unit(case['object'],case['scene'],case['object_material'],assets)
        camera = book.camera(case['scene'],case['camera'],size,height)
        scene_spec = {key:scene[key] for key in ('blend_file','frame','remove_objects')}
        scene_spec.update({'id':case['scene'],'material_preset':copy.deepcopy(book.documents['materials']['scene_surfaces'][case['scene_material']])})
        if not (PROJECT_ROOT/scene_spec['blend_file']).is_file(): raise FileNotFoundError(scene_spec['blend_file'])
        lighting = book.get('lighting',case['context_lighting'])
        if not lighting.get('retain_source_world', True) or not lighting.get('retain_source_lights', True):
            raise ValueError('P1 context lighting must retain the authored scene; edit target lighting separately')
        fill = copy.deepcopy(scene['fill_lights'])
        for light in fill: light['energy'] *= lighting['fill_multiplier']
        recipe = {'schema_version':'1.2','generator_version':'p1-v1',
                  'identity':{'sample_id':sid,'scene_root_id':sid,'recipe_id':uuid.uuid4().hex,
                              'template_id':'single_semantic_unit','category':'single_semantic_unit',
                              'camera_id':case['camera'],'light_id':case['context_lighting'],
                              'asset_lineage_ids':[obj['lineage_id']] if obj.get('lineage_id') else []},
                  'scene_template':scene_spec,'presets':case,
                  'sets':{'TARGET':['target'],'ENVIRONMENT':['__scene_environment__'],'BACKPLATE':[]},
                  'objects':[obj],'relations':[],'camera':camera,
                  'environment':{'world_color':[.035]*3,'lights':fill},
                  'background_protocol':{'mode':'physical' if obj['role']=='assembly' else 'core','probes':[]},
                  'render':{**export,'engine':'CYCLES','exposure':scene['exposure'],
                            'adaptive_sampling':False,'caustics_reflective':True,'caustics_refractive':True},
                  'seeds':{'master':seed,'render':seed,'geometry':seed,'material':seed},
                  'diagnostics':{'variants':['scene_full']},
                  'semantics':{'objects':[obj['semantic_object']],'relations':[],
                               'materials':['source PBR' if obj['role']=='asset' else 'glass/liquid/straw assembly'],
                               'lighting':'authored context','target_group':['target'],'expected_effects':[]}}
        path = run/'recipes'/(sid+'.json')
        if path.exists():
            previous = read_json(path); current = copy.deepcopy(recipe); prior = copy.deepcopy(previous)
            current['identity'].pop('recipe_id'); prior['identity'].pop('recipe_id')
            if current == prior: recipe['identity']['recipe_id'] = previous['identity']['recipe_id']
        target = object_recipe(recipe,'target',resolution=export['resolution'][0],samples=export['samples'],seed=seed,
                               framing='scene',lighting=book.get('lighting',case['target_lighting']))
        target['identity']['sample_id'] = sid
        for key in ('samples','view_transform','denoise','clamp_indirect','max_bounces','transmission_bounces','transparent_bounces'):
            target['render'][key] = export[key]
        target['render']['resolution'] = export['resolution']
        target['render']['exposure'] = export['target_exposure']
        target['presets'] = case
        for value in (recipe,target):
            errors = validate_recipe(value)
            if errors: raise ValueError(f'{sid}: {errors}')
        write_json(path,recipe)
        target_path = run/'recipes'/(sid+'.target.json')
        write_json(target_path,target)
        rows.append({'id':sid,'recipe':str(path),'target_recipe':str(target_path),'presets':case,
                     'semantic_object':obj['semantic_object'],'component_count':len(obj.get('components',[])) or 1,
                     'asset_lineage_ids':recipe['identity']['asset_lineage_ids']})
    (run/'render_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write_json(run/'experiment.json',{'suite':'p1_pairs_v1','count':len(rows),'seed':seed,
                                    'luna_pilot':book.documents['suite']['luna_pilot'],
                                    'contract':'same camera/geometry; target relit under uniform white illumination'})
    shutil.copytree(book.root,run/'preset_snapshot',dirs_exist_ok=True)
    for folder in ('blender','rgba_pipeline'):
        target_dir=run/'source_snapshot'/folder;target_dir.mkdir(parents=True,exist_ok=True)
        for path in (PROJECT_ROOT/folder).glob('*.py'): shutil.copy2(path,target_dir/path.name)
    shutil.copy2(PROJECT_ROOT/'scripts/run_p1.sh',run/'source_snapshot/run_p1.sh')
    return run


def validate_pair(sample, source_recipe, target_recipe):
    qa = validate_isolated_output(sample/'target',target_recipe)
    errors = list(qa['errors'])
    camera = read_json(sample/'condition/camera.json')
    isolated = read_json(sample/'target/isolation.json')
    delta = float(np.max(np.abs(np.asarray(camera['matrix_world'])-np.asarray(isolated['camera']['matrix_world']))))
    if delta > 1e-6 or abs(camera['lens_mm']-isolated['camera']['lens_mm']) > 1e-6:
        errors.append('source and target cameras differ')
    with Image.open(sample/'condition/rgb.png') as image:
        if list(image.size) != target_recipe['render']['resolution']: errors.append('source/target image dimensions differ')
    alpha=read_exr(sample/'target/alpha.exr')[...,0]
    border=np.concatenate([alpha[0],alpha[-1],alpha[:,0],alpha[:,-1]])
    if (border>.01).any(): errors.append('target touches the frame boundary')
    result={'sample_id':source_recipe['identity']['sample_id'],'qa_status':'pass' if not errors else 'fail',
            'errors':errors,'camera_matrix_max_error':delta,'target':qa,
            'scope':'pair registration, framing, semantic assembly, isolation and buffer integrity'}
    write_json(sample/'qa.json',result)
    return result


def render_p1(run, *, max_workers=2, sample_ids=None, resume=True):
    run=Path(run).resolve()
    rows=[json.loads(line) for line in (run/'render_manifest.jsonl').read_text().splitlines() if line.strip()]
    selected=[row for row in rows if sample_ids is None or row['id'] in sample_ids]
    if not selected: raise ValueError('no P1 samples selected')
    if sample_ids and sample_ids-set(row['id'] for row in selected):
        raise ValueError('unknown P1 sample IDs: '+', '.join(sorted(sample_ids-set(row['id'] for row in selected))))
    devices=eligible_gpus(max_workers)
    if not devices: raise RuntimeError('no idle GPU is available')
    def worker(index):
        results=[]
        for row in selected[index::len(devices)]:
            sid=row['id'];sample=run/'samples'/sid
            source=read_json(Path(row['recipe']));target=read_json(Path(row['target_recipe']))
            sample.mkdir(parents=True,exist_ok=True)
            identity_path=sample/'recipes.json'
            identities={'source':source,'target':target}
            if identity_path.exists() and read_json(identity_path)!=identities:
                archive=run/'archive'/(sid+'_'+uuid.uuid4().hex[:8]);archive.parent.mkdir(exist_ok=True)
                shutil.move(str(sample),str(archive));sample.mkdir()
            write_json(identity_path,identities)
            try:
                for variant,path,folder,required in [
                    ('scene_full',Path(row['recipe']),sample/'condition',('rgb.png','linear_premult.exr','camera.json','render_log.json')),
                    ('isolated_object',Path(row['target_recipe']),sample/'target',tuple(name for name in ISOLATED_OUTPUTS if name != 'qa.json'))]:
                    if not (resume and all((folder/name).is_file() for name in required)):
                        run_variant(path,folder,variant,gpu_uuid=devices[index]['uuid'])
                qa=validate_pair(sample,source,target)
                result={'id':sid,'status':'succeeded' if qa['qa_status']=='pass' else 'failed','errors':qa['errors']}
            except Exception as exc:
                result={'id':sid,'status':'failed','error':str(exc)}
            results.append(result)
            print('SUCCESS' if result['status']=='succeeded' else 'ERROR',sid,flush=True)
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for group in pool.map(worker,range(len(devices))): results.extend(group)
    write_json(run/'last_render_batch.json',{'results':results,'failures':[r for r in results if r['status']=='failed']})
    records=[]
    for row in rows:
        sample=run/'samples'/row['id'];qa_path=sample/'qa.json'
        if not qa_path.exists(): continue
        qa=read_json(qa_path)
        record={**row,'condition_image':f'samples/{row["id"]}/condition/rgb.png',
                'target_rgba':f'samples/{row["id"]}/target/straight_rgba.png',
                'target_linear_exr':f'samples/{row["id"]}/target/linear_premult.exr',
                'target_alpha_exr':f'samples/{row["id"]}/target/alpha.exr',
                'qa_status':qa['qa_status'],'target_lighting_normalized':True,'luna_status':'not_run'}
        records.append(record)
    from .luna import save_dataset
    save_dataset(run, records)
    result={'rendered':len(records),'passed':sum(r['qa_status']=='pass' for r in records),
            'failures':[r for r in results if r['status']=='failed']}
    write_json(run/'render_summary.json',result)
    return result
