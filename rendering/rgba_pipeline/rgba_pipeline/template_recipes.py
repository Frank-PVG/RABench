"""Small, explicit template suite; evaluation categories stay independent of sets."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from .assets import prepared_manifest
from .io import write_json
from .paths import PROJECT_ROOT, RUNS_ROOT
from .recipes import PROBES, validate_recipe


def build_recipe(sample: dict, template: dict, assets: dict, *, resolution: int, samples: int, seed: int) -> dict:
    objects = []
    for binding in sample.get('assets', []):
        asset = assets[binding['asset_id']]
        slot = template['slots'][binding['slot']]
        size = float(binding.get('size_m', asset['size_m']))
        if size > slot['max_size_m']:
            raise ValueError(f"{sample['id']}: {asset['asset_id']} exceeds the slot's allowed size")
        objects.append({
            'id': binding['id'], 'role': 'asset', 'asset_id': asset['asset_id'],
            'asset_path': str(Path(asset['local_path']).relative_to(PROJECT_ROOT)),
            'source_uid': asset['uid'], 'lineage_id': asset['lineage_id'],
            'source_dataset': asset['dataset'], 'source_url': asset['source_page'], 'source_license': asset['license'],
            'material_policy': 'preserve_source', 'surface_overrides': {}, 'size_m': size,
            'support_z': binding.get('location', slot['position'])[2],
            'transform': {'location': binding.get('location', slot['position']),
                          'rotation': binding.get('rotation', asset.get('rotation') or [0,0,0]), 'scale': [1,1,1]},
            'semantic_object': asset['name'],
        })
    objects.extend(copy.deepcopy(sample.get('procedural_objects', [])))
    target = [obj['id'] for obj in objects]
    scene = {key: copy.deepcopy(template[key]) for key in ('id', 'blend_file', 'frame', 'remove_objects')}
    scene['material_preset'] = copy.deepcopy(template['material_presets'][sample.get('material_preset','original')])
    mode = sample['mode']
    recipe = {
        'schema_version': '1.2', 'generator_version': 'template-pbr-1',
        'identity': {'sample_id': sample['id'], 'scene_root_id': sample['id'], 'recipe_id': uuid.uuid4().hex,
                     'template_id': sample['category'], 'category': sample['category'],
                     'camera_id': 'C1', 'light_id': 'L1', 'split': 'template_development',
                     'asset_lineage_ids': [obj['lineage_id'] for obj in objects if obj.get('lineage_id')]},
        'scene_template': scene, 'sets': {'TARGET': target, 'ENVIRONMENT': ['__scene_environment__'], 'BACKPLATE': []},
        'objects': objects, 'relations': copy.deepcopy(sample.get('relations', [])),
        'camera': copy.deepcopy(sample.get('camera', template['camera'])),
        'environment': {'world_color': [.035,.035,.035], 'lights': copy.deepcopy(template['lights'])},
        'background_protocol': {'mode': mode, 'probes': PROBES.copy()},
        'render': {'engine': 'CYCLES', 'resolution': [resolution, round(resolution*2/3)], 'samples': samples,
                   'max_bounces': 16, 'transparent_bounces': 16, 'transmission_bounces': 12,
                   'clamp_direct': 0.0, 'clamp_indirect': 10.0, 'denoise': False, 'adaptive_sampling': False,
                   'caustics_reflective': True, 'caustics_refractive': True, 'view_transform': 'AgX',
                   'exposure': sample.get('exposure', template.get('exposure',.3)), 'color_space': 'scene_linear', 'alpha_encoding': 'linear'},
        'seeds': {'master': seed, 'render': seed, 'geometry': seed, 'material': seed},
        'diagnostics': {'variants': ['joint','independent','intervention','geometry_debug'],
                        'independent_order': target.copy(), 'intervention': copy.deepcopy(sample['intervention']),
                        'renderer_alpha_semantics': 'renderer_alpha_only' if mode=='physical' else 'core_effective_alpha'},
        'semantics': {'objects': [obj['semantic_object'] for obj in objects], 'relations': sample.get('relations',[]),
                      'materials': ['source PBR'], 'lighting': 'authored scene lighting and explicit fill',
                      'expected_effects': [sample['category']], 'target_group': target},
    }
    recipe['render'].update(copy.deepcopy(sample.get('render_overrides', {})))
    errors = validate_recipe(recipe)
    if errors:
        raise ValueError(f"{sample['id']}: {errors}")
    return recipe


def generate_template_suite(run_name: str, *, resolution: int=1024, samples: int=256, seed: int=42) -> Path:
    if resolution < 64 or samples < 1:
        raise ValueError('resolution must be >=64 and samples positive')
    configuration = json.loads((PROJECT_ROOT/'configs/template_pbr_v1.json').read_text())
    assets = {record['asset_id']: record for record in prepared_manifest()}
    run = RUNS_ROOT/run_name
    (run/'recipes').mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in configuration['samples']:
        template = configuration['templates'][sample['template']]
        if not (PROJECT_ROOT/template['blend_file']).is_file():
            raise FileNotFoundError(f"download scene template first: {template['blend_file']}")
        recipe = build_recipe(sample, template, assets, resolution=resolution, samples=samples, seed=seed)
        recipe_path = run/'recipes'/(sample['id']+'.json')
        if recipe_path.exists():
            previous = json.loads(recipe_path.read_text())
            current = copy.deepcopy(recipe); old = copy.deepcopy(previous)
            current['identity'].pop('recipe_id',None); old['identity'].pop('recipe_id',None)
            if current==old: recipe['identity']['recipe_id']=previous['identity']['recipe_id']
        write_json(recipe_path,recipe)
        manifest.append({'suite':'template_pbr_v1','sample_id':sample['id'],'scene_id':sample['id'],
                         'category':sample['category'],'recipe':str(recipe_path),'recipe_id':recipe['identity']['recipe_id'],'status':'pending'})
    path=run/'acquisition_manifest.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in manifest))
    write_json(run/'experiment.json',{'suite':'template_pbr_v1','schema_version':'1.2','seed':seed,
                                    'scene_templates':list(configuration['templates']), 'samples':len(manifest),
                                    'asset_count':len(assets),'resolution':resolution,'render_samples':samples})
    write_json(run/'selection/selected_assets.json',{'selected':assets,'source_appearance_policy':'preserve original source meshes, UVs and PBR textures'})
    write_json(run/'template_configuration.json',configuration)
    return path
