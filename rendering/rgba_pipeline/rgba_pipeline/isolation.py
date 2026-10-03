"""Render each semantic object independently under neutral white lighting."""
from __future__ import annotations

import copy
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json, recipe_reference, write_json
from .recipes import validate_recipe
from .render import run_variant
from .runner import eligible_gpus
from .validate import read_exr

OUTPUTS = ('linear_premult.exr', 'alpha.exr', 'straight_rgba.png', 'preview_checker.png',
           'preview_black.png', 'preview_white.png', 'alpha_preview.png',
           'scene.blend', 'isolation.json', 'render_log.json', 'qa.json')


def object_recipe(source: dict, object_id: str, *, resolution: int, samples: int,
                  seed: int, framing: str, lighting: dict | None = None) -> dict:
    if object_id not in source['sets']['TARGET']:
        raise ValueError(f'{object_id!r} is not in the source TARGET set')
    if resolution < 64 or samples < 1 or framing not in {'object', 'scene'}:
        raise ValueError('invalid isolation resolution, sample count or framing')
    obj = copy.deepcopy(next(obj for obj in source['objects'] if obj['id'] == object_id))
    camera = copy.deepcopy(source['camera'])
    if camera.get('use_template'):
        raise ValueError('single-object rendering needs an explicit source camera')
    source_size = source['render']['resolution']
    ratio = source_size[1] / source_size[0] if isinstance(source_size, list) else 1
    identity = source['identity']
    result = {
        'schema_version': '1.3', 'generator_version': 'isolated-object-1',
        'identity': {'sample_id': identity['sample_id'] + '__' + object_id,
                     'scene_root_id': identity['scene_root_id'], 'template_id': 'isolated_object',
                     'category': 'isolated_object', 'recipe_id': recipe_reference(source) + '.isolated.' + object_id,
                     'source_sample_id': identity['sample_id'], 'source_recipe_id': recipe_reference(source),
                     'asset_lineage_ids': [obj['lineage_id']] if obj.get('lineage_id') else []},
        'objects': [obj], 'sets': {'TARGET': [object_id], 'ENVIRONMENT': [], 'BACKPLATE': []},
        'relations': [], 'camera': camera,
        'environment': {'world_color': [0, 0, 0], 'lights': []},
        'background_protocol': {'mode': 'isolated', 'probes': []},
        'render': {'engine': 'CYCLES', 'resolution': [resolution, round(resolution * ratio) if framing == 'scene' else resolution],
                   'samples': samples, 'max_bounces': 16, 'transparent_bounces': 16,
                   'transmission_bounces': 12, 'clamp_direct': 0.0, 'clamp_indirect': 10.0,
                   'denoise': False, 'adaptive_sampling': False, 'caustics_reflective': True,
                   'caustics_refractive': True, 'view_transform': 'AgX', 'exposure': 0.0,
                   'color_space': 'scene_linear', 'alpha_encoding': 'linear'},
        'seeds': {'master': seed, 'render': seed, 'geometry': seed, 'material': seed},
        'isolation': {'protocol': 'neutral-white-object-v2', 'object_id': object_id,
                      'framing': framing, 'frame_fill': .8,
                      'transparent_glass': True, 'transparent_glass_roughness': .1,
                      'lighting_preset': 'uniform_white', 'world_color': [1,1,1], 'world_strength': 1.0,
                      'lights': []},
        'diagnostics': {'variants': ['isolated_object'], 'renderer_alpha_semantics': 'renderer_alpha_only'},
        'semantics': {'objects': [obj.get('semantic_object', object_id)], 'relations': [],
                      'materials': ['source PBR' if obj['role'] == 'asset' else 'optical assembly' if obj['role'] == 'assembly' else obj['material']['preset']],
                      'lighting': 'uniform neutral white illumination without directional falloff',
                      'expected_effects': ['self shading only'], 'target_group': [object_id]},
    }
    if lighting: result['isolation'].update(copy.deepcopy(lighting))
    return result


def validate_isolated_output(output: Path, recipe: dict) -> dict:
    errors = []
    for name in OUTPUTS:
        if name != 'qa.json' and not (output / name).is_file(): errors.append(f'missing {name}')
    metrics: dict[str, Any] = {}
    if not errors:
        rgba = read_exr(output / 'linear_premult.exr')
        alpha = read_exr(output / 'alpha.exr')[..., 0]
        expected = tuple(reversed(recipe['render']['resolution']))
        if rgba.shape != (*expected, 4) or alpha.shape != expected:
            errors.append('unexpected image dimensions')
        if not np.isfinite(rgba).all() or not np.isfinite(alpha).all(): errors.append('non-finite image values')
        if alpha.min() < -1e-5 or alpha.max() > 1.00001: errors.append('alpha outside [0,1]')
        if not (alpha > 1e-4).any(): errors.append('empty target render')
        yy, xx = np.nonzero(alpha > 1e-4)
        frame_span = max((xx.max()-xx.min()+1)/alpha.shape[1], (yy.max()-yy.min()+1)/alpha.shape[0]) if len(xx) else 0
        if recipe['isolation']['framing'] == 'object' and frame_span < .35:
            errors.append('object is unexpectedly small after automatic framing')
        audit = read_json(output / 'isolation.json')
        if audit['semantic_object_ids'] != [recipe['isolation']['object_id']]: errors.append('unexpected semantic object')
        if audit['source_environment_loaded'] or audit['other_objects_loaded']: errors.append('isolation violation')
        if audit['world_strength'] != recipe['isolation'].get('world_strength',0) or audit['world_texture_count'] != 0:
            errors.append('unexpected world illumination')
        if audit.get('world_color', [0,0,0]) != recipe['isolation'].get('world_color', [0,0,0]):
            errors.append('unexpected world color')
        expected_components = {part['id'] for part in recipe['objects'][0].get('components', [])}
        if expected_components and set(audit.get('component_ids', [])) != expected_components:
            errors.append('semantic assembly has missing or extra components')
        if any(light['color'] != [1, 1, 1] for light in audit['lights']): errors.append('non-neutral light')
        metrics = {'alpha_min': float(alpha.min()), 'alpha_max': float(alpha.max()),
                   'alpha_support_fraction': float((alpha > 1e-4).mean()),
                   'longest_projected_span_fraction': float(frame_span),
                   'semantic_object_count': len(audit['semantic_object_ids']),
                   'geometry_part_count': len(audit['rendered_geometry']), 'light_count': len(audit['lights'])}
    result = {'qa_status': 'pass' if not errors else 'fail', 'representation_class': 'isolated_object',
              'sample_id': recipe['identity']['sample_id'], 'errors': errors, 'metrics': metrics,
              'scope': 'single-object scene isolation and render-buffer integrity',
              'renderer_alpha_semantics': 'renderer_alpha_only'}
    write_json(output / 'qa.json', result)
    return result


def render_isolated_run(source_run: Path, output_run: Path, *, resolution: int = 1024,
                        samples: int = 256, seed: int = 42, framing: str = 'object',
                        max_workers: int = 2, resume: bool = True,
                        sample_ids: set[str] | None = None, object_ids: set[str] | None = None) -> dict:
    source_run, output_run = source_run.resolve(), output_run.resolve()
    if source_run == output_run:
        raise ValueError('use a separate output run for the single-object dataset')
    jobs = []
    matched_samples, matched_objects = set(), set()
    for path in sorted((source_run / 'recipes').glob('*.json')):
        source = read_json(path)
        sid = source['identity']['sample_id']
        if sample_ids is not None and sid not in sample_ids: continue
        matched_samples.add(sid)
        for oid in source['sets']['TARGET']:
            if object_ids is not None and oid not in object_ids: continue
            matched_objects.add(oid)
            if any(x in {'.', '..'} or '/' in x or '\\' in x for x in (sid, oid)):
                raise ValueError('sample/object IDs must be single path components')
            recipe = object_recipe(source, oid, resolution=resolution, samples=samples, seed=seed, framing=framing)
            errors = validate_recipe(recipe)
            if errors: raise ValueError(errors)
            jobs.append({'recipe': recipe, 'source_recipe': str(path)})
    if sample_ids and sample_ids - matched_samples: raise ValueError(f'unknown samples: {sample_ids - matched_samples}')
    if object_ids and object_ids - matched_objects: raise ValueError(f'unknown objects: {object_ids - matched_objects}')
    if not jobs: raise ValueError('no selected target objects')
    output_run.mkdir(parents=True, exist_ok=True)
    write_json(output_run / 'experiment.json', {'suite': 'isolated_objects_v1', 'source_run': str(source_run),
                                             'seed': seed, 'resolution': resolution, 'samples': samples,
                                             'framing': framing, 'object_count': len(jobs)})
    statuses, pending = [], []
    for job in jobs:
        recipe = job['recipe']; sid = recipe['identity']['sample_id']
        output = output_run / 'samples' / sid
        previous = output / 'scene_recipe.json'
        matching = previous.exists() and read_json(previous) == recipe
        if resume and matching and all((output / name).is_file() for name in OUTPUTS) and read_json(output / 'qa.json')['qa_status'] == 'pass':
            statuses.append({'sample_id': sid, 'status': 'resumed', 'source_recipe': job['source_recipe']})
            continue
        if previous.exists() and not matching:
            archive = output_run / 'archive' / (sid + '_' + uuid.uuid4().hex[:8])
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(archive))
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'scene_recipe.json', recipe)
        pending.append({**job, 'output': output})
    if pending:
        devices = eligible_gpus(max_workers)
        if not devices: raise RuntimeError('no idle GPU is available for single-object rendering')

        def worker(index: int) -> list[dict]:
            completed = []
            for job in pending[index::len(devices)]:
                output, recipe = job['output'], job['recipe']
                sid = recipe['identity']['sample_id']
                try:
                    run_variant(output / 'scene_recipe.json', output, 'isolated_object', gpu_uuid=devices[index]['uuid'])
                    qa = validate_isolated_output(output, recipe)
                    result = {'sample_id': sid, 'source_recipe': job['source_recipe'],
                              'status': 'succeeded' if qa['qa_status'] == 'pass' else 'failed', 'errors': qa['errors']}
                except Exception as exc:
                    result = {'sample_id': sid, 'source_recipe': job['source_recipe'], 'status': 'failed', 'error': str(exc)}
                completed.append(result)
                print('SUCCESS' if result['status'] == 'succeeded' else 'ERROR', sid, flush=True)
            return completed

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for records in pool.map(worker, range(len(devices))): statuses.extend(records)
    statuses.sort(key=lambda row: row['sample_id'])
    result = {'run': str(output_run), 'object_count': len(jobs),
              'completed': sum(row['status'] in {'succeeded', 'resumed'} for row in statuses),
              'failures': [row for row in statuses if row['status'] == 'failed'], 'objects': statuses}
    write_json(output_run / 'isolation_summary.json', result)
    from .report import make_isolation_report
    result['report'] = str(make_isolation_report(output_run))
    return result
