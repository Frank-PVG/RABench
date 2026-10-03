"""Check the final saved template geometry without rerendering images."""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from template_support import check_placement

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
failed = []
for scene in sorted((args.run / 'samples').glob('*/scene.blend')):
    recipe = json.loads((scene.parent / 'scene_recipe.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(scene), load_ui=False, use_scripts=False)
    groups = {oid: [obj for obj in bpy.context.scene.objects if obj.get('semantic_object_id') == oid]
              for oid in recipe['sets']['TARGET']}
    result = check_placement(recipe, groups)
    (scene.parent / 'joint/placement_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['errors']: failed.append(scene.parent.name)
    print('PLACEMENT', scene.parent.name, result['status'], result['errors'], flush=True)
if failed:
    raise RuntimeError(f'Placement failed for {failed}')
