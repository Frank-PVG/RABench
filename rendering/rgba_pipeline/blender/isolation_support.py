"""Isolate one semantic unit, which may contain multiple functional parts."""
import bpy
from mathutils import Vector
from geometry_support import GEOMETRY_TYPES, geometry_points
from camera_support import fit_camera, camera_record
from lighting_support import neutral_world, neutral_lights


def empty_scene(spec=None):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    neutral_world(spec or {})


def configure_isolation(recipe, objects):
    scene = bpy.context.scene
    spec = recipe['isolation']
    oid = spec['object_id']
    if recipe['sets']['TARGET'] != [oid] or len(recipe['objects']) != 1:
        raise ValueError('isolation requires exactly one semantic unit')
    points = geometry_points(objects)
    if not points: raise ValueError(f'{oid}: no renderable geometry')
    lower = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    upper = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center, size = (lower+upper)/2, max(upper-lower)
    if size <= 1e-8: raise ValueError(f'{oid}: degenerate geometry bounds')
    camera = scene.camera
    if spec['framing'] == 'object':
        fit_camera(camera, points, center, size, float(spec['frame_fill']))
    elif spec['framing'] != 'scene':
        raise ValueError(f"unknown isolation framing {spec['framing']!r}")
    lights = neutral_lights(spec, center, size, camera)
    scene.cycles.film_transparent_glass = bool(spec['transparent_glass'])
    scene.cycles.film_transparent_roughness = float(spec['transparent_glass_roughness'])
    bpy.context.view_layer.update()
    geometry = [obj for obj in scene.objects if obj.type in GEOMETRY_TYPES]
    foreign = [obj.name for obj in geometry if obj.get('semantic_object_id') != oid]
    instances = [obj.name for obj in scene.objects if obj.instance_type != 'NONE']
    if foreign or instances: raise RuntimeError(f'non-target geometry: {foreign+instances}')
    if sum(obj.type == 'LIGHT' for obj in scene.objects) != len(spec.get('lights', [])):
        raise RuntimeError('unexpected source-scene light')
    world = next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND')
    return {'protocol': spec['protocol'], 'lighting_preset': spec.get('lighting_preset', 'legacy_three_point'),
            'object_id': oid, 'semantic_object_ids': [oid],
            'component_ids': sorted({obj.get('semantic_component_id') for obj in geometry if obj.get('semantic_component_id')}),
            'rendered_geometry': [obj.name for obj in geometry],
            'source_environment_loaded': False, 'other_objects_loaded': False,
            'world_color': list(world.inputs['Color'].default_value[:3]),
            'world_strength': float(world.inputs['Strength'].default_value),
            'world_texture_count': sum(n.type in {'TEX_ENVIRONMENT','TEX_IMAGE'} for n in scene.world.node_tree.nodes),
            'lights': lights, 'framing': spec['framing'], 'bounds': {'min':list(lower),'max':list(upper)},
            'camera': camera_record(camera), 'alpha_semantics': 'renderer_alpha_only',
            'transparent_glass': spec['transparent_glass']}
