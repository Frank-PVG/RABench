"""Transforms, placement bounds and geometry constraints."""
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME'}

def geometry_points(objects):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        if obj.type not in GEOMETRY_TYPES: continue
        evaluated = obj.evaluated_get(depsgraph)
        if obj.type in {'CURVE','SURFACE','FONT','META'}:
            mesh = evaluated.to_mesh()
            points.extend(evaluated.matrix_world@vertex.co for vertex in mesh.vertices)
            evaluated.to_mesh_clear()
        else:
            points.extend(evaluated.matrix_world@Vector(p) for p in evaluated.bound_box)
    return points

def apply_transform(obj, spec: dict) -> None:
    transform = spec['transform']
    obj.location = transform['location']
    obj.rotation_euler = transform['rotation']
    obj.scale = transform['scale']


def slot_bounds(objects: list) -> dict:
    bpy.context.view_layer.update()
    points = [obj.matrix_world @ Vector(p) for obj in objects if obj.type == 'MESH' for p in obj.bound_box]
    return {'min': [min(p[i] for p in points) for i in range(3)],
            'max': [max(p[i] for p in points) for i in range(3)]}


def check_placement(recipe: dict, groups: dict) -> dict:
    """Check authored slots before rendering; physical containment is allowed.

    Bounds provide a conservative framing check. BVH tests detect intersecting
    surfaces between separate targets; declared contains-relations are exempt.
    Support contact is checked against the declared, measured scene height.
    """
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    objects = {item['id']: item for item in recipe['objects']}
    bounds, errors, frames, contacts = {}, [], {}, {}
    targets = recipe['sets']['TARGET']
    for oid in targets:
        meshes = [obj for obj in groups[oid] if obj.type == 'MESH']
        if not meshes:
            continue
        bounds[oid] = slot_bounds(meshes)
        projected = [world_to_camera_view(scene, scene.camera, obj.matrix_world @ Vector(p))
                     for obj in meshes for p in obj.bound_box]
        frames[oid] = {'min_xy': [min(p[i] for p in projected) for i in (0, 1)],
                       'max_xy': [max(p[i] for p in projected) for i in (0, 1)],
                       'in_front': all(p.z > scene.camera.data.clip_start for p in projected)}
        frame = frames[oid]
        if not frame['in_front'] or any(x < 0 for x in frame['min_xy']) or any(x > 1 for x in frame['max_xy']):
            errors.append(f'{oid}: target bounds cross the camera frame')
        if 'support_z' in objects[oid]:
            gap = bounds[oid]['min'][2] - objects[oid]['support_z']
            contacts[oid] = {'gap_m': gap, 'tolerance_m': .002}
            if abs(gap) > .002:
                errors.append(f'{oid}: target is not on its support plane')
    allowed = {frozenset((r['container'], r['content'])) for r in recipe.get('relations', [])
               if r.get('type') == 'contains'}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    trees = {}

    def tree(oid):
        if oid not in trees:
            vertices, faces = [], []
            for obj in groups[oid]:
                if obj.type != 'MESH': continue
                evaluated = obj.evaluated_get(depsgraph)
                mesh = evaluated.to_mesh()
                offset = len(vertices)
                vertices.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
                faces.extend(tuple(offset + i for i in polygon.vertices) for polygon in mesh.polygons)
                evaluated.to_mesh_clear()
            trees[oid] = BVHTree.FromPolygons(vertices, faces)
        return trees[oid]

    intersections = []
    for index, left in enumerate(bounds):
        for right in list(bounds)[index + 1:]:
            if frozenset((left, right)) in allowed: continue
            if not all(min(bounds[left]['max'][i], bounds[right]['max'][i]) >
                       max(bounds[left]['min'][i], bounds[right]['min'][i]) + 1e-5 for i in range(3)):
                continue
            if tree(left).overlap(tree(right)):
                intersections.append([left, right])
                errors.append(f'{left}/{right}: target surfaces intersect')
    return {'status': 'pass' if not errors else 'fail', 'errors': errors,
            'bounds': bounds, 'camera_framing': frames, 'support_contact': contacts,
            'intersecting_target_pairs': intersections,
            'scope': 'mesh target framing, declared support height, and target/target surface intersection'}
