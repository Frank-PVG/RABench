"""Load editable source scenes while preserving authored assets."""
import fnmatch
import bpy
from render_paths import resolve_path
from pbr_materials import apply_material_preset

def load_template(spec: dict) -> list:
    path = resolve_path(spec['blend_file'])
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    bpy.context.scene.frame_set(int(spec.get('frame', 1)))
    patterns = spec.get('remove_objects', [])
    for obj in list(bpy.context.scene.objects):
        if any(fnmatch.fnmatchcase(obj.name, pattern) for pattern in patterns):
            bpy.data.objects.remove(obj, do_unlink=True)
    # The old downloadable scenes use relative external textures. Their native
    # directory stays available; no re-baking or flattening of material graphs.
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            image.filepath = bpy.path.abspath(image.filepath)
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    for obj in scene.objects:
        if obj.type == 'CAMERA':
            obj.data.dof.use_dof = False
    apply_material_preset(spec.get('material_preset', {}))
    environment = [obj for obj in bpy.data.objects if obj.type in {'MESH', 'CURVE', 'VOLUME'}]
    for obj in environment:
        obj['_rgba_original_camera_visibility'] = obj.visible_camera
    return environment
