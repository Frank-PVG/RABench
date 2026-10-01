from __future__ import annotations
import json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector
def arg(n):
    a=sys.argv[sys.argv.index("--")+1:];return a[a.index(n)+1]
source=Path(arg("--source")).resolve();output=Path(arg("--output")).resolve();metadata=Path(arg("--metadata")).resolve()
bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False);bpy.ops.import_scene.gltf(filepath=str(source));meshes=[x for x in bpy.context.scene.objects if x.type=="MESH"]
if not meshes:raise RuntimeError("No mesh")
for x in bpy.context.scene.objects:
    if x.type in {"CAMERA","LIGHT"}:bpy.data.objects.remove(x,do_unlink=True)
bpy.ops.object.select_all(action="DESELECT")
for x in meshes:x.select_set(True)
# Joining keeps the active object's object-space transform.  Bake *all* of it
# before measuring and normalising; otherwise the exported GLB retains its
# source translation and appears many metres away when re-imported.
bpy.context.view_layer.objects.active=meshes[0];bpy.ops.object.join();obj=bpy.context.object;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True);vs=[obj.matrix_world@v.co for v in obj.data.vertices]
if not vs or any(not math.isfinite(q) for p in vs for q in p):raise RuntimeError("Invalid vertices")
lo=Vector(tuple(min(p[i] for p in vs) for i in range(3)));hi=Vector(tuple(max(p[i] for p in vs) for i in range(3)));span=hi-lo
if max(span)<=0:raise RuntimeError("Degenerate bounds")
scale=.2/max(span);offset=Vector((-(lo.x+hi.x)/2,-(lo.y+hi.y)/2,-lo.z))
for v in obj.data.vertices:v.co=(v.co+offset)*scale
tri=sum(len(p.vertices)-2 for p in obj.data.polygons)
if tri>200000:raise RuntimeError(f"Triangle limit exceeded: {tri}")
obj.name="normalized_asset";output.parent.mkdir(parents=True,exist_ok=True);bpy.ops.object.select_all(action="DESELECT");obj.select_set(True);bpy.context.view_layer.objects.active=obj;bpy.ops.export_scene.gltf(filepath=str(output),export_format="GLB",use_selection=True,export_materials="EXPORT")
metadata.write_text(json.dumps({"source":str(source),"prepared_path":str(output),"bounds_before":{"min":list(lo),"max":list(hi)},"normalization":{"scale":scale,"translate":list(offset)},"triangles":tri,"components":[obj.name],"preprocess_version":"2"},indent=2))
