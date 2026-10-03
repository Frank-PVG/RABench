"""Build one semantic object, including a multi-part functional assembly."""
import math
from pathlib import Path
import bpy
from mathutils import Vector
from geometry_support import apply_transform, geometry_points
from pbr_materials import apply_source_surface_overrides, setmat

def build_object(spec):
    if spec['role'] == 'assembly':
        root = bpy.data.objects.new('UNIT_'+spec['id'], None)
        bpy.context.collection.objects.link(root)
        objects = [root]
        for component in spec['components']:
            parts = build_object(component)
            for obj in parts:
                if obj.parent is None: obj.parent = root
                obj['semantic_component_id'] = component['id']
            objects.extend(parts)
        apply_transform(root, spec)
        if 'support_z' in spec:
            points = geometry_points(objects)
            root.location.z += float(spec['support_z'])-min(p.z for p in points)
        return objects
    if spec['role'] == 'asset': return asset(spec)
    builders = {'rounded_box':box, 'sphere':sphere, 'veil':veil, 'thin_film':thin_film,
                'cup':cup, 'liquid':liquid, 'straw':straw}
    return builders[spec['geometry']['type']](spec)

def asset(s):
    before=set(bpy.data.objects)
    source=Path(s['asset_path'])
    if not source.is_absolute(): source=Path(__file__).resolve().parents[1]/source
    suffix=source.suffix.lower()
    if suffix in {".glb", ".gltf"}: bpy.ops.import_scene.gltf(filepath=str(source))
    elif suffix == ".obj": bpy.ops.wm.obj_import(filepath=str(source))
    else: raise RuntimeError(f"unsupported asset format {suffix!r} for {s['id']}")
    imported=[o for o in bpy.data.objects if o not in before]; meshes=[o for o in imported if o.type=="MESH"]
    if not meshes: raise RuntimeError("asset import had no meshes")
    # Bake GLB node transforms into mesh data without replacing source materials.
    world_matrices=[o.matrix_world.copy() for o in meshes]
    world_vertices=[]
    for o,matrix in zip(meshes,world_matrices):
        world_vertices.append((o,matrix,[matrix @ v.co for v in o.data.vertices]))
    lo=Vector(tuple(min(v[i] for _,_,vs in world_vertices for v in vs) for i in range(3))); hi=Vector(tuple(max(v[i] for _,_,vs in world_vertices for v in vs) for i in range(3)))
    extent=max(hi-lo)
    if not math.isfinite(extent) or extent <= 1e-8: raise RuntimeError(f"asset {s['id']} has invalid bounds")
    normalization=s.get("normalization")
    if normalization:
        scale=float(normalization["scale"])
        offset=Vector(tuple(float(x) for x in normalization["translate"]))
        if not math.isfinite(scale) or scale <= 0 or not all(math.isfinite(x) for x in offset):
            raise RuntimeError(f"asset {s['id']} has invalid saved normalization")
    else:
        scale=s.get('size_m',.20)/extent; offset=Vector((-(lo.x+hi.x)/2, -(lo.y+hi.y)/2, -lo.z))
    for o,matrix,vs in world_vertices:
        # Imported node instances can share one mesh datablock; normalization is per instance.
        o.data=o.data.copy()
        o.parent=None; o.matrix_world.identity(); o.data.transform(matrix)
        for vertex,point in zip(o.data.vertices,vs): vertex.co=(point+offset)*scale
        o.data.update()
    for o in imported:
        if o.type in {"CAMERA","LIGHT"}: bpy.data.objects.remove(o,do_unlink=True)
    parent=bpy.data.objects.new("GROUP_"+s["id"],None); bpy.context.collection.objects.link(parent)
    for o in meshes:o.parent=parent
    x=s["transform"]; parent.location=x["location"]; parent.rotation_euler=x["rotation"]; parent.scale=x["scale"]
    if 'support_z' in s:
        bpy.context.view_layer.update()
        bottom=min((o.matrix_world@Vector(corner)).z for o in meshes for corner in o.bound_box)
        parent.location.z += float(s['support_z'])-bottom
    policy=s.get("material_policy","explicit_override")
    if policy=="explicit_override": setmat(meshes,s["material"],"MAT_"+s["id"])
    elif policy=="preserve_source":
        apply_source_surface_overrides(meshes,s.get("surface_overrides",{}),s["id"])
        # The original downloaded image maps stay attached and are packed into
        # the portable .blend deliverable without changing shader values.
        packed_names=set()
        for mesh in meshes:
            if mesh.type!="MESH": continue
            for source_material in mesh.data.materials:
                if not source_material or not source_material.use_nodes or not source_material.node_tree: continue
                for node in source_material.node_tree.nodes:
                    image=node.image if node.type=="TEX_IMAGE" else None
                    if not image or image.name in packed_names: continue
                    packed_names.add(image.name)
                    if image.source=="FILE" and not image.packed_file: image.pack()
    else: raise RuntimeError(f"unknown material policy {policy!r} for {s['id']}")
    return [parent,*meshes]


def box(s):
    bpy.ops.mesh.primitive_cube_add(location=s["transform"]["location"]); o=bpy.context.object; o.name=s["id"]; o.dimensions=s["geometry"]["size"]; o.rotation_euler=s["transform"]["rotation"]; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); b=o.modifiers.new("rounded_edges","BEVEL"); b.width=s["geometry"].get("bevel",0); b.segments=3; setmat([o],s["material"],"MAT_"+s["id"]); return[o]


def veil(s):
    # The veil is actual opaque woven microgeometry. Fractional alpha arises from subpixel coverage,
    # which preserves the Core alpha-over relation across independently rendered backplates.
    g=s["geometry"]; n=g["grid"]; verts=[]; faces=[]; width=g["width"]; height=g["height"]
    spacing=min(width,height)/n; strand=spacing*(1-math.sqrt(1-g["coverage"]))
    def fold(x,z): return g["fold_amplitude"]*math.sin(2.5*math.pi*x/width+g.get("fold_phase",0))*math.cos(math.pi*z/height)
    def quad(points):
        base=len(verts); verts.extend(points); faces.append((base,base+1,base+2,base+3))
    # Warp and weft strips provide the requested hanging, folded veil while leaving true holes.
    for i in range(n+1):
        x=-width/2+i*width/n; half=strand/2
        quad([(x-half,fold(x-half,-height/2),-height/2),(x+half,fold(x+half,-height/2),-height/2),(x+half,fold(x+half,height/2),height/2),(x-half,fold(x-half,height/2),height/2)])
    for i in range(n+1):
        z=-height/2+i*height/n; half=strand/2
        quad([(-width/2,fold(-width/2,z-half)+.00015,z-half),(width/2,fold(width/2,z-half)+.00015,z-half),(width/2,fold(width/2,z+half)+.00015,z+half),(-width/2,fold(-width/2,z+half)+.00015,z+half)])
    me=bpy.data.meshes.new(s["id"]); me.from_pydata(verts,[],faces); ob=bpy.data.objects.new(s["id"],me); bpy.context.collection.objects.link(ob); ob.location=s["transform"]["location"]; ob.rotation_euler=s["transform"]["rotation"]; setmat([ob],s["material"],"MAT_"+s["id"]); return[ob]


def sphere(s):
    g=s["geometry"]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=g.get("segments",48), ring_count=g.get("rings",24), radius=g["radius"], location=s["transform"]["location"])
    o=bpy.context.object; o.name=s["id"]; o.rotation_euler=s["transform"]["rotation"]
    for face in o.data.polygons: face.use_smooth=True
    setmat([o],s["material"],"MAT_"+s["id"])
    return[o]


def thin_film(s):
    g=s["geometry"]
    bpy.ops.mesh.primitive_cube_add(location=s["transform"]["location"])
    o=bpy.context.object; o.name=s["id"]; o.dimensions=(g["width"],g.get("thickness",0.0008),g["height"])
    o.rotation_euler=s["transform"]["rotation"]
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    setmat([o],s["material"],"MAT_"+s["id"])
    return[o]


def lathe(name,profile):
    seg=128; vs=[]; fs=[]; rings=[]
    for r,z in profile:
        if abs(r)<1e-9: rings.append([len(vs)]); vs.append((0,0,z))
        else:
            ring=[]
            for i in range(seg):
                ring.append(len(vs)); a=2*math.pi*i/seg; vs.append((r*math.cos(a),r*math.sin(a),z))
            rings.append(ring)
    for previous,current in zip(rings,rings[1:]):
        for i in range(seg):
            j=(i+1)%seg
            if len(previous)==1: fs.append((previous[0],current[j],current[i]))
            elif len(current)==1: fs.append((previous[i],previous[j],current[0]))
            else: fs.append((previous[i],previous[j],current[j],current[i]))
    me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);me.update()
    for face in me.polygons: face.use_smooth=abs(face.normal.z)<.5
    ob=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(ob);return ob


def cup(s):
    g=s["geometry"];r=g["outer_radius"];h=g["height"];w=g["wall"];b=g["bottom"];o=lathe(s["id"],[(0,0),(r,0),(r,h),(r-w,h),(r-w,b),(0,b)]);setmat([o],s["material"],"MAT_"+s["id"])
    bevel=o.modifiers.new('Glass edge bevel','BEVEL'); bevel.width=min(w*.2,.0006); bevel.segments=3
    apply_transform(o,s);return[o]


def liquid(s):
    g=s["geometry"];o=lathe(s["id"],[(0,0),(g["radius"],0),(g["radius"],g["height"]),(0,g["height"])]);setmat([o],s["material"],"MAT_"+s["id"]);apply_transform(o,s);return[o]


def straw(s):
    cu=bpy.data.curves.new(s["id"],"CURVE");cu.dimensions="3D";cu.bevel_depth=s["geometry"]["radius"];cu.bevel_resolution=4;sp=cu.splines.new("BEZIER");pts=s["geometry"]["points"];sp.bezier_points.add(len(pts)-1)
    for p,c in zip(sp.bezier_points,pts):p.co=c;p.handle_left_type="AUTO";p.handle_right_type="AUTO"
    o=bpy.data.objects.new(s["id"],cu);bpy.context.collection.objects.link(o);setmat([o],s["material"],"MAT_"+s["id"]);apply_transform(o,s);return[o]
