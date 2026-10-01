"""Inspect one already-selected local GLB without rewriting geometry or materials."""
from __future__ import annotations
import hashlib, json, math, shlex, struct, sys, urllib.parse
from pathlib import Path
import bpy
from mathutils import Vector


def arg(name: str) -> str:
    args = sys.argv[sys.argv.index("--") + 1:]
    return args[args.index(name) + 1]


def gltf_document(path: Path):
    if path.suffix.lower() == ".gltf":
        document = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".glb":
        data = path.read_bytes()
        if len(data) < 20 or data[:4] != b"glTF":
            return None, []
        version, total = struct.unpack_from("<II", data, 4)
        if version != 2 or total != len(data):
            return None, []
        offset = 12; document = None
        while offset + 8 <= len(data):
            size, kind = struct.unpack_from("<II", data, offset); offset += 8
            chunk = data[offset:offset+size]; offset += size
            if kind == 0x4E4F534A:
                document = json.loads(chunk.rstrip(b" \0\t\r\n"))
                break
    else:
        return None, []
    deps = []
    if document:
        for field in ("buffers", "images"):
            for item in document.get(field, []):
                uri = item.get("uri")
                if uri and not uri.startswith("data:"):
                    target = (path.parent / urllib.parse.unquote(uri)).resolve()
                    deps.append({"kind": field[:-1], "uri": uri, "exists": target.is_file()})
    return document, deps


def obj_dependencies(path: Path):
    deps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = shlex.split(line.strip())
        if not parts:
            continue
        if parts[0].lower() != "mtllib":
            continue
        for name in parts[1:]:
            target = (path.parent / name).resolve()
            deps.append({"kind": "material_library", "uri": name, "exists": target.is_file()})
            if not target.is_file():
                continue
            for material_line in target.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = shlex.split(material_line.strip())
                if fields and fields[0].lower().startswith("map_") and len(fields) > 1:
                    uri = fields[-1]
                    image_path = (target.parent / uri).resolve()
                    deps.append({"kind": "texture", "uri": uri, "exists": image_path.is_file()})
    return deps


def components(mesh):
    n = len(mesh.vertices)
    parent = list(range(n)); rank = [0] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a=find(a); b=find(b)
        if a == b: return
        if rank[a] < rank[b]: a,b=b,a
        parent[b]=a
        if rank[a] == rank[b]: rank[a]+=1
    for edge in mesh.edges: union(edge.vertices[0], edge.vertices[1])
    return len({find(i) for i in range(n)}) if n else 0


def main():
    source=Path(arg("--source")).resolve(); output=Path(arg("--output")).resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before=set(bpy.data.objects)
    suffix=source.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(source))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source))
    else:
        raise RuntimeError(f"unsupported selected asset format: {suffix}")
    imported=[o for o in bpy.data.objects if o not in before]
    meshes=[o for o in imported if o.type=="MESH"]
    if not meshes: raise RuntimeError("no mesh object in selected GLB")
    world=[]; triangles=0; vertices=0; nonfinite=0; component_total=0
    components_by_mesh=[]
    for o in meshes:
        matrix=o.matrix_world.copy()
        local_components=components(o.data)
        components_by_mesh.append({"object":o.name,"components":local_components,
                                   "vertices":len(o.data.vertices),"polygons":len(o.data.polygons)})
        component_total+=local_components; vertices+=len(o.data.vertices)
        triangles+=sum(max(0,len(poly.vertices)-2) for poly in o.data.polygons)
        for v in o.data.vertices:
            p=matrix @ v.co
            if not all(math.isfinite(c) for c in p): nonfinite+=1
            else: world.append((float(p.x),float(p.y),float(p.z)))
    if not world: raise RuntimeError("selected GLB has no finite world-space vertices")
    lo=[min(p[i] for p in world) for i in range(3)]; hi=[max(p[i] for p in world) for i in range(3)]
    extent=max(hi[i]-lo[i] for i in range(3))
    if not math.isfinite(extent) or extent<=1e-8: raise RuntimeError("selected GLB has degenerate bounds")
    scale=.20/extent; translate=[-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]]
    center=[(lo[i]+hi[i])/2 for i in range(3)]
    distances=sorted(int(round(math.dist(p,center)/extent*10000)) for p in world)
    fingerprint=hashlib.sha256(json.dumps({"vertices":len(world),"triangles":triangles,"radial":distances},separators=(",",":")).encode()).hexdigest()
    materials=[]; images={}; image_nodes=0; linked_basecolor_maps=0
    for material in bpy.data.materials:
        used=any(material in o.data.materials[:] for o in meshes)
        if not used: continue
        record={"name":material.name,"node_material":bool(material.use_nodes),"textures":[]}
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type=="TEX_IMAGE" and node.image:
                    image_nodes+=1; image=node.image; images[image.name]=image
                    record["textures"].append({"image":image.name,"filepath":image.filepath,
                        "source":image.source,"packed":bool(image.packed_file),"has_data":bool(image.has_data)})
            for node in material.node_tree.nodes:
                if node.type=="BSDF_PRINCIPLED":
                    base=node.inputs.get("Base Color")
                    if base and base.is_linked and any(link.from_node.type=="TEX_IMAGE" for link in base.links):
                        linked_basecolor_maps+=1
        materials.append(record)
    if source.suffix.lower() in {".glb", ".gltf"}:
        document,external=gltf_document(source)
    else:
        document,external=None,obj_dependencies(source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    largest=max(hi[i]-lo[i] for i in range(3)); matrix=[[scale,0,0,scale*translate[0]],[0,scale,0,scale*translate[1]],[0,0,scale,scale*translate[2]],[0,0,0,1]]
    result={"source_path":str(source),"source_sha256":digest,"source_bytes":source.stat().st_size,
        "format":source.suffix.lower().lstrip("."),"glb_version":2 if source.suffix.lower()==".glb" and document else None,"mesh_count":len(meshes),
        "object_count":len(imported),"objects":[{"name":o.name,"type":o.type,"parent":o.parent.name if o.parent else None} for o in imported],
        "vertex_count":vertices,"triangle_count":triangles,"nonfinite_vertex_count":nonfinite,
        "component_count":component_total,"components_by_mesh":components_by_mesh,
        "world_bounds":{"min":lo,"max":hi,"extent":[hi[i]-lo[i] for i in range(3)]},
        "normalization":{"scale":scale,"translate":translate,"matrix_4x4":matrix,
                         "longest_edge_m":.2,"xy_center_m":[0,0],"lowest_z_m":0},
        "geometry_fingerprint":fingerprint,"material_count":len(materials),"image_count":len(images),
        "image_texture_node_count":image_nodes,"base_color_image_links":linked_basecolor_maps,
        "materials":materials,"external_dependencies":external,
        "qa":{"has_mesh":bool(meshes),"finite_geometry":nonfinite==0,"has_materials":bool(materials),
              "source_texture_nodes":image_nodes,"missing_dependency_count":sum(not x.get("exists",True) for x in external),
              "triangle_limit_ok":triangles<=1_000_000}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("RGBA_ASSET_INSPECT:",json.dumps({"uid":source.stem,"triangles":triangles,"components":component_total,
        "textures":len(images),"external_dependencies":len(external),"fingerprint":fingerprint},ensure_ascii=False),flush=True)

main()
