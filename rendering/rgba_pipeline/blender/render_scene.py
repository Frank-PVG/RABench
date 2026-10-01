"""Blender-side renderer, called by `python -m rgba_pipeline render`."""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path
import bpy
from mathutils import Vector

def arg(name):
    args = sys.argv[sys.argv.index("--") + 1:]
    return args[args.index(name) + 1]

RECIPE_PATH = Path(arg("--recipe")).resolve(); OUT = Path(arg("--output")).resolve(); VARIANT = arg("--variant"); PROBE = os.environ.get("RGBA_PROBE", "")
recipe = json.loads(RECIPE_PATH.read_text()); OUT.mkdir(parents=True, exist_ok=True); semantic = {}
def log(m): print("RGBA_PIPELINE:", m, flush=True)
def look(obj, at): obj.rotation_euler = (Vector(at)-obj.location).to_track_quat("-Z", "Y").to_euler()

def gpu(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences; errs=[]
    for backend in ("OPTIX","CUDA"):
        try:
            prefs.compute_device_type=backend
            if hasattr(prefs,"get_devices"): prefs.get_devices()
            devices=list(prefs.devices); choices=[x for x in devices if x.type==backend]
            want=os.environ.get("RGBA_CYCLES_DEVICE_UUID", "")
            visible=os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if want and visible != want:
                raise RuntimeError("GPU worker UUID does not match CUDA_VISIBLE_DEVICES")
            if not choices: raise RuntimeError("no requested GPU UUID is available: "+want)
            # The scheduler masks the process to exactly one physical UUID.
            if want and len(choices) != 1:
                raise RuntimeError(f"GPU worker expected one visible card, found {len(choices)}")
            for x in devices: x.use=False
            choices[0].use=True; scene.cycles.device="GPU"
            return {"backend":backend,"name":choices[0].name,"type":choices[0].type,"uuid":want or None}
        except Exception as e: errs.append(f"{backend}: {e}")
    raise RuntimeError("Cycles GPU unavailable: "+" | ".join(errs))

def setup():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; cfg=recipe["render"]
    scene.render.engine="CYCLES"; scene.render.resolution_x=cfg["resolution"]; scene.render.resolution_y=cfg["resolution"]; scene.render.resolution_percentage=100
    scene.render.film_transparent=True; scene.cycles.samples=cfg["samples"]; scene.cycles.seed=recipe["seeds"]["render"]; scene.cycles.use_denoising=False; scene.cycles.use_adaptive_sampling=False
    scene.cycles.max_bounces=cfg["max_bounces"]; scene.cycles.transparent_max_bounces=cfg["transparent_bounces"]; scene.cycles.transmission_bounces=cfg["transmission_bounces"]; scene.cycles.sample_clamp_direct=cfg.get("clamp_direct",0); scene.cycles.sample_clamp_indirect=cfg.get("clamp_indirect",0)
    if hasattr(scene.render,"filter_type"): scene.render.filter_type="BOX"
    scene.render.image_settings.color_mode="RGBA"; scene.render.image_settings.color_depth="16"; scene.view_settings.view_transform="Standard"; scene.view_settings.look="None"
    scene.world.use_nodes=True; bg=scene.world.node_tree.nodes.get("Background"); bg.inputs["Color"].default_value=(*recipe["environment"]["world_color"],1); bg.inputs["Strength"].default_value=.25
    device=gpu(scene); c=recipe["camera"]
    bpy.ops.object.camera_add(location=c["location"]); cam=bpy.context.object; cam.name="Camera"; cam.data.lens=c["lens_mm"]; cam.data.sensor_width=c["sensor_width_mm"]; cam.data.clip_start=c["clip_start"]; cam.data.clip_end=c["clip_end"]; look(cam,c["look_at"]); scene.camera=cam
    for item in recipe["environment"]["lights"]:
        bpy.ops.object.light_add(type=item["type"],location=item["location"]); light=bpy.context.object; light.name="ENV_"+item["id"]; light.data.energy=item["energy"]; light.data.shape="DISK"; light.data.size=item["size"]; look(light,c["look_at"])
    return scene,device

def material(name,spec):
    mat=bpy.data.materials.new(name); mat.use_nodes=True; t=mat.node_tree; t.nodes.clear(); out=t.nodes.new("ShaderNodeOutputMaterial"); p=spec["preset"]
    if p=="opaque_principled":
        n=t.nodes.new("ShaderNodeBsdfPrincipled"); n.inputs["Base Color"].default_value=spec.get("base_color",[.8,.8,.8,1]); n.inputs["Roughness"].default_value=spec.get("roughness",.5); n.inputs["Metallic"].default_value=spec.get("metallic",0.0); t.links.new(n.outputs[0],out.inputs["Surface"])
    elif p=="neutral_thin_coverage":
        mix=t.nodes.new("ShaderNodeMixShader"); mix.inputs[0].default_value=spec["coverage"]; a=t.nodes.new("ShaderNodeBsdfTransparent"); b=t.nodes.new("ShaderNodeBsdfDiffuse"); b.inputs["Color"].default_value=spec.get("base_color",[.78,.78,.78,1]); t.links.new(a.outputs[0],mix.inputs[1]); t.links.new(b.outputs[0],mix.inputs[2]); t.links.new(mix.outputs[0],out.inputs["Surface"])
    elif p=="glass":
        n=t.nodes.new("ShaderNodeBsdfGlass"); n.inputs["IOR"].default_value=spec["ior"]; n.inputs["Roughness"].default_value=spec.get("roughness",0); t.links.new(n.outputs[0],out.inputs["Surface"])
    elif p=="liquid":
        n=t.nodes.new("ShaderNodeBsdfGlass"); n.inputs["IOR"].default_value=spec["ior"]; t.links.new(n.outputs[0],out.inputs["Surface"]); v=t.nodes.new("ShaderNodeVolumeAbsorption"); v.inputs["Color"].default_value=(*spec["absorption_color"],1); v.inputs["Density"].default_value=spec["absorption_density"]; t.links.new(v.outputs[0],out.inputs["Volume"])
    else: raise RuntimeError("unknown material "+p)
    return mat
def apply_source_surface_overrides(meshes,overrides,asset_id):
    # Change only requested shader channels while retaining downloaded image maps.
    for mesh in meshes:
        for material in mesh.data.materials:
            if not material or not material.use_nodes: continue
            for shader in (n for n in material.node_tree.nodes if n.type=="BSDF_PRINCIPLED"):
                if "metallic" in overrides: shader.inputs["Metallic"].default_value=float(overrides["metallic"])
                if "roughness" in overrides: shader.inputs["Roughness"].default_value=float(overrides["roughness"])
                if "base_color_tint" in overrides:
                    socket=shader.inputs["Base Color"]
                    links=list(socket.links)
                    if links:
                        source=links[0].from_socket
                        for link in list(socket.links): material.node_tree.links.remove(link)
                        gray=material.node_tree.nodes.new("ShaderNodeRGBToBW")
                        mix=material.node_tree.nodes.new("ShaderNodeMixRGB")
                        mix.blend_type="MULTIPLY"; mix.inputs[0].default_value=1.0
                        mix.inputs[2].default_value=(*overrides["base_color_tint"][:3],1.0)
                        material.node_tree.links.new(source,gray.inputs[0])
                        material.node_tree.links.new(gray.outputs[0],mix.inputs[1])
                        material.node_tree.links.new(mix.outputs[0],socket)
                    else:
                        socket.default_value=(*overrides["base_color_tint"][:3],1.0)


def setmat(objs,spec,name):
    m=material(name,spec)
    for o in objs:
        if o.type in {"MESH","CURVE"}: o.data.materials.clear(); o.data.materials.append(m)
def tag(oid,objs): semantic[oid]=objs; [o.__setitem__("semantic_object_id",oid) for o in objs]
def asset(s):
    before=set(bpy.data.objects)
    suffix=Path(s["asset_path"]).suffix.lower()
    if suffix in {".glb", ".gltf"}: bpy.ops.import_scene.gltf(filepath=s["asset_path"])
    elif suffix == ".obj": bpy.ops.wm.obj_import(filepath=s["asset_path"])
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
        scale=.20/extent; offset=Vector((-(lo.x+hi.x)/2, -(lo.y+hi.y)/2, -lo.z))
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
    seg=96; vs=[]; fs=[]
    for r,z in profile:
        for i in range(seg): a=2*math.pi*i/seg;vs.append((r*math.cos(a),r*math.sin(a),z))
    for j in range(len(profile)-1):
        for i in range(seg): a=j*seg+i;b=j*seg+(i+1)%seg;fs.append((a,b,b+seg,a+seg))
    me=bpy.data.meshes.new(name);me.from_pydata(vs,[],fs);ob=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(ob);return ob
def cup(s):
    g=s["geometry"];r=g["outer_radius"];h=g["height"];w=g["wall"];b=g["bottom"];o=lathe(s["id"],[(0,0),(r,0),(r,h),(r-w,h),(r-w,b),(0,b)]);setmat([o],s["material"],"MAT_"+s["id"]);return[o]
def liquid(s):
    g=s["geometry"];o=lathe(s["id"],[(0,0),(g["radius"],0),(g["radius"],g["height"]),(0,g["height"])]);o.location=s["transform"]["location"];setmat([o],s["material"],"MAT_"+s["id"]);return[o]
def straw(s):
    cu=bpy.data.curves.new(s["id"],"CURVE");cu.dimensions="3D";cu.bevel_depth=s["geometry"]["radius"];cu.bevel_resolution=4;sp=cu.splines.new("BEZIER");pts=s["geometry"]["points"];sp.bezier_points.add(len(pts)-1)
    for p,c in zip(sp.bezier_points,pts):p.co=c;p.handle_left_type="AUTO";p.handle_right_type="AUTO"
    o=bpy.data.objects.new(s["id"],cu);bpy.context.collection.objects.link(o);setmat([o],s["material"],"MAT_"+s["id"]);return[o]
def build():
    builders={"rounded_box":box,"sphere":sphere,"veil":veil,"thin_film":thin_film,"cup":cup,"liquid":liquid,"straw":straw}
    for s in recipe["objects"]:
        k=s.get("geometry",{}).get("type");objs=asset(s) if s["role"]=="asset" else builders[k](s);tag(s["id"],objs)
    for oid in recipe["sets"].get("ENVIRONMENT",[]):
        if oid not in semantic: raise RuntimeError(f"environment set references unknown object {oid}")
        for o in semantic[oid]:
            if o.type in {"MESH","CURVE"}: o.visible_camera=False
def visible(oid,on):
    for o in semantic[oid]:o.hide_render=not on

def apply_intervention_recipe():
    cfg=recipe.get("diagnostics",{}).get("intervention",{})
    kind=cfg.get("kind")
    if kind in {"translate", "color", "neutral_color", "coverage"}:
        oid=cfg["object"]
        spec=next(x for x in recipe["objects"] if x["id"]==oid)
        if kind=="translate":
            offset=cfg["offset"]
            spec["transform"]["location"]=[a+b for a,b in zip(spec["transform"]["location"],offset)]
        elif kind in {"color", "neutral_color"}:
            color=cfg["value"]
            if spec["role"]=="asset" and spec.get("material_policy")=="preserve_source":
                spec.setdefault("surface_overrides",{})["base_color_tint"]=color[:3]
            else:
                spec["material"]["base_color"]=color
        elif kind=="coverage":
            if spec.get("geometry",{}).get("type")=="veil":
                spec["geometry"]["coverage"]=cfg["value"]
            elif spec.get("material",{}).get("preset")=="neutral_thin_coverage":
                spec["material"]["coverage"]=cfg["value"]
            else:
                raise RuntimeError(f"coverage intervention is unsupported for {oid}")
    elif kind=="remove_environment":
        cfg["hide_objects"]=list(recipe["sets"].get("ENVIRONMENT",[]))
    else:
        raise RuntimeError(f"unsupported categorical intervention: {kind}")

def background(kind,mode):
    sc=bpy.context.scene;cam=sc.camera;f=cam.matrix_world.to_quaternion()@Vector((0,0,-1));bpy.ops.mesh.primitive_plane_add(size=20,location=cam.location+f*3);p=bpy.context.object;p.name="BACKPLATE_"+kind;p.rotation_euler=cam.rotation_euler
    mat=bpy.data.materials.new("BG_"+kind);mat.use_nodes=True;t=mat.node_tree;t.nodes.clear();out=t.nodes.new("ShaderNodeOutputMaterial");colors={"black":(0,0,0,1),"white":(1,1,1,1),"red":(1,0,0,1),"blue":(0,0,1,1)}
    shader=t.nodes.new("ShaderNodeEmission" if mode=="core" else "ShaderNodeBsdfDiffuse")
    if kind in colors: shader.inputs["Color"].default_value=colors[kind]
    else:
        coord=t.nodes.new("ShaderNodeTexCoord")
        if kind.startswith("checker"):
            tex=t.nodes.new("ShaderNodeTexChecker")
            tex.inputs["Scale"].default_value=32 if kind=="checker_fine" else 8
            tex.inputs["Color1"].default_value=(.08,.18,.85,1)
            tex.inputs["Color2"].default_value=(.93,.75,.08,1)
            t.links.new(coord.outputs["Generated"],tex.inputs["Vector"])
            t.links.new(tex.outputs["Color"],shader.inputs["Color"])
        elif kind == "noise":
            tex=t.nodes.new("ShaderNodeTexNoise");tex.inputs["Scale"].default_value=12;tex.inputs["Detail"].default_value=2
            ramp=t.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color=(.08,.18,.85,1);ramp.color_ramp.elements[1].color=(.93,.75,.08,1)
            t.links.new(coord.outputs["Generated"],tex.inputs["Vector"]);t.links.new(tex.outputs["Fac"],ramp.inputs["Fac"])
            t.links.new(ramp.outputs["Color"],shader.inputs["Color"])
        elif kind == "stripes":
            tex=t.nodes.new("ShaderNodeTexWave");tex.wave_type="BANDS";tex.bands_direction="X"
            tex.inputs["Scale"].default_value=9;tex.inputs["Distortion"].default_value=.15
            ramp=t.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color=(.08,.18,.85,1);ramp.color_ramp.elements[1].color=(.93,.75,.08,1)
            t.links.new(coord.outputs["Generated"],tex.inputs["Vector"]);t.links.new(tex.outputs["Fac"],ramp.inputs["Fac"])
            t.links.new(ramp.outputs["Color"],shader.inputs["Color"])
        else: raise RuntimeError("unsupported background probe "+kind)
    if mode=="core":
        shader.inputs["Strength"].default_value=1;mix=t.nodes.new("ShaderNodeMixShader");tr=t.nodes.new("ShaderNodeBsdfTransparent");lp=t.nodes.new("ShaderNodeLightPath");t.links.new(lp.outputs["Is Camera Ray"],mix.inputs[0]);t.links.new(tr.outputs[0],mix.inputs[1]);t.links.new(shader.outputs[0],mix.inputs[2]);t.links.new(mix.outputs[0],out.inputs["Surface"]);p.visible_diffuse=False;p.visible_glossy=False;p.visible_transmission=False;p.visible_volume_scatter=False;p.visible_shadow=False
    else:t.links.new(shader.outputs[0],out.inputs["Surface"])
    p.data.materials.append(mat);return p

def save(out):
    out.mkdir(parents=True,exist_ok=True);sc=bpy.context.scene;res=bpy.data.images["Render Result"]
    # Blender 5.2 does not expose Render Result pixels. Round-trip the exact render buffer through a temporary linear EXR.
    raw=out/"_render_straight.exr";sc.render.image_settings.file_format="OPEN_EXR";sc.render.image_settings.color_mode="RGBA";sc.render.image_settings.color_depth="32";res.save_render(str(raw),scene=sc)
    source=bpy.data.images.load(str(raw),check_existing=False);px=list(source.pixels[:]);w,h=source.size;p=[];a=[];straight=[]
    for i in range(0,len(px),4):
        # A saved OpenEXR is read back by Blender as premultiplied pixels. Do not
        # multiply RGB by alpha again: that would turn P into alpha*P for thin
        # geometry. The PNG compatibility image is the safely unpremultiplied F.
        r,g,b,x=px[i:i+4]
        # Keep sub-threshold coverage consistent across premultiplied RGB,
        # alpha, and straight RGBA. Cycles can leave tiny camera-path values in
        # pixels the dataset QA classifies as transparent, violating P = alpha*F.
        if x < 1e-4:
            r=g=b=x=0.0
        p.extend((r,g,b,x));a.extend((x,x,x,1));straight.extend((r/x if x>1e-8 else 0,g/x if x>1e-8 else 0,b/x if x>1e-8 else 0,x))
    def im(n,v,alpha):z=bpy.data.images.new(n,w,h,alpha=alpha,float_buffer=True);z.pixels.foreach_set(v);return z
    pi=im("Premult",p,True);ai=im("Alpha",a,True);si=im("Straight",straight,True);sc.render.image_settings.file_format="OPEN_EXR";sc.render.image_settings.color_mode="RGBA";sc.render.image_settings.color_depth="32";pi.save_render(str(out/"linear_premult.exr"),scene=sc);sc.render.image_settings.color_mode="RGB";ai.save_render(str(out/"alpha.exr"),scene=sc);sc.render.image_settings.file_format="PNG";sc.render.image_settings.color_mode="RGBA";sc.render.image_settings.color_depth="8";si.save_render(str(out/"straight_rgba.png"),scene=sc);sc.render.image_settings.color_mode="RGB";ai.save_render(str(out/"alpha_preview.png"),scene=sc)
    for bg,n in ((0,"preview_black.png"),(1,"preview_white.png"),(.55,"preview_checker.png")):
        vals=[]
        for i in range(0,len(p),4):
            r,g,b,x=p[i:i+4];q=bg if n!="preview_checker.png" else (.35 if ((i//4//w//32+(i//4)%w//32)%2) else .75);vals.extend((r+q*(1-x),g+q*(1-x),b+q*(1-x),1))
        z=im(n,vals,False);sc.render.image_settings.color_mode="RGB";z.save_render(str(out/n),scene=sc);bpy.data.images.remove(z)
    for z in (pi,ai,si,source):bpy.data.images.remove(z)
    raw.unlink()
def render(out):bpy.ops.render.render(write_still=False);save(out)
def occlusion_depth(out):
    sc=bpy.context.scene
    view_layer=sc.view_layers[0]
    view_layer.use_pass_z=True
    view_layer.use_pass_object_index=True
    view_layer.update_render_passes()
    if not hasattr(sc,"compositing_node_group"):
        raise RuntimeError("occlusion depth export requires the Blender 5.2 compositor API")
    tree=bpy.data.node_groups.new("OcclusionDepthOutputs","CompositorNodeTree")
    sc.compositing_node_group=tree
    layers=tree.nodes.new("CompositorNodeRLayers"); layers.layer=view_layer.name; layers.update()
    output=tree.nodes.new("CompositorNodeOutputFile")
    output.directory=str(out)
    output.format.file_format="OPEN_EXR_MULTILAYER"
    output.format.color_mode="RGB"
    output.file_output_items.new("FLOAT","Depth")
    output.file_output_items.new("FLOAT","ObjectIndex")
    tree.links.new(layers.outputs["Depth"],output.inputs["Depth"])
    tree.links.new(layers.outputs["Object Index"],output.inputs["ObjectIndex"])
    sc.cycles.samples=1
    records=[]
    for index,oid in enumerate(recipe["sets"]["TARGET"],start=1):
        for object_id in recipe["sets"]["TARGET"]:
            visible(object_id,object_id==oid)
        for obj in semantic[oid]:
            obj.pass_index=index
        output.file_name=oid
        bpy.ops.render.render(write_still=False)
        records.append({"object_id":oid,"pass_index":index,"depth_exr":f"{oid}.exr"})
    (out/"depth_manifest.json").write_text(json.dumps({"sample_id":recipe["identity"]["sample_id"],"records":records,"depth_tolerance_m":0.001},indent=2))

def main():
    if VARIANT=="intervention":apply_intervention_recipe()
    _,dev=setup();build();target=recipe["sets"]["TARGET"];record={"variant":VARIANT,"probe":PROBE or None,"device":dev,"started_at":time.time(),"sample_id":recipe["identity"]["sample_id"]}
    if VARIANT == "joint":
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT.parent/"scene.blend"))
    if VARIANT in {"joint","physical_full"}:
        if VARIANT=="physical_full":background(PROBE or "checker_fine","physical")
        render(OUT)
    elif VARIANT=="without_target":
        background(PROBE or "checker_fine","physical");[visible(x,False) for x in target];render(OUT)
    elif VARIANT=="core_probe":background(PROBE,"core");render(OUT)
    elif VARIANT=="core_backplate":
        background(PROBE,"core");[visible(x,False) for x in target];render(OUT)
    elif VARIANT=="empty_core":
        # Calibration-only empty film: no backplate geometry is added, so the
        # transparent film itself must export zero alpha and zero premult RGB.
        [visible(x,False) for x in target];render(OUT)
    elif VARIANT=="independent":
        for keep in recipe["diagnostics"]["independent_order"]:
            for x in target:visible(x,x==keep)
            render(OUT/"independent"/keep)
    elif VARIANT=="intervention":
        c=recipe["diagnostics"]["intervention"]
        for oid in c.get("hide_objects",[]):visible(oid,False)
        render(OUT)
    elif VARIANT=="occlusion_depth":
        occlusion_depth(OUT)
    elif VARIANT=="geometry_debug":
        for i,x in enumerate(target):setmat([y for y in semantic[x] if y.type=="MESH"],{"preset":"opaque_principled","base_color":((i*73%255)/255,(i*151%255)/255,(i*211%255)/255,1),"roughness":1},"DEBUG_"+x)
        render(OUT);(OUT/"object_id_map.json").write_text(json.dumps({x:i for i,x in enumerate(target)},indent=2))
    else:raise RuntimeError("unsupported variant "+VARIANT)
    record["completed_at"]=time.time();(OUT/"render_log.json").write_text(json.dumps(record,indent=2));log("complete "+json.dumps(record))
main()
