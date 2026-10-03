"""Blender-side renderer, called by `python -m rgba_pipeline render`."""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_templates import load_template
from render_export import save_render_buffers
from geometry_support import slot_bounds, check_placement
from pbr_materials import setmat
from object_builder import build_object
from camera_support import create_camera, camera_record
from isolation_support import empty_scene, configure_isolation

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
    isolated = VARIANT == 'isolated_object'
    if isolated:
        empty_scene(recipe['isolation'])
    elif recipe.get('scene_template'):
        semantic['__scene_environment__'] = load_template(recipe['scene_template'])
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
    else:
        bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; cfg=recipe["render"]
    resolution=cfg['resolution']; width,height=resolution if isinstance(resolution,list) else (resolution,resolution)
    scene.render.engine="CYCLES"; scene.render.resolution_x=width; scene.render.resolution_y=height; scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=1.0;scene.render.pixel_aspect_y=1.0
    scene.render.film_transparent=VARIANT not in {'scene_full','scene_without_target'}; scene.cycles.samples=cfg["samples"]; scene.cycles.seed=recipe["seeds"]["render"]; scene.cycles.use_denoising=cfg.get('denoise',False); scene.cycles.use_adaptive_sampling=cfg.get('adaptive_sampling',False)
    if hasattr(scene.cycles, 'use_light_tree'): scene.cycles.use_light_tree=True
    scene.cycles.caustics_reflective=cfg.get('caustics_reflective',True)
    scene.cycles.caustics_refractive=cfg.get('caustics_refractive',True)
    scene.render.threads_mode='FIXED'; scene.render.threads=cfg.get('threads',8)
    scene.cycles.max_bounces=cfg["max_bounces"]; scene.cycles.transparent_max_bounces=cfg["transparent_bounces"]; scene.cycles.transmission_bounces=cfg["transmission_bounces"]; scene.cycles.sample_clamp_direct=cfg.get("clamp_direct",0); scene.cycles.sample_clamp_indirect=cfg.get("clamp_indirect",0)
    if hasattr(scene.render,"filter_type"): scene.render.filter_type="BOX"
    scene.render.image_settings.color_mode="RGBA"; scene.render.image_settings.color_depth="16"; scene.view_settings.view_transform=cfg.get('view_transform','Standard'); scene.view_settings.look="None"
    scene.view_settings.exposure=cfg.get('exposure',0.0)
    if not isolated and not recipe.get('scene_template'):
        if scene.world is None: scene.world=bpy.data.worlds.new('Environment')
        scene.world.use_nodes=True; bg=scene.world.node_tree.nodes.get("Background"); bg.inputs["Color"].default_value=(*recipe["environment"]["world_color"],1); bg.inputs["Strength"].default_value=.25
    device=gpu(scene); c=recipe["camera"]
    create_camera(c)
    for item in ([] if isolated else recipe["environment"].get("lights",[])):
        bpy.ops.object.light_add(type=item["type"],location=item["location"]); light=bpy.context.object; light.name="ENV_"+item["id"]; light.data.energy=item["energy"]; light.data.shape="DISK"; light.data.size=item["size"]; look(light,c["look_at"])
    return scene,device



def tag(oid,objs): semantic[oid]=objs; [o.__setitem__("semantic_object_id",oid) for o in objs]


def build():
    for s in recipe["objects"]:
        objs=build_object(s);tag(s["id"],objs)
    for oid in recipe["sets"].get("ENVIRONMENT",[]):
        if oid not in semantic: raise RuntimeError(f"environment set references unknown object {oid}")
        for o in semantic[oid]:
            if o.type in {"MESH","CURVE","VOLUME"}: o.visible_camera=bool(o.get('_rgba_original_camera_visibility',True)) if VARIANT in {'scene_full','scene_without_target'} else False
    if recipe.get('scene_template'):
        (OUT/'placement.json').write_text(json.dumps({oid:slot_bounds(semantic[oid]) for oid in recipe['sets']['TARGET'] if any(o.type=='MESH' for o in semantic[oid])},indent=2))
        if VARIANT == 'joint':
            checks = check_placement(recipe, semantic)
            (OUT/'placement_checks.json').write_text(json.dumps(checks, indent=2))
            if checks['errors']: raise RuntimeError('; '.join(checks['errors']))
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
    save_render_buffers(out,{**recipe['render'],'context_output':VARIANT in {'scene_full','scene_without_target'}})
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
    if VARIANT == 'isolated_object':
        audit = configure_isolation(recipe, semantic[recipe['isolation']['object_id']])
        (OUT/'isolation.json').write_text(json.dumps(audit, indent=2))
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'scene.blend'))
        render(OUT)
    elif VARIANT in {'scene_full','scene_without_target'}:
        (OUT/'camera.json').write_text(json.dumps(camera_record(bpy.context.scene.camera),indent=2))
        if VARIANT=='scene_without_target': [visible(x,False) for x in target]
        render(OUT)
    elif VARIANT in {"joint","physical_full"}:
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
