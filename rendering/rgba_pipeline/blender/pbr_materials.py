"""Source PBR, procedural optical materials and explicit scene surface edits."""
import fnmatch
import bpy
from render_paths import resolve_path

def _color_input(material, socket, tint, name):
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    if socket.is_linked:
        upstream = socket.links[0].from_socket
        for link in list(socket.links):
            links.remove(link)
        mix = nodes.new('ShaderNodeMixRGB')
        mix.name = name
        mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = 1.0
        mix.inputs[2].default_value = (*tint[:3], 1.0)
        links.new(upstream, mix.inputs[1])
        links.new(mix.outputs[0], socket)
    else:
        original = list(socket.default_value)
        socket.default_value = (*[original[i] * tint[i] for i in range(3)], original[3])


def apply_material_preset(preset: dict) -> None:
    """Copy only selected material instances; retain texture and normal details.

Optional `maps` replaces a complete, explicitly authored PBR surface set. Object
atlases and nonselected material slots retain their original UVs and graph.
"""
    for edit in preset.get('edits', []):
        copies = {}
        modified_shaders = 0
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            if not any(fnmatch.fnmatchcase(obj.name, p) for p in edit.get('objects', ['*'])):
                continue
            obj.data = obj.data.copy()
            for material_index, slot in enumerate(obj.material_slots):
                old = slot.material
                if old is None or not old.use_nodes:
                    continue
                if not any(fnmatch.fnmatchcase(old.name, p) for p in edit.get('materials', ['*'])):
                    continue
                if edit.get('maps'):
                    uv=obj.data.uv_layers.get('TemplateSurfaceUV') or obj.data.uv_layers.new(name='TemplateSurfaceUV')
                    uv.active_render=False
                    scale=float(edit.get('texture_size_m',1.0))
                    for polygon in obj.data.polygons:
                        if polygon.material_index!=material_index: continue
                        normal=obj.matrix_world.inverted().transposed().to_3x3()@polygon.normal
                        axis=max(range(3),key=lambda i:abs(normal[i]))
                        axes=(0,1) if axis==2 else (0,2) if axis==1 else (1,2)
                        for loop in polygon.loop_indices:
                            point=obj.matrix_world@obj.data.vertices[obj.data.loops[loop].vertex_index].co
                            uv.data[loop].uv=(point[axes[0]]/scale,point[axes[1]]/scale)
                if old.name not in copies:
                    mat = old.copy()
                    mat.name = old.name + '__' + preset.get('id', 'variant')
                    copies[old.name] = mat
                    for shader in [n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']:
                        modified_shaders += 1
                        if 'base_color_tint' in edit:
                            _color_input(mat, shader.inputs['Base Color'], edit['base_color_tint'], 'TemplateTint')
                        if 'roughness_multiplier' in edit:
                            multiply_socket(mat, shader.inputs['Roughness'], edit['roughness_multiplier'])
                        for channel, filename in edit.get('maps', {}).items():
                            texture = mat.node_tree.nodes.new('ShaderNodeTexImage')
                            texture.image = bpy.data.images.load(str(resolve_path(filename)), check_existing=False)
                            texture.image.colorspace_settings.name = 'sRGB' if channel == 'base_color' else 'Non-Color'
                            uv_node=mat.node_tree.nodes.new('ShaderNodeUVMap'); uv_node.uv_map='TemplateSurfaceUV'
                            mat.node_tree.links.new(uv_node.outputs['UV'],texture.inputs['Vector'])
                            output = texture.outputs['Color']
                            if channel == 'normal':
                                normal = mat.node_tree.nodes.new('ShaderNodeNormalMap')
                                normal.uv_map='TemplateSurfaceUV'
                                mat.node_tree.links.new(output, normal.inputs['Color'])
                                output = normal.outputs['Normal']
                            sockets = {'base_color': 'Base Color', 'roughness': 'Roughness', 'metallic': 'Metallic', 'normal': 'Normal'}
                            socket = shader.inputs[sockets[channel]]
                            for link in list(socket.links):
                                mat.node_tree.links.remove(link)
                            mat.node_tree.links.new(output, socket)
                slot.material = copies[old.name]
        if not modified_shaders:
            raise ValueError(f"Material preset {preset.get('id')!r} matched no editable Principled shader: {edit}")


def multiply_socket(material, socket, factor: float) -> None:
    if socket.is_linked:
        upstream = socket.links[0].from_socket
        for link in list(socket.links):
            material.node_tree.links.remove(link)
        node = material.node_tree.nodes.new('ShaderNodeMath')
        node.operation = 'MULTIPLY'
        node.inputs[1].default_value = factor
        material.node_tree.links.new(upstream, node.inputs[0])
        material.node_tree.links.new(node.outputs[0], socket)
    else:
        socket.default_value *= factor


def set_scalar(material, socket, value: float) -> None:
    for link in list(socket.links):
        material.node_tree.links.remove(link)
    socket.default_value = value


def material(name,spec):
    mat=bpy.data.materials.new(name); mat.use_nodes=True; t=mat.node_tree; t.nodes.clear(); out=t.nodes.new("ShaderNodeOutputMaterial"); p=spec["preset"]
    if p=="opaque_principled":
        n=t.nodes.new("ShaderNodeBsdfPrincipled"); n.inputs["Base Color"].default_value=spec.get("base_color",[.8,.8,.8,1]); n.inputs["Roughness"].default_value=spec.get("roughness",.5); n.inputs["Metallic"].default_value=spec.get("metallic",0.0); t.links.new(n.outputs[0],out.inputs["Surface"])
    elif p=="neutral_thin_coverage":
        mix=t.nodes.new("ShaderNodeMixShader"); mix.inputs[0].default_value=spec["coverage"]; a=t.nodes.new("ShaderNodeBsdfTransparent"); b=t.nodes.new("ShaderNodeBsdfDiffuse"); b.inputs["Color"].default_value=spec.get("base_color",[.78,.78,.78,1]); t.links.new(a.outputs[0],mix.inputs[1]); t.links.new(b.outputs[0],mix.inputs[2]); t.links.new(mix.outputs[0],out.inputs["Surface"])
    elif p=="glass":
        n=t.nodes.new("ShaderNodeBsdfGlass"); n.inputs["IOR"].default_value=spec["ior"]; n.inputs["Roughness"].default_value=spec.get("roughness",0); t.links.new(n.outputs[0],out.inputs["Surface"])
    elif p=="liquid":
        n=t.nodes.new("ShaderNodeBsdfGlass"); n.inputs["IOR"].default_value=spec["ior"]; n.inputs['Roughness'].default_value=spec.get('roughness',0.0); t.links.new(n.outputs[0],out.inputs["Surface"]); v=t.nodes.new("ShaderNodeVolumeAbsorption"); v.inputs["Color"].default_value=(*spec["absorption_color"],1); v.inputs["Density"].default_value=spec["absorption_density"]; t.links.new(v.outputs[0],out.inputs["Volume"])
    else: raise RuntimeError("unknown material "+p)
    return mat


def apply_source_surface_overrides(meshes,overrides,asset_id):
    # Change only requested shader channels while retaining downloaded image maps.
    if not overrides: return
    material_copies={}
    for mesh in meshes:
        for slot in mesh.material_slots:
            original=slot.material
            if original and original.name in material_copies:
                slot.material=material_copies[original.name]; continue
            material=original.copy() if original else None
            if original: material_copies[original.name]=material; slot.material=material
            if not material or not material.use_nodes: continue
            for shader in (n for n in material.node_tree.nodes if n.type=="BSDF_PRINCIPLED"):
                if "metallic" in overrides: set_scalar(material,shader.inputs["Metallic"],float(overrides["metallic"]))
                if "roughness" in overrides: set_scalar(material,shader.inputs["Roughness"],float(overrides["roughness"]))
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
