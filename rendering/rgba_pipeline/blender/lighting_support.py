"""Explicit neutral illumination, with no original scene environment or HDRI."""
import bpy
from mathutils import Vector
from camera_support import aim


def neutral_world(spec):
    world = bpy.data.worlds.new('NeutralWhiteWorld')
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new('ShaderNodeBackground')
    color = spec.get('world_color', [0,0,0])
    background.inputs['Color'].default_value = (*color, 1)
    background.inputs['Strength'].default_value = float(spec.get('world_strength', 0))
    output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(background.outputs[0], output.inputs['Surface'])
    bpy.context.scene.world = world


def neutral_lights(spec, center, size, camera):
    orientation = camera.matrix_world.to_quaternion()
    right, up, forward = (orientation @ Vector(axis) for axis in ((1,0,0),(0,1,0),(0,0,-1)))
    records = []
    for item in spec.get('lights', []):
        x,y,z = item['offset_in_object_sizes']
        location = center + size*(right*x+forward*y+up*z)
        data = bpy.data.lights.new('ISOLATION_'+item['id'], 'AREA')
        data.color = (1,1,1)
        data.energy = float(item['power_at_one_meter'])*size**2
        data.shape = 'DISK'
        data.size = float(item['size_in_object_sizes'])*size
        light = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        aim(light, center)
        records.append({'id': item['id'], 'location': list(location), 'energy': data.energy,
                        'size': data.size, 'color': list(data.color)})
    return records
