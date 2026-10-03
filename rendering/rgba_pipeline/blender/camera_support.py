"""Camera construction and framing shared by paired and isolated renders."""
import bpy
from mathutils import Vector


def aim(obj, point):
    obj.rotation_euler = (Vector(point)-obj.location).to_track_quat('-Z', 'Y').to_euler()


def create_camera(spec):
    if spec.get('use_template'):
        return bpy.context.scene.camera
    bpy.ops.object.camera_add(location=spec['location'])
    camera = bpy.context.object
    camera.name = 'TargetCamera'
    camera.data.lens = spec['lens_mm']
    camera.data.sensor_width = spec['sensor_width_mm']
    camera.data.clip_start = spec['clip_start']
    camera.data.clip_end = spec['clip_end']
    camera.data.dof.use_dof = False
    aim(camera, spec['look_at'])
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def fit_camera(camera, points, center, size, fill):
    orientation = camera.matrix_world.to_quaternion()
    right, up, forward = (orientation @ Vector(axis) for axis in ((1,0,0),(0,1,0),(0,0,-1)))
    frame = camera.data.view_frame(scene=bpy.context.scene)
    sx = max(abs(p.x/p.z) for p in frame)
    sy = max(abs(p.y/p.z) for p in frame)
    distance = max(max(abs((p-center).dot(right))/(sx*fill), abs((p-center).dot(up))/(sy*fill))
                   - (p-center).dot(forward) for p in points)
    camera.location = center-forward*max(distance, size)
    camera.data.clip_start = min(camera.data.clip_start, size*.001)
    camera.data.clip_end = max(camera.data.clip_end, size*100)
    bpy.context.view_layer.update()


def camera_record(camera):
    return {'location': list(camera.location), 'rotation': list(camera.rotation_euler),
            'lens_mm': camera.data.lens, 'sensor_width_mm': camera.data.sensor_width,
            'matrix_world': [list(row) for row in camera.matrix_world]}
