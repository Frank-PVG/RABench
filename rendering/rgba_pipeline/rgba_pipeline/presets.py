"""Resolve the small, explicit P1 preset registry into self-contained recipes."""
from __future__ import annotations
import copy
import math
from pathlib import Path

from .io import read_json
from .paths import CONFIG_ROOT, PROJECT_ROOT


class PresetBook:
    def __init__(self, root: Path | None = None):
        self.root = root or CONFIG_ROOT/'p1'
        self.documents = {name:read_json(self.root/(name+'.json')) for name in
                          ('scenes','materials','cameras','lighting','export','objects','suite')}
        if any(doc['version'] != 'p1-v1' for doc in self.documents.values()):
            raise ValueError('P1 preset versions must agree')

    def get(self, kind, name):
        return copy.deepcopy(self.documents[kind]['presets'][name])

    def surface(self, name):
        return copy.deepcopy(self.documents['materials']['object_surfaces'][name])

    def unit(self, name, scene_id, material_variant, assets):
        spec = self.get('objects', name)
        scene = self.get('scenes', scene_id)
        base = self.documents['cameras']['bases'][scene['camera_basis']]
        reference = self.documents['cameras']['bases']['blue_wall']
        def azimuth(camera):
            d = [camera['location'][i]-camera['look_at'][i] for i in range(3)]
            return math.atan2(d[0], -d[1])
        rotation = [0,0,spec['yaw_radians']+azimuth(base)-azimuth(reference)]
        transform = {'location':scene['anchor'], 'rotation':rotation, 'scale':[1,1,1]}
        if spec['role'] == 'asset':
            asset = assets[spec['asset_id']]
            size = spec.get('size_m',asset['size_m'])
            if size > scene['max_size_m']: raise ValueError(f'{name} exceeds {scene_id} size limit')
            extent = asset['source_bounds']['extent']
            height = size*extent[2]/max(extent)
            obj = {'id':'target','role':'asset','asset_id':asset['asset_id'],
                   'asset_path':str(Path(asset['local_path']).relative_to(PROJECT_ROOT)),
                   'source_uid':asset['uid'],'lineage_id':asset['lineage_id'],
                   'source_url':asset['source_page'],'source_license':asset['license'],'source_author':asset['author'],
                   'material_policy':'preserve_source','surface_overrides':{},'size_m':size,
                   'semantic_object':asset['name'],'support_z':scene['anchor'][2],'transform':transform}
        else:
            size, height = spec['size_m'], spec['height_m']
            components = spec['components']
            overrides = spec['material_variants'][material_variant]
            for part in components:
                material_id = overrides.get(part['id'],part.pop('material_preset'))
                part['material'] = self.surface(material_id)
            obj = {'id':'target','role':'assembly','semantic_object':spec['semantic_object'],
                   'components':components,'size_m':size,'transform':transform,'support_z':scene['anchor'][2]}
        return obj, size, height

    def camera(self, scene_id, preset_id, size, height):
        scene = self.get('scenes',scene_id)
        base = copy.deepcopy(self.documents['cameras']['bases'][scene['camera_basis']])
        preset = self.get('cameras',preset_id)
        delta = [base['location'][i]-base['look_at'][i] for i in range(3)]
        angle = math.radians(preset['orbit_degrees'])
        x,y = delta[:2]
        delta[:2] = [math.cos(angle)*x-math.sin(angle)*y, math.sin(angle)*x+math.cos(angle)*y]
        scale = max(.6,min(1.2,size/.22))*preset['distance_scale']
        delta = [value*scale for value in delta]
        delta[2] += preset['height_offset_m']
        look = [*scene['anchor'][:2],scene['anchor'][2]+height*.5]
        base.update({'look_at':look,'location':[a+b for a,b in zip(look,delta)]})
        return base
