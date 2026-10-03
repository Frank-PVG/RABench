"""Download six maps for two explicit CC0 wood surface presets."""
import concurrent.futures
import json
import urllib.request
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=next((p for p in PROJECT.parents if (p/'runtime/bpy_runtime').is_dir()),PROJECT.parents[1])

def fetch(item):
    material,channel,suffix=item
    name=f'{material}_{suffix}_2k.jpg'
    url=f'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/{material}/{name}'
    output=PROJECT/'assets/materials'/material/name
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists():
        temporary=WORKSPACE/'codex_tmp/rgba_pipeline'/name
        temporary.parent.mkdir(parents=True,exist_ok=True)
        direct_url=url.replace('https://dl.polyhaven.org/','https://f002.backblazeb2.com/')
        with urllib.request.urlopen(direct_url,timeout=90) as response, temporary.open('wb') as out:
            while chunk:=response.read(1024*1024): out.write(chunk)
        temporary.replace(output)
    return material,channel,str(output.relative_to(PROJECT)),url

if __name__=='__main__':
    jobs=[(material,channel,suffix) for material in ('wood_table_worn','wood_floor_worn')
          for channel,suffix in [('base_color','diff'),('roughness','rough'),('normal','nor_gl')]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(fetch,jobs))
    manifest={}
    for material,channel,path,url in rows:
        entry=manifest.setdefault(material,{'license':'CC0','source_page':f'https://polyhaven.com/a/{material}','maps':{},'urls':{}})
        entry['maps'][channel]=path; entry['urls'][channel]=url
    (PROJECT/'assets/materials/source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('MATERIALS_READY',len(manifest),flush=True)
