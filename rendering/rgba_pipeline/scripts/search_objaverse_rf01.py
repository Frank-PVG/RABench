from __future__ import annotations
import gzip, hashlib, json, re
from pathlib import Path
from rgba_pipeline.budget import BudgetedDownloader
from rgba_pipeline.discovery import CACHE, OBJAVERSE_BASE, _category_uids, _read_gzip_json
from rgba_pipeline.experiment import EXPERIMENT_ID
from rgba_pipeline.io import write_json

RUN=Path(__file__).resolve().parents[1]/'runs'/EXPERIMENT_ID
MAX_UIDS=60
MAX_PREVIEWS=12
TERMS=('white bowl','porcelain bowl','ceramic bowl','white ceramic','ivory bowl','bowl')
def norm(s): return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
def score(text):
 t=norm(text)
 return sum(weight for term,weight in [('white bowl',12),('porcelain bowl',12),('ceramic bowl',10),('ivory bowl',10),('white',3),('porcelain',3),('ceramic',2),('bowl',2)] if term in t)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 dl=BudgetedDownloader(RUN/'download_budget.sqlite',RUN)
 lvis=_read_gzip_json(CACHE/'lvis-annotations.json.gz');paths=_read_gzip_json(CACHE/'object-paths.json.gz')
 index=_category_uids(lvis,paths,'bowl')[:MAX_UIDS]
 shards={}
 for _,uid in index:
  rel=paths[uid]; shard=Path(rel).parent.name
  if shard in shards: continue
  local=RUN/'assets'/'metadata'/f'{shard}.json.gz'; cached=CACHE/'metadata'/f'{shard}.json.gz'
  if local.exists(): shards[shard]=local
  elif cached.exists(): shards[shard]=cached
  else:
   rec=dl.fetch(f'{OBJAVERSE_BASE}/metadata/{shard}.json.gz',local,purpose='selected_objaverse_candidate_metadata',asset_id=f'metadata:{shard}',max_file_bytes=16*1024*1024)
   shards[shard]=Path(rec['path'])
 docs={s:json.loads(gzip.decompress(p.read_bytes())) for s,p in shards.items()}
 rows=[]
 for category,uid in index:
  shard=Path(paths[uid]).parent.name;a=docs[shard].get(uid)
  if not a:continue
  glb=(a.get('archives') or {}).get('glb') or {};license=str(a.get('license','')).lower();size=int(glb.get('size') or 0);faces=int(a.get('faceCount') or glb.get('faceCount') or 0)
  tags=[str(x.get('name','')) for x in a.get('tags',[])];name=str(a.get('name',''));s=score(' '.join([name,category,*tags]))
  if not a.get('isDownloadable') or license not in {'cc0','by','cc0-1.0','cc-by-4.0'} or not size or size>512*1024*1024 or faces>1_000_000: continue
  images=((a.get('thumbnails') or {}).get('images') or []);images=[x for x in images if x.get('url')];images.sort(key=lambda x:(int(x.get('width') or 512)>720,abs(int(x.get('width') or 512)-512)))
  if images:
   rows.append({'scene_id':'RF01','dataset':'Objaverse 1.0','uid':uid,'name':name,'tags':tags,'author':(a.get('user') or {}).get('displayName'),'license':license,'face_count':faces,'glb_size':size,'object_path':paths[uid],'metadata_shard':shard,'metadata_sha256':sha(shards[shard]),'source_url':a.get('viewerUrl'),'thumbnail_url':images[0]['url'],'score':s,'selection_status':'eligible_thumbnail_review'})
 rows.sort(key=lambda x:(-x['score'],x['glb_size'],x['uid']))
 preview_root=RUN/'selection'/'targeted_rf01'/'thumbnails';preview_root.mkdir(parents=True,exist_ok=True)
 for n,row in enumerate(rows[:MAX_PREVIEWS],1):
  path=preview_root/f'{n:02d}_{row["uid"]}.jpg'
  try:
   result=dl.fetch(row['thumbnail_url'],path,purpose='selected_asset_thumbnail',asset_id=f'thumbnail:RF01-targeted:{row["uid"]}',max_file_bytes=8*1024*1024)
   row['thumbnail']=result['path'];row['thumbnail_sha256']=result['sha256']
  except Exception as exc:row['thumbnail_error']=str(exc)
 result={'experiment_id':EXPERIMENT_ID,'search':'Objaverse LVIS bowl category, first 60 deterministic candidates; exact UID metadata then CC0/CC-BY and geometry-size gates; top 12 candidate previews','terms':list(TERMS),'candidates':rows,'download_budget':dl.status()}
 out=RUN/'selection'/'targeted_rf01'/'candidates.json';write_json(out,result)
 from PIL import Image,ImageDraw
 shown=[x for x in rows[:MAX_PREVIEWS] if x.get('thumbnail')]
 w,h=210,180;sheet=Image.new('RGB',(w*4,h*((len(shown)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
 for i,row in enumerate(shown):
  x=i%4*w;y=i//4*h;im=Image.open(row['thumbnail']).convert('RGB');im.thumbnail((w-8,h-38));sheet.paste(im,(x+(w-im.width)//2,y+25));draw.text((x+3,y+3),f"{row['uid'][:10]} score={row['score']}",fill='black');draw.text((x+3,y+h-10),row['name'][:28],fill='#333')
 sheet.save(out.parent/'contact_sheet.jpg',quality=86)
 print(json.dumps({'candidates':len(rows),'previewed':len(shown),'budget':dl.status(),'top':[{k:r.get(k) for k in ('uid','name','license','score','glb_size','thumbnail')} for r in shown]},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
