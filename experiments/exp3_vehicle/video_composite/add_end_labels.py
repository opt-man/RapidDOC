"""Add endpoint annotations without changing any source composite pixels elsewhere."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent / 'generated'
meta=json.loads((ROOT/'composite_metadata_connected.json').read_text(encoding='utf-8'))
panels={}
font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',46)
COLORS={
    'before': (244,5,6),    # #F40506
    'after': (5,3,242),     # #0503F2
}
for condition in ('before','after'):
    panel=Image.open(ROOT/f'vehicle_{condition}_connected.png').convert('RGB')
    last=meta['segments'][condition]['frames'][-1]
    x,y,w,h=last['bbox_xywh']
    x-=meta['crop_xyxy'][0]
    y-=meta['crop_xyxy'][1]
    tip=np.array([x+w-12,y+h*0.60],dtype=float)
    text_center=np.array([x+w+155,y+h*0.80],dtype=float)
    tail=text_center-np.array([67,0])
    direction=(tip-tail)/np.linalg.norm(tip-tail)
    normal=np.array([-direction[1],direction[0]])
    head=[tuple(tip),tuple(tip-22*direction+10*normal),tuple(tip-22*direction-10*normal)]
    draw=ImageDraw.Draw(panel)
    draw.line([tuple(tail),tuple(tip)],fill='white',width=9)
    draw.line([tuple(tail),tuple(tip)],fill=COLORS[condition],width=5)
    draw.polygon(head,fill=COLORS[condition])
    draw.text(tuple(text_center),'End',font=font,anchor='mm',fill=COLORS[condition],
              stroke_width=2,stroke_fill='white')
    panel.save(ROOT/f'vehicle_{condition}_endpoints.png',dpi=(500,500))
    panels[condition]=panel
    meta['segments'][condition]['end_annotation']={
        'text':'End','source_video_time_s':last['video_time_s'],
        'tip_xy_in_crop':tip.tolist(),'text_xy_in_crop':text_center.tolist(),
        'note':'Points to terminal vehicle extent in the actual last frame; terminal pose may overlap the initial pose.'}

w,h=panels['before'].size
combined=Image.new('RGB',(2*w+28,h+100),'white')
draw=ImageDraw.Draw(combined)
label_font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',42)
for i,condition in enumerate(('before','after')):
    px=i*(w+28)
    combined.paste(panels[condition],(px,0))
    label=f"({'ab'[i]}) {'Before' if condition=='before' else 'After'} learning"
    draw.text((px+w/2,h+34),label,font=label_font,fill='black',anchor='mt')
combined.save(ROOT/'vehicle_before_after_endpoints.png',dpi=(500,500))
(ROOT/'composite_metadata_endpoints.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print(ROOT/'vehicle_before_after_endpoints.png')
