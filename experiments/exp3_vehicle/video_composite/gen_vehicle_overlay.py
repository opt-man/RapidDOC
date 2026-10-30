"""Real-video temporal composites. No generated/repainted vehicle pixels.

Run with a Python environment containing opencv-python, numpy, Pillow.
Raw selected frames, masks, timestamps and settings are saved for audit.
"""
from pathlib import Path
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2

OUT = Path(__file__).resolve().parent / 'generated'
CROP = (0, 120, 1920, 1020)
TRAJECTORY_COLORS_BGR = {
    'before': (6, 5, 244),    # #F40506, sampled from the reference
    'after': (242, 3, 5),     # #0503F2, sampled from the reference
}
TRAJECTORY_COLORS_RGB = {
    'before': (244, 5, 6),    # #F40506
    'after': (5, 3, 242),     # #0503F2
}
TRAJECTORY_ALPHA = 0.60


def save_rgb(path, bgr):
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)


def get_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000*float(t))
    ok, f = cap.read()
    if not ok:
        raise ValueError(f'Cannot decode time {t}')
    return f


def vehicle_mask(frame, background):
    delta = np.max(np.abs(frame.astype(np.int16)-background.astype(np.int16)), axis=2)
    mask = np.uint8(delta > 29) * 255
    mask[:220] = 0
    mask[:, :12] = 0
    mask[:, 1908:] = 0
    mask[1010:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17,17),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c)<1500:
        raise ValueError('Vehicle detection too small')
    result = np.zeros_like(mask)
    cv2.drawContours(result,[c],-1,255,cv2.FILLED)
    result = cv2.dilate(result,np.ones((11,11),np.uint8))
    x,y,w,h=cv2.boundingRect(result)
    return result,[int(x),int(y),int(w),int(h)]


def mark_start(panel, bbox, color):
    x,y,w,h=bbox
    x-=CROP[0]
    y-=CROP[1]
    tip=np.array([x+15,y+h*0.48],dtype=float)
    text_center=np.array([x-175,y+h*0.60],dtype=float)
    tail=text_center+np.array([82,0])
    direction=(tip-tail)/np.linalg.norm(tip-tail)
    normal=np.array([-direction[1],direction[0]])
    head=[tuple(tip),tuple(tip-22*direction+10*normal),tuple(tip-22*direction-10*normal)]
    draw=ImageDraw.Draw(panel)
    draw.line([tuple(tail),tuple(tip)],fill='white',width=9)
    draw.line([tuple(tail),tuple(tip)],fill=color,width=5)
    draw.polygon(head,fill=color)
    font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',46)
    draw.text(tuple(text_center),'Start',font=font,anchor='mm',fill=color,
              stroke_width=2,stroke_fill='white')
    return {'text':'Start','tip_xy_in_crop':tip.tolist(),'text_xy_in_crop':text_center.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True, help='Original raw camera recording.')
    parser.add_argument('--after-start',type=float,default=12.8)
    parser.add_argument('--before-start',type=float,default=100.8)
    parser.add_argument('--before-frames',type=int,default=11)
    parser.add_argument('--after-frames',type=int,default=11)
    parser.add_argument('--temporal-fade',action='store_true',
                        help='Fade vehicle opacity from 0.35 (early) to 1.0 (late); save separate outputs.')
    parser.add_argument('--start-marked',action='store_true',
                        help='Include motion onset, fade 1.0 to 0.35 and label Start; save separate outputs.')
    parser.add_argument('--selection-file',type=Path,
                        help='JSON with selected elapsed times under before/after for a connected trajectory.')
    parser.add_argument('--draw-trajectory',action='store_true',
                        help='Overlay the measured image-space trajectory detected from the same video segment.')
    args=parser.parse_args()
    if min(args.before_frames,args.after_frames)<2:
        parser.error('Each condition needs at least two frames.')
    custom=json.loads(args.selection_file.read_text(encoding='utf-8')) if args.selection_file else None
    if custom and not args.start_marked:
        parser.error('--selection-file requires --start-marked')
    suffix='_connected' if custom else ('_startfade' if args.start_marked else ('_timefade' if args.temporal_fade else ''))
    first_offset=0 if args.start_marked else 5
    OUT.mkdir(parents=True, exist_ok=True)
    cap=cv2.VideoCapture(str(args.video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        parser.error(f'Cannot open raw video: {args.video}')
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    meta={'video':str(args.video),'fps':fps,'duration_s':duration,'crop_xyxy':CROP,
          'condition_order_in_video':['after','before'],
          'background':'per-channel temporal median of 23 actual frames per segment',
          'vehicle_pixels':'unwarped source RGB, alpha composited over temporal-median background; 3px feathered mask edge',
          'opacity_rule':('linear in elapsed time, 1.00 at first source time to 0.35 at last source time' if args.start_marked else
                         ('linear in elapsed time, 0.35 at 5s to 1.00 at 65s'
                          if args.temporal_fade else 'constant 0.85')),
          'selection_rule':custom.get('description','spatial-distance sampling from actual video frames') if custom else f'equispaced frames at {first_offset} to 65s after estimated motion onset: before={args.before_frames}, after={args.after_frames}',
          'selection_file':str(args.selection_file) if custom else None,
          'layer_order':'earliest on top' if args.start_marked else 'latest on top',
          'segmentation':'difference threshold29, morphology open3 close17 dilation11',
          'trajectory_overlay':(
              'vehicle-mask centroid sampled every 0.25 s from the same physical video segment; '
              'before=#F40506, after=#0503F2, opacity=0.60'
              if args.draw_trajectory else None),
          'no_generative_editing':True,'segments':{}}
    panels={}
    for label,start in [('after',args.after_start),('before',args.before_start)]:
        folder=OUT/f'{label}{suffix}'
        folder.mkdir(parents=True,exist_ok=True)
        bg_times=np.linspace(start+5,start+68,23)
        background=np.median(np.stack([get_frame(cap,t) for t in bg_times]),axis=0).astype(np.uint8)
        save_rgb(folder/'background.png',background)
        composite=background.copy()
        offsets=np.array(custom[label],dtype=float) if custom else np.linspace(first_offset,65,args.before_frames if label=='before' else args.after_frames)
        if np.any(np.diff(offsets)<=0) or offsets[0]<0 or offsets[-1]>70.7:
            raise ValueError('Invalid selected elapsed times')
        nframes=len(offsets)
        times=start+offsets
        records=[]
        layers=[]
        sheet=Image.new('RGB',(1200,((nframes+3)//4)*195),'white')
        for j,t in enumerate(times):
            frame=get_frame(cap,t)
            mask,bbox=vehicle_mask(frame,background)
            save_rgb(folder/f'frame_{j:02d}_{t:.3f}s.png',frame)
            Image.fromarray(mask).save(folder/f'mask_{j:02d}.png')
            distance=cv2.distanceTransform(mask,cv2.DIST_L2,3)
            fraction=(t-times[0])/(times[-1]-times[0])
            opacity=(float(1.0-0.65*fraction) if args.start_marked else
                     (float(0.35+0.65*fraction) if args.temporal_fade else 0.85))
            alpha=(opacity*np.clip(distance/3,0,1))[:,:,None]
            layers.append((frame,alpha))
            records.append({'index':j,'video_time_s':float(t),
                            'motion_elapsed_s':float(t-start),'opacity':opacity,'bbox_xywh':bbox})
            preview=frame.copy()
            cv2.rectangle(preview,tuple(bbox[:2]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,255),3)
            im=Image.fromarray(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)).resize((300,169))
            px,py=(j%4)*300,(j//4)*195
            sheet.paste(im,(px,py+22))
            ImageDraw.Draw(sheet).text((px+5,py+4),f'{label} #{j}  {t:.2f}s',fill='black')
        sheet.save(folder/'selected_frames_review.png')
        for frame,alpha in (reversed(layers) if args.start_marked else layers):
            composite=np.round(alpha*frame+(1-alpha)*composite).astype(np.uint8)
        x1,y1,x2,y2=CROP
        trajectory_records=[]
        if args.draw_trajectory:
            trajectory_offsets=np.arange(0.0,70.5001,0.25)
            trajectory_points=[]
            for offset in trajectory_offsets:
                frame=get_frame(cap,start+float(offset))
                mask,_=vehicle_mask(frame,background)
                moments=cv2.moments(mask, binaryImage=True)
                if moments['m00']<=0:
                    raise ValueError(f'Empty vehicle mask at {start+offset:.3f}s')
                cx=moments['m10']/moments['m00']-x1
                cy=moments['m01']/moments['m00']-y1
                trajectory_points.append((int(round(cx)),int(round(cy))))
                trajectory_records.append({
                    'video_time_s':float(start+offset),
                    'motion_elapsed_s':float(offset),
                    'center_xy_in_crop':[float(cx),float(cy)],
                })
            cropped=composite[y1:y2,x1:x2].copy()
            polyline=np.asarray(trajectory_points,dtype=np.int32).reshape((-1,1,2))
            # Match Fig. 5: red before learning and blue after learning.
            # Draw both the white halo and colored centerline at 60% opacity.
            trajectory_layer=cropped.copy()
            cv2.polylines(trajectory_layer,[polyline],False,(248,248,248),10,cv2.LINE_AA)
            cv2.polylines(
                trajectory_layer,[polyline],False,
                TRAJECTORY_COLORS_BGR[label],5,cv2.LINE_AA)
            cropped=cv2.addWeighted(
                trajectory_layer,TRAJECTORY_ALPHA,
                cropped,1.0-TRAJECTORY_ALPHA,0.0)
        else:
            cropped=composite[y1:y2,x1:x2]
        panels[label]=Image.fromarray(cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB))
        annotation=(
            mark_start(
                panels[label],records[0]['bbox_xywh'],
                TRAJECTORY_COLORS_RGB[label])
            if args.start_marked else None)
        panels[label].save(OUT/f'vehicle_{label}{suffix}.png',dpi=(500,500))
        meta['segments'][label]={'approximate_start_s':start,'frame_count':nframes,
                                 'sampling_interval_s':None if custom else (65-first_offset)/(nframes-1),
                                 'sampled_duration_s':float(offsets[-1]-offsets[0]),
                                 'start_annotation':annotation,
                                 'measured_trajectory':trajectory_records,
                                 'background_times_s':bg_times.tolist(),'frames':records}
    cap.release()
    w,h=panels['before'].size
    gap=28
    combined=Image.new('RGB',(w*2+gap,h+100),'white')
    fontpath=Path('C:/Windows/Fonts/times.ttf')
    font=ImageFont.truetype(str(fontpath),42)
    draw=ImageDraw.Draw(combined)
    for k,label in enumerate(['before','after']):
        combined.paste(panels[label],(k*(w+gap),0))
        text=f"({'ab'[k]}) {'Before' if label=='before' else 'After'} learning"
        draw.text((k*(w+gap)+w/2,h+34),text,font=font,fill='black',anchor='mt')
    combined.save(OUT/f'vehicle_before_after{suffix}.png',dpi=(500,500))
    (OUT/f'composite_metadata{suffix}.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(OUT/f'vehicle_before_after{suffix}.png')


if __name__=='__main__':
    main()
