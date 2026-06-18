#!/usr/bin/env python3
"""
aruco_collage_a4.py — true-scale ArUco markers across A4 pages (no marker is cut).

A marker's BLACK SQUARE = BORDER_RATIO x total. The black square is what must land
inside the printable area; the surrounding white border may overrun into the printer
margin harmlessly. So a 200 mm marker (black 155.6 mm) fits WHOLE on one A4:
  - markers whose total <= packable width  -> shelf-packed (several per page)
  - markers whose total > packable but black square fits the page -> own centred page
  - anything genuinely larger than the page -> tiled (cut & join) as a last resort
Arrangement is not preserved; SIZES are true. marker_sizes.yaml holds black-square m.

Print at 100% (actual size). Trim to the corner ticks (= total print size).
"""
import argparse, os, tempfile, math
import cv2, numpy as np, yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

DOCK_MARKERS = [(201,200.0),(202,200.0),(301,100.0),
                (302,60.0),(303,60.0),(304,60.0),(305,60.0),(401,47.5),(402,47.5)]
BORDER_RATIO = 0.7782

def get_dict(name):
    k=getattr(cv2.aruco,name)
    try: return cv2.aruco.getPredefinedDictionary(k)
    except AttributeError: return cv2.aruco.Dictionary_get(k)

def gen(d,mid,px):
    try: return cv2.aruco.generateImageMarker(d,mid,px)
    except AttributeError: return cv2.aruco.drawMarker(d,mid,px)

def verify(d,img,mid):
    p=cv2.copyMakeBorder(img,40,40,40,40,cv2.BORDER_CONSTANT,value=255)
    try: _,ids,_=cv2.aruco.ArucoDetector(d).detectMarkers(p)
    except AttributeError: _,ids,_=cv2.aruco.detectMarkers(p,d)
    assert ids is not None and mid in ids.flatten(), f"marker {mid} decode fail"

def full_tile(d,mid,total_mm,dpi,ratio):
    tp=max(200,int(round(total_mm/25.4*dpi))); bp=max(80,int(round(tp*ratio)))
    m=gen(d,mid,bp); img=np.full((tp,tp),255,np.uint8); o=(tp-bp)//2
    img[o:o+bp,o:o+bp]=m; verify(d,img,mid); return img

def ticks(c,x,y,w,h,t=4):
    for (px,py,dx,dy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        c.line(px,py,px+dx*t*mm,py); c.line(px,py,px,py+dy*t*mm)

def build(out_pdf,dict_name,markers,dpi=300,margin_mm=10.0,gap_mm=10.0,label_mm=7.0,
          page_safety_mm=4.0,ratio=BORDER_RATIO):
    d=get_dict(dict_name); pw,ph=A4
    pw_mm,ph_mm=pw/mm,ph/mm
    usable_w=pw-2*margin_mm*mm                       # packable area
    usable_w_mm=usable_w/mm
    page_fit_mm=pw_mm-2*page_safety_mm               # max total that fits whole on a page
    c=canvas.Canvas(out_pdf,pagesize=A4); tmp=tempfile.mkdtemp(); sizes={}

    def hdr(_txt=""):
        return

    pack=[(i,s) for i,s in markers if s<=usable_w_mm]
    own =[(i,s) for i,s in markers if usable_w_mm<s<=page_fit_mm and s<=ph_mm-30]
    tile=[(i,s) for i,s in markers if s>page_fit_mm or s>ph_mm-30]

    # ---- whole markers on their own centred page (e.g. 200 mm) ----
    for mid,total_mm in own:
        sizes[int(mid)]=round(total_mm*ratio/1000.0,5)
        png=os.path.join(tmp,f"m{mid}.png"); cv2.imwrite(png,full_tile(d,mid,total_mm,dpi,ratio))
        hdr()
        w=total_mm*mm
        x=(pw-w)/2; y=ph-(margin_mm+12)*mm-w
        c.drawImage(png,x,y,width=w,height=w)
        c.setFont("Helvetica",8); c.drawString(x,y-6*mm,f"ID {mid} · {total_mm:g} mm")
        c.showPage()

    # ---- shelf-pack the small markers ----
    fit=sorted(pack,key=lambda m:-m[1]); started=False
    x=margin_mm*mm; y_top=ph-(margin_mm+8)*mm; row_h=0.0
    def newpage():
        nonlocal x,y_top,row_h
        hdr()
        x=margin_mm*mm; y_top=ph-(margin_mm+12)*mm; row_h=0.0
    def newline():
        nonlocal x,y_top,row_h
        y_top-=(row_h+label_mm*mm+gap_mm*mm); x=margin_mm*mm; row_h=0.0
    if fit: newpage(); started=True
    for mid,total_mm in fit:
        sizes[int(mid)]=round(total_mm*ratio/1000.0,5); w=total_mm*mm
        if x+w>margin_mm*mm+usable_w+1e-6: newline()
        if (y_top-w)<margin_mm*mm+label_mm*mm: c.showPage(); newpage()
        png=os.path.join(tmp,f"m{mid}.png"); cv2.imwrite(png,full_tile(d,mid,total_mm,dpi,ratio))
        yb=y_top-w; c.drawImage(png,x,yb,width=w,height=w)
        c.setFont("Helvetica",8); c.drawString(x,yb-6*mm,f"ID {mid} · {total_mm:g} mm")
        x+=w+gap_mm*mm; row_h=max(row_h,w)
    if started: c.showPage()

    # ---- genuinely-too-big -> tile (none for the dock set) ----
    for mid,total_mm in tile:
        sizes[int(mid)]=round(total_mm*ratio/1000.0,5)
        img=full_tile(d,mid,total_mm,dpi,ratio); H,W=img.shape
        cols=math.ceil(total_mm/usable_w_mm); rows=math.ceil(total_mm/(ph_mm-30))
        for r in range(rows):
            for col in range(cols):
                sub=img[int(r*H/rows):int((r+1)*H/rows),int(col*W/cols):int((col+1)*W/cols)]
                p=os.path.join(tmp,f"m{mid}_{r}_{col}.png"); cv2.imwrite(p,sub)
                hdr()
                w=(total_mm/cols)*mm; h=(total_mm/rows)*mm
                x=margin_mm*mm; y=ph-(margin_mm+12)*mm-h
                c.drawImage(p,x,y,width=w,height=h)
                c.showPage()
    c.save()
    sc=os.path.join(os.path.dirname(out_pdf) or ".","marker_sizes.yaml")
    with open(sc,"w") as f:
        yaml.safe_dump({"dictionary":dict_name,"note":"black-square side length (m)",
                        "marker_size_m":sizes},f,sort_keys=True)
    return sc

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="aruco_collage_A4.pdf")
    ap.add_argument("--dict",default="DICT_ARUCO_ORIGINAL"); a=ap.parse_args()
    print("wrote",a.out,"and",build(a.out,a.dict,DOCK_MARKERS))
