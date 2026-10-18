import os, json, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from recognizer import ResourceRecognizer, imread_unicode
rr=ResourceRecognizer(os.path.dirname(__file__))
mani=json.load(open(os.path.join(os.path.dirname(__file__),"manifest4.json"),encoding="utf-8"))

def split_digits(boximg, gtval):
    """split digit box image into single digits via vertical projection; return (crops, digits)."""
    g=cv2.cvtColor(boximg,cv2.COLOR_BGR2GRAY)
    # digits are bright; Otsu threshold
    _,bw=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    proj=bw.sum(axis=0)/255.0
    cols=np.where(proj>0)[0]
    if len(cols)==0: return [],str(gtval)
    segs=[]; s=cols[0]; p=cols[0]
    for c in cols[1:]:
        if c-p>2:
            segs.append((s,p)); s=c
        p=c
    segs.append((s,p))
    # merge tiny gaps
    merged=[]; cur=list(segs[0])
    for seg in segs[1:]:
        if seg[0]-cur[1]<=3: cur[1]=seg[1]
        else: merged.append(tuple(cur)); cur=list(seg)
    merged.append(tuple(cur))
    n=len(str(gtval))
    crops=[]
    for (a,b) in merged:
        w=b-a+1
        if w<4: continue
        c=boximg[:, a:b]
        # pad square
        h,w=c.shape[:2]
        side=max(h,w)
        sq=np.full((side,side,3),40,dtype=np.uint8)
        yo=(side-h)//2; xo=(side-w)//2
        sq[yo:yo+h, xo:xo+w]=c
        crops.append(cv2.resize(sq,(28,28),interpolation=cv2.INTER_AREA))
    return crops,str(gtval)

samples=[]; err=[]
D=os.path.join(os.path.dirname(__file__),"digit_train")
os.makedirs(D,exist_ok=True)
idx=0
for m in mani:
    im=imread_unicode(m["path"]); gt=m["counts"]
    items=rr._ocr_all(im)
    best={}
    for text,box,conf in items:
        crop=rr._crop_icon(im,box)
        s,sc=rr._classify(crop)
        if s is None: continue
        if s not in best or conf>best[s][0]: best[s]=(conf,text,box)
    for s,(conf,text,box) in best.items():
        if s==3: continue
        gv=gt[s]
        if gv<=0: continue
        x0,y0,x1,y1=[int(v) for v in box]
        boximg=im[max(0,y0-2):y1+2, max(0,x0-2):x1+2]
        crops,gtstr=split_digits(boximg,gv)
        if len(crops)!=len(gtstr):
            err.append((m["img"],s,gv,text,len(crops),len(gtstr)))
            continue
        for c,ch in zip(crops,gtstr):
            samples.append((c,int(ch),m["img"],s))
    idx+=1
print("split errors (n_crops != n_gt_digits):",len(err))
for e in err[:20]: print("  ",e)
print("digit samples:",len(samples))
from collections import Counter
print(Counter(s[1] for s in samples))
X=np.stack([s[0] for s in samples]).astype(np.float32)
y=np.array([s[1] for s in samples],dtype=np.int64)
np.savez(os.path.join(os.path.dirname(__file__),"digit_train.npz"),X=X,y=y)
json.dump([{"img":s[2],"slot":s[3]} for s in samples],open(os.path.join(os.path.dirname(__file__),"digit_train_meta.json"),"w",encoding="utf-8"))
print("saved digit_train.npz",X.shape)
