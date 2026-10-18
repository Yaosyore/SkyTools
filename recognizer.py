# -*- coding: utf-8 -*-
"""光遇资源识别核心（ONNX 推理版，无 torch 依赖）
图标CNN: icon_cnn5.onnx (5类: 白蜡/爱心/季蜡/红蜡/代币)
数字CNN: digit_cnn.onnx (0-9，用于1~2位数字，3位回退RapidOCR)
"""
import os, sys, json, re, datetime
import numpy as np, cv2
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR

SLOT_NAMES = ["白蜡","爱心","季蜡","季心","红蜡","代币"]

def imread_unicode(p):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

class ResourceRecognizer:
    def __init__(self, model_dir):
        self.ocr = RapidOCR(text_score=0.05)
        self._sess_opt = ort.SessionOptions()
        self._sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._providers = ["CPUExecutionProvider"]
        self.icon_sess = ort.InferenceSession(
            os.path.join(model_dir, "icon_cnn5.onnx"), self._sess_opt, providers=self._providers)
        self.digit_sess = ort.InferenceSession(
            os.path.join(model_dir, "digit_cnn.onnx"), self._sess_opt, providers=self._providers)
        mpath = os.path.join(model_dir,"icon_cnn5_meta.json")
        self.slot_map = [0,1,2,4,5]
        if os.path.exists(mpath):
            meta = json.load(open(mpath, encoding="utf-8"))
            self.slot_map = meta.get("slot_map", [0,1,2,4,5])
        self.cal_act, self.cal_sea = self._load_cal(model_dir)
    def _load_cal(self, md):
        acts=[]; seas=[]
        cands=[os.path.join(md,"calendar.txt"),
               r"C:\Users\24223\Desktop\SkyTools\SkyDataTool\Settings\calendar.txt"]
        p=next((c for c in cands if os.path.exists(c)), None)
        if p:
            for line in open(p,encoding="utf-8"):
                line=line.strip()
                if not line or line.startswith("#"): continue
                parts=line.split("|")
                if len(parts)>=4:
                    try:
                        d0=datetime.date.fromisoformat(parts[2]); d1=datetime.date.fromisoformat(parts[3])
                        if parts[0]=="活动": acts.append((d0,d1))
                        elif parts[0]=="季节": seas.append((d0,d1))
                    except Exception: pass
        return acts, seas
    def _date_filter(self, counts, path):
        """按文件名日期过滤：非活动期代币置0；非季节期季蜡/季心置0。无日期则不限定。"""
        acts, seas = self.cal_act, self.cal_sea
        if not acts and not seas: return counts
        m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', os.path.basename(path))
        if not m: return counts
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception: return counts
        if acts and not any(a0 <= d <= a1 for a0,a1 in acts):
            counts[5] = 0
        if seas and not any(s0 <= d <= s1 for s0,s1 in seas):
            counts[2] = 0; counts[3] = 0
        return counts
    def _crop_icon(self, img, box):
        h,w = img.shape[:2]
        x0,y0,x1,y1 = [int(v) for v in box]
        bh = y1-y0
        iw = int(bh*2.0)
        ix0 = max(0, x0-iw); ix1 = x0
        iy0 = max(0, y0-int(bh*0.8)); iy1 = min(h, y1+int(bh*0.8))
        return img[iy0:iy1, ix0:ix1]
    def _classify(self, crop):
        if crop is None or crop.size == 0: return None, 0.0
        try:
            r = cv2.resize(crop, (48,48), interpolation=cv2.INTER_AREA)
        except Exception:
            return None, 0.0
        t = r[:,:,::-1].transpose(2,0,1).astype(np.float32)[None] / 255.0
        logits = self.icon_sess.run(None, {"input": t})[0][0]
        cls = int(np.argmax(logits)); conf = float(np.max(logits))
        return self.slot_map[cls], conf
    def _split_digits(self, boximg):
        """按亮阈值投影把数字区切成单数字（左->右）。"""
        g = cv2.cvtColor(boximg, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
        proj = bw.sum(axis=0)/255.0
        cols = np.where(proj>0)[0]
        if len(cols) == 0: return []
        segs = []; s = cols[0]; p = cols[0]
        for c in cols[1:]:
            if c-p > 2:
                segs.append((s,p)); s = c
            p = c
        segs.append((s,p))
        out = []
        for a,b in segs:
            if b-a+1 < 4: continue
            c = boximg[:, a:b]
            ch,cw = c.shape[:2]
            side = max(ch,cw); sq = np.full((side,side,3), 40, dtype=np.uint8)
            yo = (side-ch)//2; xo = (side-cw)//2
            sq[yo:yo+ch, xo:xo+cw] = c
            out.append(cv2.resize(sq,(28,28),interpolation=cv2.INTER_AREA))
        return out
    def _digit_cls(self, crop):
        r = cv2.resize(crop,(28,28),interpolation=cv2.INTER_AREA)
        t = r[:,:,::-1].transpose(2,0,1).astype(np.float32)[None] / 255.0
        logits = self.digit_sess.run(None, {"input": t})[0][0]
        return int(np.argmax(logits))
    def _read_number(self, boximg, ocr_text):
        """1~2位数字用数字CNN，否则回退RapidOCR文本。"""
        ot = str(ocr_text)
        crops = self._split_digits(boximg)
        if 1 <= len(crops) <= 2 and len(crops) == len(ot):
            try:
                digits = [self._digit_cls(c) for c in crops]
                cnn_val = int("".join(str(d) for d in digits))
                # 1位数字时如果CNN和OCR不一致，信OCR
                if len(ot) == 1 and str(cnn_val) != ot:
                    return int(ot)
                return cnn_val
            except Exception:
                pass
        return int(ot)
    def _ocr_all(self, img):
        """多尺度OCR，返回 (文本, 原图坐标框, 置信度)。"""
        h,w = img.shape[:2]
        variants = [(img,1.0)]
        variants.append((cv2.resize(img,(w*2,h*2),interpolation=cv2.INTER_CUBIC), 2.0))
        out = []
        for im, s in variants:
            res, _ = self.ocr(im)
            if not res: continue
            for box,text,conf in res:
                text = str(text).strip()
                if not text.isdigit(): continue
                xs = [pt[0]/s for pt in box]; ys = [pt[1]/s for pt in box]
                out.append((text, (min(xs),min(ys),max(xs),max(ys)), float(conf)))
        out.sort(key=lambda t: t[1][0])
        merged = []
        for text,bx,conf in out:
            dup = None
            for k,(mt,mb,mc) in enumerate(merged):
                ox = min(bx[2],mb[2])-max(bx[0],mb[0]); oy = min(bx[3],mb[3])-max(bx[1],mb[1])
                if ox>0 and oy>0:
                    inter = ox*oy
                    small = min((bx[2]-bx[0])*(bx[3]-bx[1]),(mb[2]-mb[0])*(mb[3]-mb[1]))
                    if small>0 and inter/small>0.3:
                        dup = k; break
            if dup is None:
                merged.append([text,bx,conf])
            else:
                if conf > merged[dup][2]:
                    merged[dup] = [text,bx,conf]
        return [(t,b,c) for t,b,c in merged]
    def predict(self, path):
        """识别6槽位。非季心槽位沿用ET4(图标CNN 5类，每槽取最高分)；
        季心槽位用"剔除已识别"法：季蜡锚点(x)与红蜡锚点(x)之间的多余数字框即季心。"""
        img = imread_unicode(path)
        if img is None: raise ValueError("无法读取图片: "+path)
        h, w = img.shape[:2]
        # 检测区域 (150,0) 到 (1400,250)
        x0, y0, x1, y1 = 150, 0, 1400, 250
        img = img[y0:min(y1,h), x0:min(x1,w)]
        h, w = img.shape[:2]
        counts = [0]*6; seen = {}
        boxes = []  # (xmid, value, ocr_conf, cnn_slot, icon_conf)
        for text, box, conf in self._ocr_all(img):
            slot, sconf = self._classify(self._crop_icon(img, box))
            xmid = (box[0]+box[2])/2.0/w
            value = None
            if slot is not None and slot != 3:
                x0,y0,x1,y1 = [int(v) for v in box]
                boximg = img[max(0,int(y0)-2):int(y1)+2, max(0,int(x0)-2):int(x1)+2]
                value = self._read_number(boximg, text)
                score = sconf * conf
                if slot not in seen or score > seen[slot][2]:
                    seen[slot] = (value, sconf, score, xmid)
            boxes.append((xmid, value, conf, slot, sconf))
        # 修正：slot0为空时，从所有boxes找xmid<0.25的框补到slot0
        if 0 not in seen:
            s0_cands = sorted([(xm,v,c,sf) for (xm,v,c,sf,sconf) in boxes if v is not None and xm < 0.25])
            if s0_cands:
                seen[0] = (s0_cands[0][1], s0_cands[0][3], s0_cands[0][2]*s0_cands[0][3], s0_cands[0][0])
        # 季心检测：季蜡(slot2)锚点x 与 红蜡(slot4)锚点x 之间的数字框 = 季心
        jila_x = seen[2][3] if 2 in seen else None
        hongla_x = seen[4][3] if 4 in seen else None
        if jila_x is not None and hongla_x is not None and jila_x < hongla_x:
            cand = [(x,v,sc) for (x,v,c,s_,sc) in boxes
                    if v is not None and jila_x < x < hongla_x and c > 0.25]
            if cand:
                cand.sort(key=lambda t: -t[2])
                counts[3] = cand[0][1]
        for s,(v,c,sc,x) in seen.items():
            counts[s] = v
        counts = self._date_filter(counts, path)
        return counts, w, h
