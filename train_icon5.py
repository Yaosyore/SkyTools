import os, json
import numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
from recognizer import IconNet, imread_unicode

d=np.load(os.path.join(os.path.dirname(__file__),"icons5.npz"))
X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
print("train set",X.shape,Counter:=None)
from collections import Counter
print(Counter(y.tolist()))
# class balance to 40 each with augmentation
def aug(c, tag):
    out=[]
    for i in range(40):
        src=c[i%len(c)]
        M=cv2.getRotationMatrix2D((24,24),np.random.uniform(-8,8),1.0)
        im=cv2.warpAffine(src,M,(48,48),borderMode=cv2.BORDER_REPLICATE)
        dx,dy=np.random.randint(-4,5,2)
        M2=np.float32([[1,0,dx],[0,1,dy]])
        im=cv2.warpAffine(im,M2,(48,48),borderMode=cv2.BORDER_REPLICATE)
        im=np.clip(im*np.random.uniform(0.85,1.15),0,255)
        out.append(im)
    return np.array(out)
slot2idx={0:0,1:1,2:2,4:3,5:4}  # slot -> cnn class index
cat=[]
labels=[]
for cls in [0,1,2,4,5]:
    sel=X[y==cls]
    cat.append(aug(sel,cls)); labels += [slot2idx[cls]]*40
Xa=np.concatenate(cat,axis=0); ya=np.array(labels)
print("aug",Xa.shape,Counter(ya.tolist()))
torch.manual_seed(0)
# split train/val per class
def split(X,y,valn=6):
    trx,trv,try_,trv_=[],[],[],[]
    for cls in sorted(set(y.tolist())):
        idx=np.where(y==cls)[0]
        np.random.seed(0); np.random.shuffle(idx)
        trx.append(X[idx[valn:]]); trv.append(X[idx[:valn]])
        try_+= [cls]*(len(idx)-valn); trv_+= [cls]*valn
    return np.concatenate(trx),np.concatenate(trv),np.array(try_),np.array(trv_)
Xtr,Xva,ytr,yva=split(Xa,ya,valn=6)
print("tr",Xtr.shape,"va",Xva.shape)
def to_t(x):
    x=x[...,::-1].transpose(0,3,1,2)/255.0
    return torch.tensor(x)
net=IconNet()
opt=optim.Adam(net.parameters(),lr=1e-3)
lossf=nn.CrossEntropyLoss()
Xtr_t=to_t(Xtr); ytr_t=torch.tensor(ytr)
Xva_t=to_t(Xva); yva_t=torch.tensor(yva)
best=0
for ep in range(60):
    net.train(); opt.zero_grad()
    out=net(Xtr_t); loss=lossf(out,ytr_t); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc=(net(Xva_t).argmax(1)==yva_t).float().mean().item()
        acc_t=(net(Xtr_t).argmax(1)==ytr_t).float().mean().item()
    if ep%5==0 or ep==59: print(f"ep{ep} tracc={acc_t:.3f} vaacc={acc:.3f} loss={loss.item():.3f}")
    if acc>=best: best=acc; torch.save(net.state_dict(),os.path.join(os.path.dirname(__file__),"icon_cnn5.pt"))
json.dump({"slot_map":[0,1,2,4,5]},open(os.path.join(os.path.dirname(__file__),"icon_cnn5_meta.json"),"w",encoding="utf-8"))
print("BEST val acc",best)
