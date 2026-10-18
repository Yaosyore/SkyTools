import os, json
import numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim

d=np.load(os.path.join(os.path.dirname(__file__),"digit_train.npz"))
X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
from collections import Counter
print("samples",Counter(y.tolist()))
torch.manual_seed(0)
def aug(c,cls):
    out=[]
    for i in range(120):
        src=c[i%len(c)]
        M=cv2.getRotationMatrix2D((14,14),np.random.uniform(-12,12),1.0)
        im=cv2.warpAffine(src,M,(28,28),borderMode=cv2.BORDER_REPLICATE)
        dx,dy=np.random.randint(-3,4,2)
        M2=np.float32([[1,0,dx],[0,1,dy]])
        im=cv2.warpAffine(im,M2,(28,28),borderMode=cv2.BORDER_REPLICATE)
        im=np.clip(im*np.random.uniform(0.8,1.2),0,255)
        out.append(im)
    return np.array(out)
cat=[];lab=[]
for cls in sorted(set(y.tolist())):
    sel=X[y==cls]; cat.append(aug(sel,cls)); lab+=[cls]*120
Xa=np.concatenate(cat); ya=np.array(lab)
def split(X,y,vn=20):
    trx,trv,try_,trv_=[],[],[],[]
    for cls in sorted(set(y.tolist())):
        idx=np.where(y==cls)[0]; np.random.seed(1); np.random.shuffle(idx)
        trx.append(X[idx[vn:]]); trv.append(X[idx[:vn]])
        try_+=[cls]*(len(idx)-vn); trv_+=[cls]*vn
    return np.concatenate(trx),np.concatenate(trv),np.array(try_),np.array(trv_)
Xtr,Xva,ytr,yva=split(Xa,ya)
def to_t(x): return torch.tensor(x[...,::-1].transpose(0,3,1,2)/255.0)
net=nn.Sequential(
    nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
    nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
    nn.Flatten(),nn.Linear(128*3*3,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,10))
opt=optim.Adam(net.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss()
Xt=to_t(Xtr); yt=torch.tensor(ytr); Xv=to_t(Xva); yv=torch.tensor(yva)
best=0
for ep in range(80):
    net.train(); opt.zero_grad(); o=net(Xt); l=lossf(o,yt); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pa=(net(Xv).argmax(1)==yv).float().mean().item(); pt=(net(Xt).argmax(1)==yt).float().mean().item()
    if ep%10==0 or ep==79: print(f"ep{ep} tracc={pt:.3f} vaacc={pa:.3f}")
    if pa>=best: best=pa; torch.save(net.state_dict(),os.path.join(os.path.dirname(__file__),"digit_cnn.pt"))
print("BEST val",best)
