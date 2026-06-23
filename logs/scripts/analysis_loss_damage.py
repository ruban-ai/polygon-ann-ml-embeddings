"""COMPREHENSIVE: does the loss damage the layer it's applied to? Map recall + WJ-metric
preservation across layers, vs a training-free random projection baseline."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized, shifted_l1_simplex
DEV=torch.device('cuda:0')
qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full'); IN=qtn.shape[1]
print(f'full: corpus={qs} queries={len(qq)} in={IN}',flush=True)
class Funnel(nn.Module):
    def __init__(s,o):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(IN,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),nn.Linear(1024,o,bias=False),nn.BatchNorm1d(o))
class Mat(nn.Module):
    def __init__(s):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(IN,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,4096,bias=False),nn.BatchNorm1d(4096))
def load(cls,ck,*a):
    m=cls(*a).to(DEV); sd=torch.load(ck,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval(); return m
@torch.no_grad()
def tap(m,upto,relu_last,bs=1024):
    out=[]
    for i in range(0,len(qtn),bs):
        h=m.encoder[:upto](torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV))
        if relu_last: h=F.relu(h)
        out.append(h/h.sum(1,keepdim=True).clamp(min=1e-10))
    return torch.cat(out)
@torch.no_grad()
def randproj(o,bs=2048,seed=42):
    rng=np.random.default_rng(seed); W=torch.tensor((rng.standard_normal((IN,o))/np.sqrt(o)).astype(np.float32),device=DEV)
    Z=torch.cat([torch.tensor(np.asarray(qt[i:i+bs],dtype=np.float32),device=DEV)@W for i in range(0,len(qt),bs)]).cpu().numpy()
    return torch.tensor(shifted_l1_simplex(Z),device=DEV)
def spear(x,y):
    rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float(); rx=rx-rx.mean(); ry=ry-ry.mean()
    return ((rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)).item()
@torch.no_grad()
def recall(emb,n_q=3000,maxk=500,qc=128):
    ce=emb[:qs]; qe=emb[qs:]; sel=np.random.RandomState(0).choice(len(qe),n_q,replace=False); sel.sort(); res={50:[],500:[]}
    for s in range(0,len(sel),qc):
        idx=sel[s:s+qc]; top=torch.topk(torch.cdist(qe[torch.tensor(idx,device=DEV)],ce,p=1),maxk,dim=1,largest=False).indices.cpu().numpy()
        for j,gi in enumerate(idx):
            t=gt.get(qs+int(gi),[])
            if not t: continue
            for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
@torch.no_grad()
def spd(emb,depths=(30,100,500),n_q=150,nr=500):
    rng=np.random.RandomState(3); acc={d:[] for d in depths}
    for qi in rng.choice(len(qq),n_q,replace=False):
        gq=qs+int(qi); t=gt.get(gq,[])
        if len(t)<max(depths): continue
        for d in depths:
            cand=np.concatenate([np.array(t[:d]),rng.choice(qs,nr,replace=False)])
            qr=torch.tensor(np.asarray(qt[gq],dtype=np.float32),device=DEV).unsqueeze(0); cr=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
            raw=torch.minimum(qr,cr).sum(1)/torch.maximum(qr,cr).sum(1).clamp(min=1e-10)
            qe=emb[gq].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
            ew=torch.minimum(qe,ce).sum(1)/torch.maximum(qe,ce).sum(1).clamp(min=1e-10)
            acc[d].append(spear(ew,raw))
    return {d:float(np.mean(v)) for d,v in acc.items()}

rows=[]
def add(label,dim,loss_here,emb):
    r=recall(emb); s=spd(emb); rows.append((label,dim,loss_here,r[500],s[30],s[100],s[500])); del emb; torch.cuda.empty_cache()
    print(f'  done: {label}',flush=True)

print('random projection refs...',flush=True)
for o in [1024,2048,4096]: add(f'random-proj',o,'no-train',randproj(o))
ft=load(Funnel,'/tmp/best_sota_triplet_autoencoder_wj_512_full.pt',512)
print('funnel-triplet-512 layers...',flush=True)
add('funnel-tri512: 4096',4096,'NO',tap(ft,3,True)); add('funnel-tri512: 1024',1024,'NO',tap(ft,6,True)); add('funnel-tri512: 512-OUT',512,'YES',tap(ft,8,True))
mt=load(Mat,'/tmp/best_matryoshka_triplet_full.pt')
print('matryoshka-triplet layers...',flush=True)
add('matry-tri: first-4096',4096,'NO',tap(mt,3,True)); add('matry-tri: out-4096',4096,'YES',tap(mt,5,True))
mi=load(Mat,'/tmp/best_matryoshka_infonce_full.pt')
print('matryoshka-infonce layers...',flush=True)
add('matry-inf: first-4096',4096,'NO',tap(mi,3,True)); add('matry-inf: out-4096',4096,'YES',tap(mi,5,True))

print('\n==================== LOSS-DAMAGE ANALYSIS (full set) ====================')
print(f'{"embedding / layer":<24}{"dim":>5}{"loss?":>6}{"R@500":>8}{"sp@30":>8}{"sp@100":>8}{"sp@500":>8}')
for label,dim,lh,r5,s30,s100,s500 in rows:
    print(f'{label:<24}{dim:>5}{lh:>6}{r5:>8.4f}{s30:>8.3f}{s100:>8.3f}{s500:>8.3f}')
print('done',flush=True)
