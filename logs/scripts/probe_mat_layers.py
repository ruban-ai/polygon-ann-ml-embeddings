import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized
DEV=torch.device('cuda:0')
qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full')
class MatAE(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,4096,bias=False),nn.BatchNorm1d(4096))
def load(ck):
    m=MatAE(qtn.shape[1]).to(DEV); sd=torch.load(ck,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval(); return m
@torch.no_grad()
def tap(m,upto,bs=1024):
    out=[]
    for i in range(0,len(qtn),bs):
        h=m.encoder[:upto](torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV))
        if upto==5: h=F.relu(h)
        out.append(h/h.sum(1,keepdim=True).clamp(min=1e-10))
    return torch.cat(out)
def spear(x,y):
    rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float(); rx=rx-rx.mean(); ry=ry-ry.mean()
    return ((rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)).item()
@torch.no_grad()
def recall(emb,n_q=4000,maxk=500,qc=128):
    ce=emb[:qs]; qe=emb[qs:]; sel=np.random.RandomState(0).choice(len(qe),n_q,replace=False); sel.sort(); res={50:[],500:[]}
    for s in range(0,len(sel),qc):
        idx=sel[s:s+qc]; top=torch.topk(torch.cdist(qe[torch.tensor(idx,device=DEV)],ce,p=1),maxk,dim=1,largest=False).indices.cpu().numpy()
        for j,gi in enumerate(idx):
            t=gt.get(qs+int(gi),[])
            if not t: continue
            for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
@torch.no_grad()
def sp500(emb,n_q=150,d=500,nr=500):
    rng=np.random.RandomState(3); acc=[]
    for qi in rng.choice(len(qq),n_q,replace=False):
        gq=qs+int(qi); t=gt.get(gq,[])
        if len(t)<d: continue
        cand=np.concatenate([np.array(t[:d]),rng.choice(qs,nr,replace=False)])
        qr=torch.tensor(np.asarray(qt[gq],dtype=np.float32),device=DEV).unsqueeze(0); cr=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
        raw=torch.minimum(qr,cr).sum(1)/torch.maximum(qr,cr).sum(1).clamp(min=1e-10)
        qe=emb[gq].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
        ew=torch.minimum(qe,ce).sum(1)/torch.maximum(qe,ce).sum(1).clamp(min=1e-10)
        acc.append(spear(ew,raw))
    return float(np.mean(acc))
for name,ck in [('Matryoshka-triplet','/tmp/best_matryoshka_triplet_full.pt'),('Matryoshka-infonce','/tmp/best_matryoshka_infonce_full.pt')]:
    import os
    if not os.path.exists(ck): print(name,'MISSING'); continue
    m=load(ck); print(f'\n=== {name} ===')
    for lab,upto in [('first-4096 (NO direct loss)',3),('output-4096 (loss applied)',5)]:
        e=tap(m,upto); r=recall(e); print(f'  {lab:<28} R@500={r[500]:.4f}  sp@500={sp500(e):.3f}'); del e; torch.cuda.empty_cache()
print('done')
