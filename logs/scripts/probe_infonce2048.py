"""Probe the MID-TRAINING InfoNCE-2048 (bottlenecked 18220->4096->1024->2048, ep11/18)
vs finished InfoNCE-512. Lower bound (not fully trained) + bottlenecked, but an early read."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized
DEV=torch.device('cuda:0')
qt,gt,qs,cq,qq,cs,qtn = load_dataset_normalized('full')
class Enc(nn.Module):
    def __init__(s,d,o):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),
            nn.Linear(1024,o,bias=False),nn.BatchNorm1d(o))
    def encode(s,x): z=F.relu(s.encoder(x)); return z/z.sum(1,keepdim=True).clamp(min=1e-10)
@torch.no_grad()
def enc(ckpt,o,bs=1024):
    m=Enc(qtn.shape[1],o).to(DEV); sd=torch.load(ckpt,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval()
    return torch.cat([m.encode(torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV)) for i in range(0,len(qtn),bs)])
def spearman(x,y):
    rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float(); rx=rx-rx.mean(); ry=ry-ry.mean()
    return ((rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)).item()
@torch.no_grad()
def exact_knn_recall(emb,n_q=5000,maxk=500,qchunk=256):
    ce=emb[:qs]; qe=emb[qs:]; sel=np.random.RandomState(0).choice(len(qe),min(n_q,len(qe)),replace=False); sel.sort()
    res={50:[],500:[]}
    for s in range(0,len(sel),qchunk):
        idx=sel[s:s+qchunk]; top=torch.topk(torch.cdist(qe[torch.tensor(idx,device=DEV)],ce,p=1),maxk,dim=1,largest=False).indices.cpu().numpy()
        for j,gi in enumerate(idx):
            t=gt.get(qs+int(gi),[])
            if not t: continue
            for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
@torch.no_grad()
def eff_rank(emb,n=10000):
    sel=np.random.RandomState(1).choice(qs,n,replace=False); X=emb[:qs][torch.tensor(sel,device=DEV)].float(); X=X-X.mean(0,keepdim=True)
    lam=torch.linalg.svdvals(X)**2; return (lam.sum()**2/(lam**2).sum()).item()
@torch.no_grad()
def sp_depth(emb,n_q=200,depths=(30,100,500),n_rand=500):
    rng=np.random.RandomState(3); qsel=rng.choice(len(qq),n_q,replace=False); acc={d:[] for d in depths}
    for qi in qsel:
        gq=qs+int(qi); t=gt.get(gq,[])
        if len(t)<max(depths): continue
        for d in depths:
            cand=np.concatenate([np.array(t[:d]),rng.choice(qs,n_rand,replace=False)])
            qr=torch.tensor(np.asarray(qt[gq],dtype=np.float32),device=DEV).unsqueeze(0); cr=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
            raw=torch.minimum(qr,cr).sum(1)/torch.maximum(qr,cr).sum(1).clamp(min=1e-10)
            qe=emb[gq].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
            ew=torch.minimum(qe,ce).sum(1)/torch.maximum(qe,ce).sum(1).clamp(min=1e-10)
            acc[d].append(spearman(ew,raw))
    return {d:float(np.mean(v)) for d,v in acc.items()}

print(f'{"embedding":<26}{"R@50":>8}{"R@500":>9}{"effRank":>9}{"sp@30":>8}{"sp@100":>8}{"sp@500":>8}',flush=True)
for name,ck,o in [('InfoNCE-512 (done)','/tmp/best_filter_recall_infonce_512_full.pt',512),
                  ('InfoNCE-2048 (ep11/18!)','/tmp/infonce2048_snapshot.pt',2048)]:
    e=enc(ck,o); r=exact_knn_recall(e); pr=eff_rank(e); sp=sp_depth(e)
    print(f'{name:<26}{r[50]:>8.4f}{r[500]:>9.4f}{pr:>9.1f}{sp[30]:>8.3f}{sp[100]:>8.3f}{sp[500]:>8.3f}',flush=True)
    del e; torch.cuda.empty_cache()
print('done',flush=True)
