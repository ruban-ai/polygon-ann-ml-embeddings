"""Tap the INTERMEDIATE layers as embeddings: 4096-d (after relu1, BEFORE the 1024
bottleneck), 1024-d (after relu2), and the final output. L1-normalize each -> WJ embedding,
measure exact kNN recall + eff-rank + Spearman. Does the wider early layer hold more WJ info?"""
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
def load(ckpt,o):
    m=Enc(qtn.shape[1],o).to(DEV); sd=torch.load(ckpt,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval(); return m
@torch.no_grad()
def tap(m,upto,bs=1024):   # upto: # of encoder submodules to apply; output relu'd if last is ReLU
    out=[]
    for i in range(0,len(qtn),bs):
        x=torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV)
        h=m.encoder[:upto](x)
        if upto==8: h=F.relu(h)        # final output: encode() applies relu after BN
        z=h/h.sum(1,keepdim=True).clamp(min=1e-10)
        out.append(z)
    return torch.cat(out)
def spearman(x,y):
    rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float(); rx=rx-rx.mean(); ry=ry-ry.mean()
    return ((rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)).item()
@torch.no_grad()
def recall(emb,n_q=4000,maxk=500,qchunk=128):
    ce=emb[:qs]; qe=emb[qs:]; sel=np.random.RandomState(0).choice(len(qe),n_q,replace=False); sel.sort()
    res={50:[],500:[]}
    for s in range(0,len(sel),qchunk):
        idx=sel[s:s+qchunk]; top=torch.topk(torch.cdist(qe[torch.tensor(idx,device=DEV)],ce,p=1),maxk,dim=1,largest=False).indices.cpu().numpy()
        for j,gi in enumerate(idx):
            t=gt.get(qs+int(gi),[])
            if not t: continue
            for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
@torch.no_grad()
def eff(emb,n=10000):
    sel=np.random.RandomState(1).choice(qs,n,replace=False); X=emb[:qs][torch.tensor(sel,device=DEV)].float(); X=X-X.mean(0,keepdim=True)
    lam=torch.linalg.svdvals(X)**2; return (lam.sum()**2/(lam**2).sum()).item()
@torch.no_grad()
def sp500(emb,n_q=200,d=500,n_rand=500):
    rng=np.random.RandomState(3); acc=[]
    for qi in rng.choice(len(qq),n_q,replace=False):
        gq=qs+int(qi); t=gt.get(gq,[])
        if len(t)<d: continue
        cand=np.concatenate([np.array(t[:d]),rng.choice(qs,n_rand,replace=False)])
        qr=torch.tensor(np.asarray(qt[gq],dtype=np.float32),device=DEV).unsqueeze(0); cr=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
        raw=torch.minimum(qr,cr).sum(1)/torch.maximum(qr,cr).sum(1).clamp(min=1e-10)
        qe=emb[gq].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
        ew=torch.minimum(qe,ce).sum(1)/torch.maximum(qe,ce).sum(1).clamp(min=1e-10)
        acc.append(spearman(ew,raw))
    return float(np.mean(acc))

for mname,ck,o in [('InfoNCE-512','/tmp/best_filter_recall_infonce_512_full.pt',512),
                   ('triplet-512','/tmp/best_sota_triplet_autoencoder_wj_512_full.pt',512)]:
    m=load(ck,o)
    print(f'\n=== {mname} ===  (tap layers as WJ embedding)',flush=True)
    print(f'{"tap":<22}{"dim":>6}{"R@50":>8}{"R@500":>9}{"effRank":>9}{"sp@500":>8}',flush=True)
    for label,upto,dim in [('4096 (after relu1)',3,4096),('1024 (after relu2)',6,1024),(f'{o} output',8,o)]:
        e=tap(m,upto); r=recall(e); pr=eff(e); s=sp500(e)
        print(f'{label:<22}{dim:>6}{r[50]:>8.4f}{r[500]:>9.4f}{pr:>9.1f}{s:>8.3f}',flush=True)
        del e; torch.cuda.empty_cache()
print('\n(ref: random-4096 R@500=0.772 ; raw-18220 ceiling R@500=0.998)',flush=True)
print('done',flush=True)
