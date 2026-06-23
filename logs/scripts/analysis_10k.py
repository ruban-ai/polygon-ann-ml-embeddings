"""10K version: does the SAME bottleneck story explain the stagnant 10K recall across output
dims? For each 10K model: (a) output recall by dim, (b) tap 4096/1024/output layers."""
import sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized
DEV=torch.device('cuda:0')
qt,gt,qs,cq,qq,cs,qtn = load_dataset_normalized('10k')
IN=qtn.shape[1]; print(f'10K: corpus={qs} queries={len(qq)} in_dim={IN}',flush=True)
class Enc(nn.Module):
    def __init__(s,d,o):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),
            nn.Linear(1024,o,bias=False),nn.BatchNorm1d(o))
def out_dim(sd): return sd['encoder.6.weight'].shape[0]
def load(ck):
    sd=torch.load(ck,map_location=DEV,weights_only=True); o=out_dim(sd)
    m=Enc(IN,o).to(DEV); m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval(); return m,o
@torch.no_grad()
def tap(m,upto,bs=2048):
    out=[]
    for i in range(0,len(qtn),bs):
        h=m.encoder[:upto](torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV))
        if upto==8: h=F.relu(h)
        out.append(h/h.sum(1,keepdim=True).clamp(min=1e-10))
    return torch.cat(out)
def spearman(x,y):
    rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float(); rx=rx-rx.mean(); ry=ry-ry.mean()
    return ((rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)).item()
@torch.no_grad()
def recall(emb,maxk=500,qchunk=256):
    ce=emb[:qs]; qe=emb[qs:]; res={50:[],500:[]}
    for s in range(0,len(qe),qchunk):
        top=torch.topk(torch.cdist(qe[s:s+qchunk],ce,p=1),min(maxk,qs),dim=1,largest=False).indices.cpu().numpy()
        for j in range(len(top)):
            t=gt.get(qs+s+j,[])
            if not t: continue
            for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
@torch.no_grad()
def eff(emb,n=6000):
    sel=np.random.RandomState(1).choice(qs,min(n,qs),replace=False); X=emb[:qs][torch.tensor(sel,device=DEV)].float(); X=X-X.mean(0,keepdim=True)
    lam=torch.linalg.svdvals(X)**2; return (lam.sum()**2/(lam**2).sum()).item()
@torch.no_grad()
def sp500(emb,n_q=300,d=200,n_rand=400):
    rng=np.random.RandomState(3); acc=[]
    for qi in rng.choice(len(qq),min(n_q,len(qq)),replace=False):
        gq=qs+int(qi); t=gt.get(gq,[])
        if len(t)<30: continue
        dd=min(d,len(t)); cand=np.concatenate([np.array(t[:dd]),rng.choice(qs,n_rand,replace=False)])
        qr=torch.tensor(np.asarray(qt[gq],dtype=np.float32),device=DEV).unsqueeze(0); cr=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
        raw=torch.minimum(qr,cr).sum(1)/torch.maximum(qr,cr).sum(1).clamp(min=1e-10)
        qe=emb[gq].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
        ew=torch.minimum(qe,ce).sum(1)/torch.maximum(qe,ce).sum(1).clamp(min=1e-10)
        acc.append(spearman(ew,raw))
    return float(np.mean(acc))

CKPTS=[('triplet-256','/tmp/best_sota_triplet_autoencoder_wj_256_10k.pt'),
       ('triplet-1024','/tmp/best_sota_triplet_autoencoder_wj_1024_10k.pt'),
       ('AE-triplet','/tmp/best_autoencoder_triplet_10k.pt'),
       ('InfoNCE','/tmp/best_filter_recall_infonce_512_10k.pt')]
print(f'\n{"model":<14}{"out":>5}{"tap":<20}{"dim":>6}{"R@50":>8}{"R@500":>9}{"effRank":>9}{"sp@N":>7}',flush=True)
for name,ck in CKPTS:
    try: m,o=load(ck)
    except Exception as e: print(name,'SKIP',e); continue
    for label,upto,dim in [('4096(relu1)',3,4096),('1024(relu2)',6,1024),('output',8,o)]:
        e=tap(m,upto); r=recall(e); pr=eff(e); s=sp500(e)
        print(f'{name:<14}{o:>5}{label:<20}{dim:>6}{r[50]:>8.4f}{r[500]:>9.4f}{pr:>9.1f}{s:>7.3f}',flush=True)
        del e; torch.cuda.empty_cache()
    print('  ---')
print('done',flush=True)
