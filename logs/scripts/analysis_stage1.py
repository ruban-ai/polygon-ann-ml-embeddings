"""Stage 1: is the ~0.66 base-recall ceiling the EMBEDDING or the HNSW index?
Compute EXACT brute-force WJ kNN recall (no HNSW) + effective rank, for the 3 learned
embeddings. If exact ~= HNSW, the embedding itself is the ceiling."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized

DEV=torch.device('cuda:0')  # set by CUDA_VISIBLE_DEVICES
qtn_qt = load_dataset_normalized('full')
qt,gt,qs,cq,qq,cs,qtn = qtn_qt
print(f'corpus={qs} queries={len(qq)} dim={qtn.shape[1]}',flush=True)

class Enc(nn.Module):
    def __init__(s,d,o):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),
            nn.Linear(1024,o,bias=False),nn.BatchNorm1d(o))
    def encode(s,x): z=F.relu(s.encoder(x)); return z/z.sum(1,keepdim=True).clamp(min=1e-10)

@torch.no_grad()
def encode_all(ckpt,o,bs=1024):
    m=Enc(qtn.shape[1],o).to(DEV)
    sd=torch.load(ckpt,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval()
    out=[]
    for i in range(0,len(qtn),bs):
        x=torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV); out.append(m.encode(x))
    return torch.cat(out)   # on GPU

def recall_at(qid_sample, nbr_idx, ks=(50,500)):
    # nbr_idx: (nq, maxk) corpus indices (best first). rank-matched recall vs gt[:k]
    res={k:[] for k in ks}
    for j,qid in enumerate(qid_sample):
        truth=gt.get(qid,[])
        if not truth: continue
        ret=nbr_idx[j]
        for k in ks:
            c=set(truth[:k]); r=set(ret[:k].tolist())
            res[k].append(len(c&r)/len(c))
    return {k:float(np.mean(v)) for k,v in res.items()}

@torch.no_grad()
def exact_knn_recall(emb, n_q=5000, maxk=500, qchunk=256):
    ce=emb[:qs]; qe=emb[qs:]
    rng=np.random.RandomState(0); sel=rng.choice(len(qe),min(n_q,len(qe)),replace=False); sel.sort()
    qids=[qs+int(i) for i in sel]
    nbrs=np.zeros((len(sel),maxk),dtype=np.int64)
    for s in range(0,len(sel),qchunk):
        idx=sel[s:s+qchunk]
        q=qe[torch.tensor(idx,device=DEV)]
        l1=torch.cdist(q,ce,p=1)                 # (c,C) ; smaller L1 = higher WJ
        top=torch.topk(l1,maxk,dim=1,largest=False).indices  # nearest = smallest L1
        nbrs[s:s+len(idx)]=top.cpu().numpy()
    return recall_at(qids,nbrs)

@torch.no_grad()
def eff_rank(emb, n=10000):
    ce=emb[:qs]; rng=np.random.RandomState(1); sel=rng.choice(len(ce),n,replace=False)
    X=ce[torch.tensor(sel,device=DEV)].float(); X=X-X.mean(0,keepdim=True)
    s=torch.linalg.svdvals(X)                    # singular values
    lam=(s**2)                                   # ~eigenvalues of covariance
    pr=(lam.sum()**2/(lam**2).sum()).item()      # participation ratio (1..d)
    csum=torch.cumsum(lam,0)/lam.sum()
    d90=int((csum<0.90).sum().item())+1
    return pr, d90, X.shape[1]

CFG=[('triplet-2048','/tmp/best_triplet_autoencoder_wj_2048_full.pt',2048,0.667),
     ('triplet-512','/tmp/best_sota_triplet_autoencoder_wj_512_full.pt',512,0.656),
     ('infonce-512','/tmp/best_filter_recall_infonce_512_full.pt',512,0.727)]
print(f'\n{"config":<14}{"dim":>5}{"exactR@50":>11}{"exactR@500":>12}{"HNSW_R@500":>12}{"effRank":>9}{"d@90%var":>9}',flush=True)
for name,ck,o,hnsw in CFG:
    emb=encode_all(ck,o)
    r=exact_knn_recall(emb)
    pr,d90,d=eff_rank(emb)
    print(f'{name:<14}{o:>5}{r[50]:>11.4f}{r[500]:>12.4f}{hnsw:>12.4f}{pr:>9.1f}{d90:>9}',flush=True)
    del emb; torch.cuda.empty_cache()
print('\ndone',flush=True)
