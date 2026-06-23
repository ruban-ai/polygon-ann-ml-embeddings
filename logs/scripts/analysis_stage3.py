"""Stage 3 - PROVE it's embedding quality, find the cause, point to the fix.
  (A) CEILING: exact raw-18220-d WJ kNN recall vs GT (max any embedding could reach).
  (B) PROOF: random-projection (metric-preserving) dim sweep 256..4096 -- does recall RISE
      with dim? If yes for random but FLAT for learned triplet -> the limiter is the learned
      geometry, not the dimension.
  (C) CAUSE: Spearman(emb-WJ, raw-WJ) by neighborhood depth -- triplet only preserves the
      shallow neighborhood it was trained on (max_pos=30)."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized, shifted_l1_simplex
DEV=torch.device('cuda:0')
qt,gt,qs,cq,qq,cs,qtn = load_dataset_normalized('full')
print(f'corpus={qs} queries={len(qq)} dim={qtn.shape[1]}',flush=True)

class Enc(nn.Module):
    def __init__(s,d,o):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),
            nn.Linear(1024,o,bias=False),nn.BatchNorm1d(o))
    def encode(s,x): z=F.relu(s.encoder(x)); return z/z.sum(1,keepdim=True).clamp(min=1e-10)
@torch.no_grad()
def enc_learned(ckpt,o,bs=1024):
    m=Enc(qtn.shape[1],o).to(DEV); sd=torch.load(ckpt,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval()
    return torch.cat([m.encode(torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV)) for i in range(0,len(qtn),bs)])
@torch.no_grad()
def enc_random(o,bs=2048,seed=42):
    rng=np.random.default_rng(seed); W=torch.tensor((rng.standard_normal((qt.shape[1],o))/np.sqrt(o)).astype(np.float32),device=DEV)
    Z=torch.cat([ (torch.tensor(np.asarray(qt[i:i+bs],dtype=np.float32),device=DEV)@W) for i in range(0,len(qt),bs)]).cpu().numpy()
    return torch.tensor(shifted_l1_simplex(Z),device=DEV)

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
            truth=gt.get(qs+int(gi),[])
            if not truth: continue
            for k in res: res[k].append(len(set(truth[:k])&set(top[j][:k].tolist()))/len(set(truth[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
@torch.no_grad()
def eff_rank(emb,n=10000):
    sel=np.random.RandomState(1).choice(qs,n,replace=False); X=emb[:qs][torch.tensor(sel,device=DEV)].float(); X=X-X.mean(0,keepdim=True)
    lam=torch.linalg.svdvals(X)**2; return (lam.sum()**2/(lam**2).sum()).item()
@torch.no_grad()
def spearman_by_depth(emb,n_q=200,depths=(30,100,500),n_rand=500):
    rng=np.random.RandomState(3); qsel=rng.choice(len(qq),n_q,replace=False); acc={d:[] for d in depths}
    for qi in qsel:
        gqid=qs+int(qi); truth=gt.get(gqid,[])
        if len(truth)<max(depths): continue
        for d in depths:
            tr=np.array(truth[:d]); rd=rng.choice(qs,n_rand,replace=False); cand=np.concatenate([tr,rd])
            qraw=torch.tensor(np.asarray(qt[gqid],dtype=np.float32),device=DEV).unsqueeze(0); craw=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
            mn=torch.minimum(qraw,craw).sum(1); raw=mn/torch.maximum(qraw,craw).sum(1).clamp(min=1e-10)
            qe=emb[gqid].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
            mn2=torch.minimum(qe,ce).sum(1); ewj=mn2/torch.maximum(qe,ce).sum(1).clamp(min=1e-10)
            acc[d].append(spearman(ewj,raw))
    return {d:float(np.mean(v)) for d,v in acc.items()}

# (A) ceiling: raw 18220-d WJ kNN recall
@torch.no_grad()
def raw_ceiling(n_q=600,maxk=500):
    corpus=torch.tensor(np.asarray(qt[:qs],dtype=np.float32),device=DEV)
    sel=np.random.RandomState(0).choice(len(qq),n_q,replace=False); res={50:[],500:[]}
    for i in sel:
        gqid=qs+int(i); truth=gt.get(gqid,[])
        if not truth: continue
        q=torch.tensor(np.asarray(qt[gqid],dtype=np.float32),device=DEV)
        wj=torch.minimum(q,corpus).sum(1)/torch.maximum(q,corpus).sum(1).clamp(min=1e-10)
        top=torch.topk(wj,maxk,largest=True).indices.cpu().numpy()
        for k in res: res[k].append(len(set(truth[:k])&set(top[:k].tolist()))/len(set(truth[:k])))
    del corpus; torch.cuda.empty_cache(); return {k:float(np.mean(v)) for k,v in res.items()}

print('\n(A) CEILING -- exact raw 18220-d WJ kNN vs GT:',flush=True)
c=raw_ceiling(); print(f'   raw-WJ  R@50={c[50]:.4f}  R@500={c[500]:.4f}   <- max any {18220}->d embedding could reach',flush=True)

print('\n(B) does dimension help a METRIC-PRESERVING (random) embedding?',flush=True)
print(f'{"embedding":<16}{"dim":>6}{"R@500":>9}{"effRank":>9}{"sp(broad)":>11}',flush=True)
for d in [256,512,1024,2048,4096]:
    e=enc_random(d); r=exact_knn_recall(e); pr=eff_rank(e); sd=spearman_by_depth(e,depths=(500,))[500]
    print(f'{"random-proj":<16}{d:>6}{r[500]:>9.4f}{pr:>9.1f}{sd:>11.4f}',flush=True); del e; torch.cuda.empty_cache()

print('\n(C) learned embeddings + Spearman BY DEPTH (where does triplet break?):',flush=True)
print(f'{"embedding":<16}{"dim":>6}{"R@500":>9}{"sp@30":>8}{"sp@100":>8}{"sp@500":>8}',flush=True)
for name,ck,o in [('triplet',  '/tmp/best_sota_triplet_autoencoder_wj_512_full.pt',512),
                  ('triplet',  '/tmp/best_triplet_autoencoder_wj_2048_full.pt',2048),
                  ('infonce',  '/tmp/best_filter_recall_infonce_512_full.pt',512)]:
    e=enc_learned(ck,o); r=exact_knn_recall(e); sp=spearman_by_depth(e)
    print(f'{name:<16}{o:>6}{r[500]:>9.4f}{sp[30]:>8.3f}{sp[100]:>8.3f}{sp[500]:>8.3f}',flush=True); del e; torch.cuda.empty_cache()
print('\ndone',flush=True)
