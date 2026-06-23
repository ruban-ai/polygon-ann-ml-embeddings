"""Stage 2: WHY does triplet cap at R@500~0.65 (below random proj 0.72)?
(a) add random-projection-512 to the exact-recall + eff-rank table.
(b) WJ-rank preservation: for sampled queries, take a candidate pool (true neighbors +
    random corpus), and measure Spearman( embedding-WJ , raw-18220-WJ ) -- how faithfully
    each embedding preserves the TRUE WJ ordering that recall depends on."""
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
def enc_random(o=512,bs=2048,seed=42):
    rng=np.random.default_rng(seed); W=torch.tensor((rng.standard_normal((qt.shape[1],o))/np.sqrt(o)).astype(np.float32),device=DEV)
    out=[]
    for i in range(0,len(qt),bs):
        x=torch.tensor(np.asarray(qt[i:i+bs],dtype=np.float32),device=DEV)
        out.append((x@W))
    Z=torch.cat(out).cpu().numpy(); return torch.tensor(shifted_l1_simplex(Z),device=DEV)

def wj_rows(A,B):  # exact WJ between rows of A (m,d) and B (n,d), both nonneg -> (m,n)
    mn=torch.minimum(A.unsqueeze(1),B.unsqueeze(0)).sum(2); mx=torch.maximum(A.unsqueeze(1),B.unsqueeze(0)).sum(2).clamp(min=1e-10)
    return mn/mx

def spearman(x,y):
    rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float()
    rx=rx-rx.mean(); ry=ry-ry.mean()
    return (rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)

@torch.no_grad()
def exact_knn_recall(emb,n_q=5000,maxk=500,qchunk=256):
    ce=emb[:qs]; qe=emb[qs:]; rng=np.random.RandomState(0)
    sel=rng.choice(len(qe),min(n_q,len(qe)),replace=False); sel.sort(); qids=[qs+int(i) for i in sel]
    res={50:[],500:[]}
    for s in range(0,len(sel),qchunk):
        idx=sel[s:s+qchunk]; q=qe[torch.tensor(idx,device=DEV)]
        top=torch.topk(torch.cdist(q,ce,p=1),maxk,dim=1,largest=False).indices.cpu().numpy()
        for j,gi in enumerate(idx):
            truth=gt.get(qs+int(gi),[])
            if not truth: continue
            for k in res: res[k].append(len(set(truth[:k])&set(top[j][:k].tolist()))/len(set(truth[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}

@torch.no_grad()
def eff_rank(emb,n=10000):
    ce=emb[:qs]; sel=np.random.RandomState(1).choice(len(ce),n,replace=False)
    X=ce[torch.tensor(sel,device=DEV)].float(); X=X-X.mean(0,keepdim=True); lam=torch.linalg.svdvals(X)**2
    return (lam.sum()**2/(lam**2).sum()).item()

# raw-WJ candidate pools: per query, 200 true neighbors (broad) + 800 random corpus
@torch.no_grad()
def wj_rank_preservation(emb_dict, n_q=400, n_true=200, n_rand=800):
    rng=np.random.RandomState(3); qsel=rng.choice(len(qq),n_q,replace=False)
    raw_c=torch.tensor(np.asarray(qt[:qs],dtype=np.float32))  # keep raw corpus on CPU, move slices
    out={name:[] for name in emb_dict}; out_auc={name:[] for name in emb_dict}
    for qi in qsel:
        gqid=qs+int(qi); truth=gt.get(gqid,[])
        if len(truth)<20: continue
        tr=np.array(truth[:n_true]); rd=rng.choice(qs,n_rand,replace=False)
        cand=np.concatenate([tr,rd]);
        # raw WJ (truth ranking reference)
        qraw=torch.tensor(np.asarray(qt[gqid],dtype=np.float32),device=DEV).unsqueeze(0)
        craw=torch.tensor(np.asarray(qt[cand],dtype=np.float32),device=DEV)
        raw_wj=wj_rows(qraw,craw)[0]
        is_true=torch.tensor([1.0]*len(tr)+[0.0]*len(rd),device=DEV)
        for name,emb in emb_dict.items():
            qe=emb[qs+int(qi)].unsqueeze(0); ce=emb[torch.tensor(cand,device=DEV)]
            ewj=wj_rows(qe,ce)[0]
            out[name].append(spearman(ewj,raw_wj).item())
            # AUC: P(embWJ(true) > embWJ(rand))
            t=ewj[is_true>0]; f=ewj[is_true==0]
            auc=(t.unsqueeze(1)>f.unsqueeze(0)).float().mean().item(); out_auc[name].append(auc)
    return {n:(float(np.mean(v)),float(np.mean(out_auc[n]))) for n,v in out.items()}

print('encoding...',flush=True)
embs={'triplet-2048':enc_learned('/tmp/best_triplet_autoencoder_wj_2048_full.pt',2048),
      'triplet-512':enc_learned('/tmp/best_sota_triplet_autoencoder_wj_512_full.pt',512),
      'infonce-512':enc_learned('/tmp/best_filter_recall_infonce_512_full.pt',512),
      'random-512':enc_random(512)}
print(f'\n{"config":<14}{"exactR@500":>11}{"effRank":>9}{"spearman(eWJ,rawWJ)":>21}{"AUC(true>rand)":>16}',flush=True)
pres=wj_rank_preservation(embs)
for name,emb in embs.items():
    r=exact_knn_recall(emb); pr=eff_rank(emb); sp,auc=pres[name]
    print(f'{name:<14}{r[500]:>11.4f}{pr:>9.1f}{sp:>21.4f}{auc:>16.4f}',flush=True)
print('\ndone',flush=True)
