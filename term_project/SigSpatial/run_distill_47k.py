#!/usr/bin/env python3
"""47K SELF-CONTAINED queries-only, WJ-native Matryoshka distillation. FAST version:
- precompute the corpus-corpus raw-WJ matrix ONCE (37K x 37K ~ 5.6 GB) -> index targets per
  batch (no per-step recompute)
- DataParallel over all GPUs (now that the per-step raw-WJ bottleneck is gone)
Files: /tmp/qt_queries_only.npy (46754,18220 raw); query_start=37403; corpus=0..37402;
queries=37403..46753; eval GT /tmp/gt_lookup_queries_only.pkl (official).
ONE 4096-d model; WJ-distillation MSE on prefixes {256,512,1024,2048,4096}; train on
corpus-internal neighbor pairs (queries held out); eval held-out queries vs 47K corpus:
WJ HNSW + rerank + QPS per dim, recall vs official GT."""
import sys, os, time, csv, datetime, random, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import l1_simplex, nmslib_neighbors, rerank_wj_gpu, preload_rerank_corpus, release_rerank_corpus
DEV=torch.device('cuda:0'); THREADS=64; torch.set_num_threads(THREADS)
PREFIXES=[256,512,1024,2048,4096]; EMB=4096
B=512; EPOCHS=10; LR=1e-3; WD=1e-4; MAXPOS=30
EF=200; RB=64; CAND_KS=[500,1000]; SEED=42; QS=37403
CSV="/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def emb_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)
@torch.no_grad()
def corpus_W(Cr,rc=128,cc=1024):              # full corpus-corpus raw-WJ matrix on GPU (nC,nC)
    nC=Cr.shape[0]; W=torch.empty(nC,nC,device=DEV); t0=time.time()
    for ci,i in enumerate(range(0,nC,rc)):
        a=Cr[i:i+rc]
        for j in range(0,nC,cc):
            b=Cr[j:j+cc]
            W[i:i+rc,j:j+cc]=torch.minimum(a.unsqueeze(1),b.unsqueeze(0)).sum(2)/torch.maximum(a.unsqueeze(1),b.unsqueeze(0)).sum(2).clamp(min=1e-10)
        if ci%30==0: print(f"  WJ matrix {min(i+rc,nC)}/{nC} ({(time.time()-t0)/60:.1f}min)",flush=True)
    return W

class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
    def forward(s,x): return F.relu(s.encoder(x))
def module(m): return m.module if hasattr(m,'module') else m
class PairDS(Dataset):
    def __init__(s,knn):
        s.p=[(i,int(j)) for i in range(knn.shape[0]) for j in knn[i]]
        random.shuffle(s.p); print(f"pairs={len(s.p):,} steps/epoch~{len(s.p)//B}",flush=True)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def recall_local(GTlist,nbrs,ks=(50,500)):
    res={k:[] for k in ks}
    for q in range(len(nbrs)):
        truth=GTlist[q]
        if not truth: continue
        r=list(nbrs[q][0] if isinstance(nbrs[q],tuple) else nbrs[q])
        for k in ks:
            c=set(truth[:k]); res[k].append(len(c & set(r[:k]))/len(c))
    return {k:float(np.mean(v)) for k,v in res.items()}

def main():
    raw=np.load('/tmp/qt_queries_only.npy')
    QT=np.ascontiguousarray(raw,dtype=np.float32); QTN=l1_simplex(QT.copy())
    gtl=pickle.load(open('/tmp/gt_lookup_queries_only.pkl','rb'))
    Cn=QTN[:QS]; Cr=QT[:QS]; Qn=QTN[QS:]; Qr=QT[QS:]
    GTlist=[list(gtl.get(QS+q,[])) for q in range(len(Qn))]
    print(f"47K queries-only: corpus={len(Cn)} query={len(Qn)} dim={QT.shape[1]}",flush=True)

    print("precomputing corpus-corpus WJ matrix (one-time)...",flush=True); t0=time.time()
    W=corpus_W(torch.from_numpy(Cr).to(DEV))                       # (nC,nC) GPU ~5.6GB
    knn=W.topk(MAXPOS+1,dim=1).indices[:,1:].cpu().numpy()         # self at col0 dropped
    print(f"  WJ matrix + kNN done {(time.time()-t0)/60:.1f}min",flush=True)
    Cn_g=torch.from_numpy(Cn).to(DEV)

    model=Net(QT.shape[1]).to(DEV)
    if torch.cuda.device_count()>1: model=nn.DataParallel(model); print(f"DataParallel {torch.cuda.device_count()} GPUs",flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    loader=DataLoader(PairDS(knn),batch_size=B,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); best=1e9; t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); tl=st=0
        for ai,pi in tqdm(loader,desc=f"ep{ep:02d}/{EPOCHS}",ncols=100,mininterval=10):
            ids=torch.cat([ai,pi]).to(DEV)
            z=model(Cn_g[ids])
            tw=W[ids][:,ids]                                       # indexed target (no recompute)
            loss=sum(F.mse_loss(emb_wj(norm(z[:,:k]),norm(z[:,:k])),tw) for k in PREFIXES)/len(PREFIXES)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item(); st+=1
        sch.step(); print(f"ep{ep:02d} mse={tl/st:.6f} {(time.time()-t0)/60:.1f}min",flush=True)
        if tl/st<best: best=tl/st; torch.save(module(model).state_dict(),'/tmp/best_distill_47k.pt')
    del W; torch.cuda.empty_cache()

    # --- eval: per prefix, WJ HNSW + rerank + QPS, held-out queries vs OFFICIAL GT ---
    enc=module(model); enc.eval()
    with torch.no_grad():
        ZC=torch.cat([enc(Cn_g[i:i+1024]) for i in range(0,len(Cn),1024)])
        ZQ=torch.cat([enc(torch.from_numpy(Qn[i:i+1024]).to(DEV)) for i in range(0,len(Qn),1024)])
    Cr_sum=Cr.sum(1).astype(np.float32); today=datetime.date.today().isoformat(); rows=[]
    for k in PREFIXES:
        ce=norm(ZC[:,:k]).cpu().numpy().astype(np.float32); qe=norm(ZQ[:,:k]).cpu().numpy().astype(np.float32)
        mk=max(max(CAND_KS),500)
        nb,info=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=mk,threads=THREADS,query_params={"efSearch":EF})
        base=recall_local(GTlist,nb)
        print(f"[d={k} base] R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}",flush=True)
        rows.append([k,"base","",base[50],base[500],round(info['qps'])])
        preload_rerank_corpus(Cr,Cr_sum)
        for ck in CAND_KS:
            cand,ci=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=ck,threads=THREADS,query_params={"efSearch":EF})
            t1=time.time(); rr=rerank_wj_gpu(Qr,cand,Cr,Cr_sum,top_k=ck,batch_size=RB); e2e=len(Qr)/max(ci["query_s"]+(time.time()-t1),1e-9)
            mm=recall_local(GTlist,rr)
            print(f"[d={k} rerank K={ck}] R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",flush=True)
            rows.append([k,"rerank",ck,mm[50],mm[500],round(e2e)])
        release_rerank_corpus()
    note=f"47K queries-only distill (corpus={len(Cn)} query={len(Qn)}) WJ-native prefixes={PREFIXES}; official GT; {THREADS}thr"
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        for k,stage,ck,r50,r500,qps in rows:
            w.writerow([today,"47k_qonly",f"Distill47K-d{k}",k,stage,ck,"",round(r50,4),"",round(r500,4),qps,"run_distill_47k.py",note])
    print(f"appended {len(rows)} rows -> {CSV}",flush=True)

if __name__=='__main__': main()
