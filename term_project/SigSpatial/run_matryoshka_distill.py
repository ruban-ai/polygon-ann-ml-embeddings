#!/usr/bin/env python3
"""Matryoshka WJ-DISTILLATION on the FULL set (187K corpus, 18,220-d), WJ-native (zero cosine).
ONE 4096-d model; the WJ-distillation MSE is applied on nested prefixes {256,512,1024,2048,4096}
(each L1-renormalized) so truncating to ANY dim gives a deployable, metric-aligned embedding.
Target = raw 18,220-d WeightedJaccard (geometry); GT only selects informative batches.
80/20 query split -> trains on 80%, evals on held-out 20% with WJ HNSW + rerank + QPS per dim."""
import sys, os, time, csv, datetime, random, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (build_gt_cache, eval_recall, eval_recall_by_qids,
    load_dataset_normalized, nmslib_neighbors, preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu)

DEV=torch.device('cuda:0'); THREADS=120; torch.set_num_threads(THREADS)
PREFIXES=[256,512,1024,2048,4096]; EMB=4096
B=512; EPOCHS=20; LR=1e-3; WD=1e-4; MAX_POS=30
EF=200; RB=64; CAND_KS=[1000,2000]; SEED=42
SPLIT='/tmp/query_split_80_20.pkl'
CSV="/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def emb_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)
@torch.no_grad()
def raw_wj(R,chunk=64):                       # exact WJ on raw batch (no grad) -> (M,M) target
    M=R.shape[0]; out=torch.empty(M,M,device=R.device)
    for i in range(0,M,chunk):
        ri=R[i:i+chunk]
        mn=torch.minimum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2)
        mx=torch.maximum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2).clamp(min=1e-10)
        out[i:i+chunk]=mn/mx
    return out
def matry_distill(z,tw):                       # z:(M,4096) raw embed; tw:(M,M) raw-WJ target
    tot=0.
    for m in PREFIXES:
        zm=norm(z[:,:m]); tot=tot+F.mse_loss(emb_wj(zm,zm),tw)
    return tot/len(PREFIXES)

class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
    def forward(s,x): return F.relu(s.encoder(x))   # (.,4096) unnormalized; prefix-norm in loss/eval
def module(m): return m.module if hasattr(m,'module') else m

class DS(Dataset):
    def __init__(s,gt,qsv,allow):
        a=set(allow)
        s.p=[(q,n) for q,nb in gt.items() for n in nb[:MAX_POS] if q>=qsv and n<qsv and q in a]
        random.shuffle(s.p); print(f"pairs={len(s.p):,} steps/epoch~{len(s.p)//B}",flush=True)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def main():
    split=pickle.load(open(SPLIT,'rb')); TR=split['train_qids']; EV=np.array(split['eval_qids'],dtype=np.int64)
    print(f"split: train={len(TR):,} eval={len(EV):,}",flush=True)
    qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full'); IN=qtn.shape[1]
    QTN=np.ascontiguousarray(qtn,dtype=np.float32); QT=np.ascontiguousarray(qt,dtype=np.float32)  # CPU; move slices
    model=Net(IN).to(DEV)
    if torch.cuda.device_count()>1: model=nn.DataParallel(model); print(f"DataParallel {torch.cuda.device_count()} GPUs",flush=True)
    loader=DataLoader(DS(gt,qs,TR),batch_size=B,shuffle=True,num_workers=12,drop_last=True,persistent_workers=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); best=1e9; t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); tl=st=0
        for ai,pi in tqdm(loader,desc=f"ep{ep:02d}/{EPOCHS}",ncols=100,mininterval=10):
            ids=torch.cat([ai,pi]).numpy()
            xb=torch.from_numpy(QTN[ids]).to(DEV); rb=torch.from_numpy(QT[ids]).to(DEV)
            z=model(xb)                                   # (M,4096) DP-gathered on cuda:0
            with torch.no_grad(): tw=raw_wj(rb)
            loss=matry_distill(z,tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl+=loss.item(); st+=1
        sch.step(); print(f"ep{ep:02d} mse={tl/st:.6f} {(time.time()-t0)/60:.1f}min",flush=True)
        if tl/st<best: best=tl/st; torch.save(module(model).state_dict(),'/tmp/best_matdistill_train80.pt')

    # ---- eval: each prefix dim -> WJ HNSW + rerank + QPS, on HELD-OUT eval20 ----
    enc=module(model); enc.eval()
    with torch.no_grad():
        Z=torch.cat([enc(torch.from_numpy(QTN[i:i+512]).to(DEV)) for i in range(0,len(QTN),512)])  # (N,4096) GPU
    today=datetime.date.today().isoformat(); rows=[]; ei=EV; qq_e=qq[ei-qs]
    for m in PREFIXES:
        emb=norm(Z[:,:m]).cpu().numpy().astype(np.float32); ce=emb[:qs]; qe=emb[ei]
        mk=max(max(CAND_KS),500)
        nb,info=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=mk,threads=THREADS,query_params={"efSearch":EF})
        base=eval_recall_by_qids(gt,nb,list(ei),mk)
        print(f"[d={m} base eval20] R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}",flush=True)
        rows.append([m,"base","",base[10],base[50],base[100],base[500],round(info['qps'])])
        preload_rerank_corpus(cq,cs)
        for ck in CAND_KS:
            cand,ci=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=ck,threads=THREADS,query_params={"efSearch":EF})
            t1=time.time(); rr=rerank_wj_gpu(qq_e,cand,cq,cs,top_k=ck,batch_size=RB); e2e=len(qq_e)/max(ci["query_s"]+(time.time()-t1),1e-9)
            mm=eval_recall_by_qids(gt,rr,list(ei),ck)
            print(f"[d={m} rerank K={ck} eval20] R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",flush=True)
            rows.append([m,"rerank",ck,mm[10],mm[50],mm[100],mm[500],round(e2e)])
        release_rerank_corpus()
    note=f"MatDistill WJ-native eval20 n={len(ei)} prefixes={PREFIXES}; {THREADS}thr efSearch={EF}"
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        for m,stage,ck,r10,r50,r100,r500,qps in rows:
            w.writerow([today,"full",f"MatDistill-d{m}",m,stage,ck,round(r10,4),round(r50,4),round(r100,4),round(r500,4),qps,"run_matryoshka_distill.py",note])
    print(f"appended {len(rows)} rows -> {CSV}",flush=True)

if __name__=='__main__': main()
