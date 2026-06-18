#!/usr/bin/env python3
"""FULL-SET Matryoshka WJ-distillation: train on the 187K CORPUS (corpus-internal
neighbour pairs) so ALL 46,754 queries are held out, then evaluate ALL 46,754
queries vs the 187K corpus. DDP, per-batch raw-WJ target, prefixes {256..4096}.
Corpus self-kNN computed once on rank 0 (cdist, cached) and shared.
Launch: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 run_matdistill_fulleval_ddp.py"""
import sys, os, time, csv, datetime, random, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (load_dataset_normalized, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)

PREFIXES=[256,512,1024,2048,4096]; EMB=4096
B=512; EPOCHS=20; LR=1e-3; WD=1e-4; MAX_POS=30
EF=200; RB=64; CAND_KS=[1000,2000]; SEED=42
KNN_CACHE='/tmp/corpus_knn_full.npy'
CSV="/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def emb_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)
@torch.no_grad()
def raw_wj(R,chunk=64):
    M=R.shape[0]; out=torch.empty(M,M,device=R.device)
    for i in range(0,M,chunk):
        ri=R[i:i+chunk]
        mn=torch.minimum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2)
        mx=torch.maximum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2).clamp(min=1e-10)
        out[i:i+chunk]=mn/mx
    return out
def matry_distill(z,tw):
    tot=0.
    for m in PREFIXES: zm=norm(z[:,:m]); tot=tot+F.mse_loss(emb_wj(zm,zm),tw)
    return tot/len(PREFIXES)
@torch.no_grad()
def corpus_knn(Cn_g,k=MAX_POS,chunk=2048):     # nearest corpus neighbours by L1 (simplex) -> topk
    N=Cn_g.shape[0]; out=np.empty((N,k),dtype=np.int64)
    for i in range(0,N,chunk):
        a=Cn_g[i:i+chunk]; d=torch.cdist(a,Cn_g,p=1)
        r=torch.arange(a.shape[0],device=Cn_g.device)
        d[r,torch.arange(i,i+a.shape[0],device=Cn_g.device)]=1e9   # exclude self
        out[i:i+a.shape[0]]=torch.topk(d,k,dim=1,largest=False).indices.cpu().numpy()
    return out
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
    def forward(s,x): return F.relu(s.encoder(x))
class PairDS(Dataset):
    def __init__(s,knn):
        s.p=[(i,int(j)) for i in range(knn.shape[0]) for j in knn[i]]
        random.Random(SEED).shuffle(s.p)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def main():
    dist.init_process_group('nccl'); lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV=torch.device(f'cuda:{lr}'); main=(lr==0); world=dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full'); IN=qtn.shape[1]
    QTN=np.ascontiguousarray(qtn,dtype=np.float32); QT=np.ascontiguousarray(qt,dtype=np.float32)
    Cn_g=torch.from_numpy(QTN[:qs]).to(DEV)
    if main: print(f"DDP world={world} | corpus={qs} queries={len(qq)} dim={IN}",flush=True)

    # corpus self-kNN (rank 0 computes + caches; all load)
    if main:
        if os.path.exists(KNN_CACHE):
            knn=np.load(KNN_CACHE); print(f"loaded corpus kNN {knn.shape}",flush=True)
        else:
            t0=time.time(); knn=corpus_knn(Cn_g); np.save(KNN_CACHE,knn)
            print(f"corpus kNN {knn.shape} computed {(time.time()-t0)/60:.1f}min",flush=True)
    dist.barrier()
    knn=np.load(KNN_CACHE)

    model=DDP(Net(IN).to(DEV),device_ids=[lr])
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    ds=PairDS(knn); sampler=DistributedSampler(ds,shuffle=True,seed=SEED)
    loader=DataLoader(ds,batch_size=B,sampler=sampler,num_workers=4,drop_last=True,pin_memory=False)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); best=1e9; t0=time.time()
    if main: print(f"corpus pairs={len(ds):,} per-rank steps/epoch~{len(ds)//(world*B)}",flush=True)
    for ep in range(1,EPOCHS+1):
        model.train(); sampler.set_epoch(ep); tl=st=0
        it=tqdm(loader,desc=f"ep{ep:02d}/{EPOCHS}",ncols=100,mininterval=10) if main else loader
        for ai,pi in it:
            ids=torch.cat([ai,pi]).numpy()
            xb=torch.from_numpy(QTN[ids]).to(DEV); rb=torch.from_numpy(QT[ids]).to(DEV)
            z=model(xb)
            with torch.no_grad(): tw=raw_wj(rb)
            loss=matry_distill(z,tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item(); st+=1
        sch.step()
        if main:
            print(f"ep{ep:02d} mse={tl/st:.6f} {(time.time()-t0)/60:.1f}min",flush=True)
            if tl/st<best: best=tl/st; torch.save(model.module.state_dict(),'/tmp/best_matdistill_fulleval.pt')
    dist.barrier()
    if not main: dist.destroy_process_group(); return

    # ---- eval on rank 0: ALL 46,754 queries vs 187K corpus, raw-WJ GT ----
    enc=Net(IN).to(DEV); enc.load_state_dict(torch.load('/tmp/best_matdistill_fulleval.pt',map_location=DEV,weights_only=True)); enc.eval()
    THREADS=120; torch.set_num_threads(THREADS)
    with torch.no_grad():
        Z=torch.cat([enc(torch.from_numpy(QTN[i:i+512]).to(DEV)) for i in range(0,len(QTN),512)])
    today=datetime.date.today().isoformat(); rows=[]
    for m in PREFIXES:
        emb=norm(Z[:,:m]).cpu().numpy().astype(np.float32); ce=emb[:qs]; qe=emb[qs:]
        mk=max(max(CAND_KS),500)
        nb,info=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=mk,threads=THREADS,query_params={"efSearch":EF})
        base=eval_recall(gt,nb,qs,mk)
        print(f"[d={m} base ALLq] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}",flush=True)
        rows.append([m,"base","",base[10],base[50],base[100],base[500],round(info['qps'])])
        preload_rerank_corpus(cq,cs)
        for ck in CAND_KS:
            cand,ci=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=ck,threads=THREADS,query_params={"efSearch":EF})
            t1=time.time(); rr=rerank_wj_gpu(qq,cand,cq,cs,top_k=ck,batch_size=RB); e2e=len(qq)/max(ci["query_s"]+(time.time()-t1),1e-9)
            mm=eval_recall(gt,rr,qs,ck)
            print(f"[d={m} rerank K={ck} ALLq] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",flush=True)
            rows.append([m,"rerank",ck,mm[10],mm[50],mm[100],mm[500],round(e2e)])
        release_rerank_corpus()
    note=f"MatDistill full-eval ALL {len(qq)} queries, corpus-trained, prefixes={PREFIXES}; raw-WJ GT; efSearch={EF}"
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        for m,stage,ck,r10,r50,r100,r500,qps in rows:
            w.writerow([today,"full_allq",f"MatDistillFull-d{m}",m,stage,ck,round(r10,4),round(r50,4),round(r100,4),round(r500,4),qps,"run_matdistill_fulleval_ddp.py",note])
    print(f"appended {len(rows)} rows -> {CSV}",flush=True)
    dist.destroy_process_group()

if __name__=='__main__': main()
