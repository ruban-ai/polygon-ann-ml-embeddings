#!/usr/bin/env python3
"""Matryoshka WJ embedding: ONE 4096-d model, loss applied on nested prefixes {1024,2048,4096}
(each re-normalized to the L1 simplex). At eval, TRUNCATE to each dim -> HNSW(WJ) recall+QPS+
rerank, logged to NEW_RESULTS.csv. --loss triplet (max_pos=30) | infonce (max_pos=256).
No funnel: 18220 -> 4096 -> 4096 (so the wide embedding isn't bottlenecked)."""
import sys, os, time, csv, datetime, random, argparse, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (build_fn_mask, build_gt_cache, build_gt_gpu, eval_recall, eval_recall_by_qids,
    load_10k_train8k_normalized, load_dataset_normalized, nmslib_neighbors, preload_rerank_corpus,
    release_rerank_corpus, rerank_wj_gpu)

ap=argparse.ArgumentParser()
ap.add_argument('--loss',choices=['triplet','infonce'],required=True)
ap.add_argument('--dataset',choices=['full','10k-train8k'],default='full')
ap.add_argument('--split-file',default=None,help='e.g. /tmp/query_split_80_20.pkl (train80/eval20 of 47K queries)')
A=ap.parse_args()
DEV=torch.device('cuda:0'); THREADS=120; torch.set_num_threads(THREADS)
PREFIXES=[1024,2048,4096]; EMB=4096
BATCH=1024; LR=1e-3; WD=1e-4; MARGIN=0.3; TEMP=0.07; LAM_REC=0.1
EF=200; RB=64; CAND_KS=[1000,2000]; SEED=42
MAX_POS = 30 if A.loss=='triplet' else 256
EPOCHS  = 75 if A.loss=='triplet' else 18
if A.dataset == "10k-train8k":
    SPLIT_TAG = "10k-train8k"
elif A.split_file:
    SPLIT_TAG = "train80"
else:
    SPLIT_TAG = "full"
CKPT=f"/tmp/best_matryoshka_{A.loss}_{SPLIT_TAG}.pt"
CSV="/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print(f"loss={A.loss} max_pos={MAX_POS} epochs={EPOCHS} prefixes={PREFIXES} batch={BATCH}",flush=True)

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def cross_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)

def triplet_loss(za,zb,gtm):
    if gtm is not None: gtm=gtm.to(za.device)
    sim=cross_wj(za,zb); ap=sim.diagonal(); sc=sim.clone(); sc.fill_diagonal_(-1e9)
    if gtm is not None and gtm.any(): sc[gtm]=-1e9
    sa=sc.max(1).values; loss=F.relu(sa-ap+MARGIN); v=loss>0
    return (loss[v].mean() if v.any() else torch.tensor(0.,device=za.device,requires_grad=True)), float(v.float().mean())

def infonce_loss(za,zb,gtm):
    logits=cross_wj(za,zb)/TEMP
    if gtm is not None:
        m=gtm.to(logits.device).clone(); m.fill_diagonal_(False); logits=logits.masked_fill(m,-1e9)
    lab=torch.arange(za.shape[0],device=za.device); loss=F.cross_entropy(logits,lab)
    return loss, float((logits.argmax(1)==lab).float().mean())
LOSS_FN = triplet_loss if A.loss=='triplet' else infonce_loss

def matry(za_full,zb_full,gtm):
    tot=0.; metric=0.
    for m in PREFIXES:
        l,met=LOSS_FN(norm(za_full[:,:m]),norm(zb_full[:,:m]),gtm); tot=tot+l; metric=met
    return tot/len(PREFIXES), metric

class MatAE(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
        s.decoder=nn.Sequential(nn.Linear(EMB,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),nn.Linear(1024,d,bias=False))
    def embed(s,x): return F.relu(s.encoder(x))            # 4096, unnormalized (prefix-norm later)
    def forward(s,x): z=s.embed(x); return z, F.relu(s.decoder(norm(z)))
def module(m): return m.module if hasattr(m,'module') else m

class DS(Dataset):
    def __init__(s,gt,qsv,train_qids=None,in_corpus=False):
        allow=None if train_qids is None else set(train_qids)
        s.p=[]
        for q,nb in gt.items():
            if allow is not None and q not in allow:
                continue
            for n in nb[:MAX_POS]:
                if in_corpus:
                    if n < qsv and n != q:
                        s.p.append((q,n))
                elif q >= qsv and n < qsv:
                    s.p.append((q,n))
        random.shuffle(s.p)
        print(f"pairs={len(s.p):,} steps/epoch~{len(s.p)//BATCH}",flush=True)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def main():
    train_qids=eval_qids=None
    eval_gt=None
    in_corpus=False
    gt_train_start=0
    if A.dataset == "10k-train8k":
        if A.split_file:
            raise SystemExit("use --dataset 10k-train8k without --split-file")
        qt,gt,eval_gt,qs,cq,qq,cs,qtn=load_10k_train8k_normalized()
        train_qids=list(range(qs))
        eval_qids=list(range(qs,len(qtn)))
        in_corpus=True
        gt_train_start=0
        gt_cache_name="10k-train8k"
        print(f"train queries 0–{qs-1} eval queries {qs}–{len(qtn)-1}",flush=True)
    else:
        if A.split_file:
            with open(A.split_file,'rb') as f: split=pickle.load(f)
            train_qids,eval_qids=split['train_qids'],split['eval_qids']
            print(f"split {A.split_file}: train={len(train_qids):,} eval={len(eval_qids):,}",flush=True)
        qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full')
        gt_cache_name='full'
        gt_train_start=qs
    vecs=torch.from_numpy(np.ascontiguousarray(qtn,dtype=np.float32)).to(DEV)
    n_train_items=qs if in_corpus else len(qtn)
    gtg=build_gt_gpu(build_gt_cache(gt,n_train_items,gt_train_start,gt_cache_name),DEV)
    model=MatAE(qtn.shape[1]).to(DEV)
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model); print(f"DataParallel {torch.cuda.device_count()} GPUs",flush=True)
    loader=DataLoader(DS(gt,qs,train_qids,in_corpus=in_corpus),batch_size=BATCH,shuffle=True,num_workers=12,drop_last=True,persistent_workers=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(EPOCHS,1)); best=float('inf'); t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); tl=tm=tr=tv=st=0
        for ai,pi in tqdm(loader,desc=f"ep{ep:02d}/{EPOCHS}",ncols=110,mininterval=10):
            a=vecs[ai.to(DEV)]; p=vecs[pi.to(DEV)]; b=a.shape[0]
            z,xr=model(torch.cat([a,p])); za,zb=z[:b],z[b:]
            fn=build_fn_mask(ai,pi,gtg,gt_train_start); main_l,metric=matry(za,zb,fn); rec=F.mse_loss(xr[:b],a)
            loss=main_l+LAM_REC*rec
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl+=loss.item(); tm+=main_l.item(); tr+=rec.item(); tv+=metric; st+=1
        sch.step(); el=(time.time()-t0)/60
        mlabel='viol' if A.loss=='triplet' else 'acc'
        print(f"ep{ep:02d}/{EPOCHS} loss={tl/st:.4f} main={tm/st:.4f} rec={tr/st:.4f} {mlabel}={tv/st:.3f} {el:.1f}min eta={el/ep*(EPOCHS-ep):.1f}min",flush=True)
        if tl/st<best: best=tl/st; torch.save(module(model).state_dict(),CKPT); print(f"  -> saved {CKPT}",flush=True)

    # ---- eval: truncate to each prefix dim ----
    enc=module(model); enc.eval()
    with torch.no_grad():
        Z=torch.cat([enc.embed(torch.tensor(qtn[i:i+512],dtype=torch.float32,device=DEV)) for i in range(0,len(qtn),512)])  # (N,4096) GPU
    today=datetime.date.today().isoformat(); rows_all=[]
    eval_gt_use=eval_gt if eval_gt is not None else gt
    eval_idx=np.array(eval_qids if eval_qids is not None else list(range(qs,len(qtn))),dtype=np.int64)
    qq_eval=qq[eval_idx-qs] if eval_qids is not None else qq
    for m in PREFIXES:
        emb=norm(Z[:,:m]).cpu().numpy().astype(np.float32); ce=emb[:qs]; qe=emb[eval_idx]
        max_k=max(max(CAND_KS),500)
        nbrs,info=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=max_k,threads=THREADS,query_params={"efSearch":EF})
        base=eval_recall_by_qids(eval_gt_use,nbrs,list(eval_idx),max_k) if eval_qids else eval_recall(eval_gt_use,nbrs,qs,max_k)
        tag="eval2k" if A.dataset=="10k-train8k" else ("eval20" if eval_qids else "all_queries")
        print(f"[d={m} base {tag}] R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}",flush=True)
        rows_all.append([m,"base","",base[10],base[50],base[100],base[500],round(info['qps'])])
        preload_rerank_corpus(cq,cs)
        for ck in CAND_KS:
            cand,ci=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=ck,threads=THREADS,query_params={"efSearch":EF})
            t1=time.time(); rr=rerank_wj_gpu(qq_eval,cand,cq,cs,top_k=ck,batch_size=RB); e2e=len(qq_eval)/max(ci["query_s"]+(time.time()-t1),1e-9)
            mm=eval_recall_by_qids(eval_gt_use,rr,list(eval_idx),ck) if eval_qids else eval_recall(eval_gt_use,rr,qs,ck)
            print(f"[d={m} rerank K={ck} {tag}] R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",flush=True)
            rows_all.append([m,"rerank",ck,mm[10],mm[50],mm[100],mm[500],round(e2e)])
        release_rerank_corpus()
    split_note=f" {tag} n={len(eval_qids)}" if eval_qids else ""
    note=f"Matryoshka {A.loss} max_pos={MAX_POS} emb={EMB} prefixes={PREFIXES}; truncated+renorm; {THREADS} threads efSearch={EF}{split_note}"
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        for m,stage,ck,r10,r50,r100,r500,qps in rows_all:
            w.writerow([today,SPLIT_TAG,f"Matryoshka-{A.loss}-d{m}",m,stage,ck,round(r10,4),round(r50,4),round(r100,4),round(r500,4),qps,"run_matryoshka.py",note])
    print(f"appended {len(rows_all)} rows -> {CSV}",flush=True)

if __name__=="__main__": main()
