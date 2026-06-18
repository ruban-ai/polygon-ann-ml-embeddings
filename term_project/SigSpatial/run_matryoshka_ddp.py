#!/usr/bin/env python3
"""DDP version of run_matryoshka.py (held-out train80/eval20 baselines). Same MatAE model,
same triplet/InfoNCE+FN-filter+recon loss, same Matryoshka prefixes and eval — just true
multi-GPU DDP so the held-out 512-d baselines finish in minutes, not hours.
Launch: CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 run_matryoshka_ddp.py --loss infonce --split-file /tmp/query_split_80_20.pkl"""
import sys, os, time, csv, datetime, random, argparse, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (build_fn_mask, build_gt_cache, build_gt_gpu, eval_recall_by_qids,
    load_dataset_normalized, nmslib_neighbors, preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu)

ap=argparse.ArgumentParser()
ap.add_argument('--loss',choices=['triplet','infonce'],required=True)
ap.add_argument('--split-file',default='/tmp/query_split_80_20.pkl')
A=ap.parse_args()
PREFIXES=[256,512,1024,2048,4096]; EMB=4096
BATCH=512; LR=1e-3; WD=1e-4; MARGIN=0.3; TEMP=0.07; LAM_REC=0.1
EF=200; RB=64; CAND_KS=[1000,2000]; SEED=42
MAX_POS = 30 if A.loss=='triplet' else 256
EPOCHS  = 75 if A.loss=='triplet' else 18
CSV="/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def cross_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)
def triplet_loss(za,zb,gtm):
    sim=cross_wj(za,zb); ap_=sim.diagonal(); sc=sim.clone(); sc.fill_diagonal_(-1e9)
    if gtm is not None and gtm.any(): sc[gtm]=-1e9
    sa=sc.max(1).values; loss=F.relu(sa-ap_+MARGIN); v=loss>0
    return (loss[v].mean() if v.any() else torch.tensor(0.,device=za.device,requires_grad=True)), float(v.float().mean())
def infonce_loss(za,zb,gtm):
    logits=cross_wj(za,zb)/TEMP
    if gtm is not None:
        m=gtm.clone(); m.fill_diagonal_(False); logits=logits.masked_fill(m,-1e9)
    lab=torch.arange(za.shape[0],device=za.device); loss=F.cross_entropy(logits,lab)
    return loss, float((logits.argmax(1)==lab).float().mean())
LOSS_FN = triplet_loss if A.loss=='triplet' else infonce_loss
def matry(za,zb,gtm):
    tot=0.; metric=0.
    for m in PREFIXES:
        l,met=LOSS_FN(norm(za[:,:m]),norm(zb[:,:m]),gtm); tot=tot+l; metric=met
    return tot/len(PREFIXES), metric

class MatAE(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
        s.decoder=nn.Sequential(nn.Linear(EMB,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),nn.Linear(1024,d,bias=False))
    def embed(s,x): return F.relu(s.encoder(x))
    def forward(s,x): z=s.embed(x); return z, F.relu(s.decoder(norm(z)))

class DS(Dataset):
    def __init__(s,gt,qsv,allow):
        a=set(int(x) for x in allow); s.p=[]
        for q,nb in gt.items():
            if q not in a: continue
            for n in nb[:MAX_POS]:
                if q>=qsv and n<qsv: s.p.append((q,int(n)))
        random.Random(SEED).shuffle(s.p)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def main():
    dist.init_process_group('nccl'); lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV=torch.device(f'cuda:{lr}'); main=(lr==0); world=dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    split=pickle.load(open(A.split_file,'rb')); TR=split['train_qids']; EV=np.array(split['eval_qids'],dtype=np.int64)
    qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full')
    vecs=torch.from_numpy(np.ascontiguousarray(qtn,dtype=np.float32)).to(DEV)
    gtg=build_gt_gpu(build_gt_cache(gt,len(qtn),qs,'full'),DEV)
    if main: print(f"DDP world={world} loss={A.loss} corpus={qs} train={len(TR)} eval={len(EV)}",flush=True)
    model=DDP(MatAE(qtn.shape[1]).to(DEV),device_ids=[lr])
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    ds=DS(gt,qs,TR); sampler=DistributedSampler(ds,shuffle=True,seed=SEED)
    loader=DataLoader(ds,batch_size=BATCH,sampler=sampler,num_workers=4,drop_last=True,pin_memory=False)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); best=1e9; t0=time.time()
    if main: print(f"pairs={len(ds):,} per-rank steps/epoch~{len(ds)//(world*BATCH)}",flush=True)
    for ep in range(1,EPOCHS+1):
        model.train(); sampler.set_epoch(ep); tl=tv=st=0
        it=tqdm(loader,desc=f"ep{ep:02d}/{EPOCHS}",ncols=100,mininterval=10) if main else loader
        for ai,pi in it:
            a=vecs[ai.to(DEV)]; p=vecs[pi.to(DEV)]; b=a.shape[0]
            z,xr=model(torch.cat([a,p])); za,zb=z[:b],z[b:]
            fn=build_fn_mask(ai,pi,gtg,qs); fn=fn.to(DEV) if fn is not None else None
            main_l,metric=matry(za,zb,fn); rec=F.mse_loss(xr[:b],a)
            loss=main_l+LAM_REC*rec
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item(); tv+=metric; st+=1
        sch.step()
        if main:
            ml='viol' if A.loss=='triplet' else 'acc'
            print(f"ep{ep:02d} loss={tl/st:.4f} {ml}={tv/st:.3f} {(time.time()-t0)/60:.1f}min",flush=True)
            if tl/st<best: best=tl/st; torch.save(model.module.state_dict(),f'/tmp/best_matddp_{A.loss}.pt')
    dist.barrier()
    if not main: dist.destroy_process_group(); return

    enc=MatAE(qtn.shape[1]).to(DEV); enc.load_state_dict(torch.load(f'/tmp/best_matddp_{A.loss}.pt',map_location=DEV,weights_only=True)); enc.eval()
    THREADS=120; torch.set_num_threads(THREADS)
    with torch.no_grad():
        Z=torch.cat([enc.embed(vecs[i:i+512]) for i in range(0,len(vecs),512)])
    today=datetime.date.today().isoformat(); rows=[]; ei=EV; qq_e=qq[ei-qs]
    for m in PREFIXES:
        emb=norm(Z[:,:m]).cpu().numpy().astype(np.float32); ce=emb[:qs]; qe=emb[ei]
        mk=max(max(CAND_KS),500)
        nb,info=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=mk,threads=THREADS,query_params={"efSearch":EF})
        base=eval_recall_by_qids(gt,nb,list(ei),mk)
        print(f"[d={m} base eval20] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}",flush=True)
        rows.append([m,"base","",base[10],base[50],base[100],base[500],round(info['qps'])])
        preload_rerank_corpus(cq,cs)
        for ck in CAND_KS:
            cand,ci=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=ck,threads=THREADS,query_params={"efSearch":EF})
            t1=time.time(); rr=rerank_wj_gpu(qq_e,cand,cq,cs,top_k=ck,batch_size=RB); e2e=len(qq_e)/max(ci["query_s"]+(time.time()-t1),1e-9)
            mm=eval_recall_by_qids(gt,rr,list(ei),ck)
            print(f"[d={m} rerank K={ck} eval20] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",flush=True)
            rows.append([m,"rerank",ck,mm[10],mm[50],mm[100],mm[500],round(e2e)])
        release_rerank_corpus()
    note=f"Matryoshka-DDP {A.loss} eval20 n={len(ei)} prefixes={PREFIXES}; efSearch={EF}"
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        for m,stage,ck,r10,r50,r100,r500,qps in rows:
            w.writerow([today,"train80",f"MatDDP-{A.loss}-d{m}",m,stage,ck,round(r10,4),round(r50,4),round(r100,4),round(r500,4),qps,"run_matryoshka_ddp.py",note])
    print(f"appended {len(rows)} rows -> {CSV}",flush=True)
    dist.destroy_process_group()

if __name__=='__main__': main()
