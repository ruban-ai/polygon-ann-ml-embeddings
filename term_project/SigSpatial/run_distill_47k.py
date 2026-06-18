#!/usr/bin/env python3
"""47K SELF-CONTAINED, WJ-native Matryoshka distillation (zero cosine, single GPU).
Corpus = train_qids (80%, ~35.7K); queries = held-out eval_qids (20%, ~8.9K) from
query_split_80_20.pkl (area-stratified). GT = exact raw-18220-WJ neighbors AMONG the 47K corpus.
ONE 4096-d model; WJ-distillation MSE on nested prefixes {256,512,1024,2048,4096}. Train on
corpus-internal neighbor pairs; eval held-out queries vs the 47K corpus: WJ HNSW + rerank + QPS."""
import sys, os, time, csv, datetime, random, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized, nmslib_neighbors, rerank_wj_gpu, preload_rerank_corpus, release_rerank_corpus
DEV=torch.device('cuda:0'); THREADS=64; torch.set_num_threads(THREADS)
PREFIXES=[256,512,1024,2048,4096]; EMB=4096
B=512; EPOCHS=15; LR=1e-3; WD=1e-4; MAXPOS=30
EF=200; RB=64; CAND_KS=[500,1000]; SEED=42
CSV="/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def emb_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)
@torch.no_grad()
def raw_wj(R,chunk=64):
    M=R.shape[0]; out=torch.empty(M,M,device=R.device)
    for i in range(0,M,chunk):
        ri=R[i:i+chunk]
        out[i:i+chunk]=torch.minimum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2)/torch.maximum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2).clamp(min=1e-10)
    return out
@torch.no_grad()
def topk_raw_wj(A,Bm,k,rc=128,cc=1024):     # A:(nA,D) Bm:(nB,D) raw on GPU -> top-k B-idx per A row (largest WJ)
    nA=A.shape[0]; idx=torch.empty(nA,k,dtype=torch.long,device=DEV)
    for i in range(0,nA,rc):
        a=A[i:i+rc]; sims=torch.empty(a.shape[0],Bm.shape[0],device=DEV)
        for j in range(0,Bm.shape[0],cc):
            b=Bm[j:j+cc]
            sims[:,j:j+cc]=torch.minimum(a.unsqueeze(1),b.unsqueeze(0)).sum(2)/torch.maximum(a.unsqueeze(1),b.unsqueeze(0)).sum(2).clamp(min=1e-10)
        idx[i:i+rc]=sims.topk(k,dim=1,largest=True).indices
    return idx.cpu().numpy()

class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
    def forward(s,x): return F.relu(s.encoder(x))
class PairDS(Dataset):
    def __init__(s,knn):                      # knn:(nC,MAXPOS) local corpus idx (self removed)
        s.p=[(i,int(j)) for i in range(knn.shape[0]) for j in knn[i]]
        random.shuffle(s.p); print(f"pairs={len(s.p):,} steps/epoch~{len(s.p)//B}",flush=True)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def recall_local(GTq,nbrs,ks=(50,500)):       # GTq:(nQ,maxk) true corpus idx; nbrs:(nQ,maxk) retrieved
    res={k:[] for k in ks}
    for q in range(len(nbrs)):
        for k in ks:
            c=set(GTq[q][:k].tolist()); r=set(nbrs[q][:k].tolist()); res[k].append(len(c&r)/max(len(c),1))
    return {k:float(np.mean(v)) for k,v in res.items()}

def main():
    sp=pickle.load(open('/tmp/query_split_80_20.pkl','rb'))
    cor=np.array(sp['train_qids'],dtype=np.int64); qry=np.array(sp['eval_qids'],dtype=np.int64)
    qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full')
    print(f"47K self: corpus={len(cor)} query={len(qry)} dim={qtn.shape[1]}",flush=True)
    Cn=np.ascontiguousarray(qtn[cor],dtype=np.float32); Cr=np.ascontiguousarray(qt[cor],dtype=np.float32)
    Qn=np.ascontiguousarray(qtn[qry],dtype=np.float32); Qr=np.ascontiguousarray(qt[qry],dtype=np.float32)
    # --- GT among the 47K corpus (exact raw-WJ) ---
    print("computing corpus self-kNN + eval GT (raw-WJ)...",flush=True); t0=time.time()
    Cr_g=torch.from_numpy(Cr).to(DEV)
    knn=topk_raw_wj(Cr_g,Cr_g,MAXPOS+1)[:,1:]            # drop self (col0)
    GTq=topk_raw_wj(torch.from_numpy(Qr).to(DEV),Cr_g,500)  # (nQ,500) true corpus idx
    del Cr_g; torch.cuda.empty_cache(); print(f"  GT done in {(time.time()-t0)/60:.1f}min",flush=True)
    Cn_g=torch.from_numpy(Cn).to(DEV)                    # corpus l1-simplex on GPU (encode batches)
    Cr_cpu=torch.from_numpy(Cr)                          # raw on CPU (target per batch)

    m=Net(qtn.shape[1]).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    loader=DataLoader(PairDS(knn),batch_size=B,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); best=1e9; t0=time.time()
    for ep in range(1,EPOCHS+1):
        m.train(); tl=st=0
        for ai,pi in tqdm(loader,desc=f"ep{ep:02d}/{EPOCHS}",ncols=100,mininterval=10):
            ids=torch.cat([ai,pi])
            z=m(Cn_g[ids.to(DEV)])
            with torch.no_grad(): tw=raw_wj(Cr_cpu[ids].to(DEV))
            loss=sum(F.mse_loss(emb_wj(norm(z[:,:k]),norm(z[:,:k])),tw) for k in PREFIXES)/len(PREFIXES)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); tl+=loss.item(); st+=1
        sch.step(); print(f"ep{ep:02d} mse={tl/st:.6f} {(time.time()-t0)/60:.1f}min",flush=True)
        if tl/st<best: best=tl/st; torch.save(m.state_dict(),'/tmp/best_distill_47k.pt')

    # --- eval: per prefix, WJ HNSW on 47K corpus + rerank + QPS, held-out queries ---
    m.eval()
    with torch.no_grad():
        ZC=torch.cat([m(Cn_g[i:i+1024]) for i in range(0,len(Cn),1024)])
        ZQ=torch.cat([m(torch.from_numpy(Qn[i:i+1024]).to(DEV)) for i in range(0,len(Qn),1024)])
    Cr_sum=Cr.sum(1).astype(np.float32); today=datetime.date.today().isoformat(); rows=[]
    for k in PREFIXES:
        ce=norm(ZC[:,:k]).cpu().numpy().astype(np.float32); qe=norm(ZQ[:,:k]).cpu().numpy().astype(np.float32)
        mk=max(max(CAND_KS),500)
        nb,info=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=mk,threads=THREADS,query_params={"efSearch":EF})
        base=recall_local(GTq,np.asarray([r[0] if isinstance(r,tuple) else r for r in nb]))
        print(f"[d={k} base] R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}",flush=True)
        rows.append([k,"base","",base[50],base[500],round(info['qps'])])
        preload_rerank_corpus(Cr,Cr_sum)
        for ck in CAND_KS:
            cand,ci=nmslib_neighbors(ce,qe,space="WeightedJaccard",k=ck,threads=THREADS,query_params={"efSearch":EF})
            t1=time.time(); rr=rerank_wj_gpu(Qr,cand,Cr,Cr_sum,top_k=ck,batch_size=RB); e2e=len(Qr)/max(ci["query_s"]+(time.time()-t1),1e-9)
            mm=recall_local(GTq,np.asarray(rr))
            print(f"[d={k} rerank K={ck}] R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",flush=True)
            rows.append([k,"rerank",ck,mm[50],mm[500],round(e2e)])
        release_rerank_corpus()
    note=f"47K-self distill (corpus={len(cor)} query={len(qry)}) WJ-native prefixes={PREFIXES}; {THREADS}thr"
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        for k,stage,ck,r50,r500,qps in rows:
            w.writerow([today,"47k",f"Distill47K-d{k}",k,stage,ck,"",round(r50,4),"",round(r500,4),qps,"run_distill_47k.py",note])
    print(f"appended {len(rows)} rows -> {CSV}",flush=True)

if __name__=='__main__': main()
