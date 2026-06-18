#!/usr/bin/env python3
"""WJ-NATIVE distillation on 10K (zero cosine). 18499->4096->4096, L1-simplex output.
Loss = MSE( emb-WJ , raw-18499-WJ ) in-batch -> directly aligns the objective with the metric.
Eval: tap first-4096 (no loss) vs output-4096 (loss); WJ kNN recall (R@50/R@500) + sp@30.
If the OUTPUT is no longer damaged (output sp@30 ~ intermediate), alignment fixed it."""
import sys, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized
DEV=torch.device('cuda:0'); SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('10k'); IN=qtn.shape[1]
print(f'10K corpus={qs} q={len(qq)} in={IN}',flush=True)
RAW=torch.from_numpy(np.ascontiguousarray(qt,dtype=np.float32)).to(DEV)   # raw vectors for WJ targets
VN =torch.from_numpy(np.ascontiguousarray(qtn,dtype=np.float32)).to(DEV)  # l1-simplex model input
EMB=4096; B=512; EPOCHS=40; LR=1e-3; MAX_POS=100

def norm(x): return x/x.sum(1,keepdim=True).clamp(min=1e-10)
def emb_wj(a,b): l1=torch.cdist(a,b,p=1); return (2.0-l1)/(2.0+l1)     # simplex -> WJ (has grad)
@torch.no_grad()
def raw_wj(R,chunk=64):   # exact WJ on raw vectors (no grad) -> (M,M) target
    M=R.shape[0]; out=torch.empty(M,M,device=R.device)
    for i in range(0,M,chunk):
        ri=R[i:i+chunk]
        mn=torch.minimum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2)
        mx=torch.maximum(ri.unsqueeze(1),R.unsqueeze(0)).sum(2).clamp(min=1e-10)
        out[i:i+chunk]=mn/mx
    return out
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(IN,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,EMB,bias=False),nn.BatchNorm1d(EMB))
    def embed(s,x): return F.relu(s.encoder(x))
class DS(Dataset):
    def __init__(s):
        s.p=[(q,n) for q,nb in gt.items() for n in nb[:MAX_POS] if q>=qs and n<qs]; random.shuffle(s.p)
        print(f'pairs={len(s.p):,} steps/epoch~{len(s.p)//B}',flush=True)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]

def main():
    m=Net().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    loader=DataLoader(DS(),batch_size=B,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); t0=time.time(); best=1e9
    for ep in range(1,EPOCHS+1):
        m.train(); tl=st=0
        for ai,pi in tqdm(loader,desc=f'ep{ep:02d}/{EPOCHS}',ncols=100,mininterval=10):
            ids=torch.cat([ai,pi]).to(DEV)            # M=2B items
            z=norm(m.embed(VN[ids]))                  # (M,4096) L1-simplex
            ew=emb_wj(z,z)                            # (M,M) emb-WJ (grad)
            tw=raw_wj(RAW[ids])                       # (M,M) raw-WJ target
            loss=F.mse_loss(ew,tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            tl+=loss.item(); st+=1
        sch.step(); print(f'ep{ep:02d} mse={tl/st:.5f} {(time.time()-t0)/60:.1f}min',flush=True)
        if tl/st<best: best=tl/st; torch.save(m.state_dict(),'/tmp/best_wjdistill_native_10k.pt')

    # ---- eval: tap first-4096 (no loss) vs output (loss), WJ space ----
    m.eval()
    @torch.no_grad()
    def tap(upto,bs=2048):
        o=[]
        for i in range(0,len(qtn),bs):
            h=m.encoder[:upto](torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV))
            if upto==5: h=F.relu(h)
            o.append(norm(h))
        return torch.cat(o)
    def spear(x,y):
        rx=x.argsort().argsort().float(); ry=y.argsort().argsort().float(); rx=rx-rx.mean(); ry=ry-ry.mean()
        return ((rx*ry).sum()/(rx.norm()*ry.norm()+1e-9)).item()
    @torch.no_grad()
    def recall(emb,maxk=500,qc=256):
        ce=emb[:qs]; qe=emb[qs:]; res={50:[],500:[]}
        for s in range(0,len(qe),qc):
            top=torch.topk(torch.cdist(qe[s:s+qc],ce,p=1),min(maxk,qs),dim=1,largest=False).indices.cpu().numpy()
            for j in range(len(top)):
                t=gt.get(qs+s+j,[])
                if not t: continue
                for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
        return {k:float(np.mean(v)) for k,v in res.items()}
    @torch.no_grad()
    def spd(emb,depths=(30,500),n_q=200,nr=400):
        rng=np.random.RandomState(3); acc={d:[] for d in depths}
        for qi in rng.choice(len(qq),n_q,replace=False):
            gq=qs+int(qi); t=gt.get(gq,[])
            if len(t)<30: continue
            for d in depths:
                dd=min(d,len(t)); cand=np.concatenate([np.array(t[:dd]),rng.choice(qs,nr,replace=False)])
                rr=raw_wj(RAW[torch.tensor(np.concatenate([[gq],cand]),device=DEV)])[0,1:]
                ee=emb_wj(emb[gq].unsqueeze(0),emb[torch.tensor(cand,device=DEV)])[0]
                acc[d].append(spear(ee,rr))
        return {d:float(np.mean(v)) for d,v in acc.items()}
    print('\n== WJ-native distillation (10K) ==',flush=True)
    for lab,upto in [('first-4096 (NO loss)',3),('output-4096 (LOSS)',5)]:
        e=tap(upto); r=recall(e); s=spd(e)
        print(f'  {lab:<22} R@50={r[50]:.4f} R@500={r[500]:.4f}  sp@30={s[30]:.3f} sp@500={s[500]:.3f}',flush=True)
    print('done',flush=True)

if __name__=='__main__': main()
