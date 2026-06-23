import sys, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import build_fn_mask, build_gt_cache, build_gt_gpu, load_dataset_normalized

ap=argparse.ArgumentParser(); ap.add_argument('--out_dim',type=int); ap.add_argument('--steps',type=int,default=400)
ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--lam',type=float,default=0.1); ap.add_argument('--sep',type=int,default=0); ap.add_argument('--maxpos',type=int,default=256); a=ap.parse_args()
DEV=torch.device('cuda:0'); B=2048; MARGIN=0.3; MAX_POS=256; torch.manual_seed(42); np.random.seed(42)

def cross_wj_l1(x,y):
    l1=torch.cdist(x,y,p=1); return (2.0-l1)/(2.0+l1)
def triplet(an,po,gt):
    sim=cross_wj_l1(an,po); ap_=sim.diagonal(); sc=sim.clone(); sc.fill_diagonal_(-1e9)
    if gt is not None and gt.any(): sc[gt]=-1e9
    sa=sc.max(1).values; loss=F.relu(sa-ap_+MARGIN); v=loss>0
    return (loss[v].mean() if v.any() else torch.tensor(0.,device=DEV,requires_grad=True)), float(v.float().mean())
class AE(nn.Module):
    def __init__(s,d,o):
        super().__init__()
        s.encoder=nn.Sequential(nn.Linear(d,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),
            nn.Linear(1024,o,bias=False),nn.BatchNorm1d(o))
        s.decoder=nn.Sequential(nn.Linear(o,1024,bias=False),nn.BatchNorm1d(1024),nn.ReLU(),nn.Linear(1024,d,bias=False))
    def encode(s,x): z=F.relu(s.encoder(x)); return z/z.sum(1,keepdim=True).clamp(min=1e-10)
    def forward(s,x): z=s.encode(x); return z,F.relu(s.decoder(z))
MP=256
class DS(Dataset):
    def __init__(s,gt,qs):
        s.p=[(q,n) for q,nb in gt.items() for n in nb[:MP] if q>=qs and n<qs]; np.random.shuffle(s.p)
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return s.p[i]
def rowdiv(H): c=H-H.mean(0,keepdim=True); return (c.norm(dim=1).mean()/(H.norm(dim=1).mean()+1e-9)).item()

qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full')
vecs=torch.from_numpy(np.ascontiguousarray(qtn,dtype=np.float32)).to(DEV)
gtg=build_gt_gpu(build_gt_cache(gt,len(qtn),qs,'full'),DEV)
m=AE(qtn.shape[1],a.out_dim).to(DEV)
opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
MP=a.maxpos
loader=DataLoader(DS(gt,qs),batch_size=B,shuffle=True,num_workers=4,drop_last=True,persistent_workers=True)
print(f'OUT_DIM={a.out_dim} lr={a.lr} steps={a.steps}',flush=True)
st=0
for ai,pi in loader:
    av=vecs[ai.to(DEV)]; pv=vecs[pi.to(DEV)]
    if a.sep:
        za,xr=m(av); zp,_=m(pv)
    else:
        z,xr=m(torch.cat([av,pv])); za,zp=z[:B],z[B:]; xr=xr[:B]
    fn=build_fn_mask(ai,pi,gtg,qs); main,viol=triplet(za,zp,fn); rec=F.mse_loss(xr,av)
    loss=main+a.lam*rec; opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); st+=1
    if st%50==0 or st==1:
        with torch.no_grad(): emb_div=rowdiv(za.detach())
        print(f'  step{st:4d} loss={loss.item():.4f} main={main.item():.4f} viol={viol:.3f} embdiv={emb_div:.4f}',flush=True)
    if st>=a.steps: break
print('done',flush=True)
