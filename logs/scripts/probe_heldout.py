"""Does the train80 model's GOOD intermediate (first-4096) GENERALIZE? Compare recall on
TRAINING queries vs HELD-OUT (eval20) queries, at first-4096 (no loss) vs output (loss),
vs random projection (can't memorize). R@50 AND R@500."""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,'/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import load_dataset_normalized, shifted_l1_simplex
DEV=torch.device('cuda:0')
qt,gt,qs,cq,qq,cs,qtn=load_dataset_normalized('full'); IN=qtn.shape[1]
sp=pickle.load(open('/tmp/query_split_80_20.pkl','rb')); TR=np.array(sp['train_qids']); EV=np.array(sp['eval_qids'])
print(f'train_q={len(TR)} eval_q={len(EV)}',flush=True)
class Mat(nn.Module):
    def __init__(s):
        super().__init__(); s.encoder=nn.Sequential(nn.Linear(IN,4096,bias=False),nn.BatchNorm1d(4096),nn.ReLU(),
            nn.Linear(4096,4096,bias=False),nn.BatchNorm1d(4096))
def load(ck):
    m=Mat().to(DEV); sd=torch.load(ck,map_location=DEV,weights_only=True)
    m.load_state_dict({k:v for k,v in sd.items() if k.startswith('encoder.')},strict=False); m.eval(); return m
@torch.no_grad()
def tap(m,upto,bs=1024):
    out=[]
    for i in range(0,len(qtn),bs):
        h=m.encoder[:upto](torch.tensor(qtn[i:i+bs],dtype=torch.float32,device=DEV))
        if upto==5: h=F.relu(h)
        out.append(h/h.sum(1,keepdim=True).clamp(min=1e-10))
    return torch.cat(out)
@torch.no_grad()
def randproj(o=4096,bs=2048,seed=42):
    rng=np.random.default_rng(seed); W=torch.tensor((rng.standard_normal((IN,o))/np.sqrt(o)).astype(np.float32),device=DEV)
    Z=torch.cat([torch.tensor(np.asarray(qt[i:i+bs],dtype=np.float32),device=DEV)@W for i in range(0,len(qt),bs)]).cpu().numpy()
    return torch.tensor(shifted_l1_simplex(Z),device=DEV)
@torch.no_grad()
def recall(emb,qids,n_q=3000,maxk=500,qc=128):
    ce=emb[:qs]; sel=np.random.RandomState(0).choice(qids,min(n_q,len(qids)),replace=False); res={50:[],500:[]}
    for s in range(0,len(sel),qc):
        ids=sel[s:s+qc]; top=torch.topk(torch.cdist(emb[torch.tensor(ids,device=DEV)],ce,p=1),maxk,dim=1,largest=False).indices.cpu().numpy()
        for j,gid in enumerate(ids):
            t=gt.get(int(gid),[])
            if not t: continue
            for k in res: res[k].append(len(set(t[:k])&set(top[j][:k].tolist()))/len(set(t[:k])))
    return {k:float(np.mean(v)) for k,v in res.items()}
def line(lbl,emb):
    rt=recall(emb,TR); re=recall(emb,EV)
    print(f'{lbl:<26} train: R@50={rt[50]:.4f} R@500={rt[500]:.4f}  |  eval20: R@50={re[50]:.4f} R@500={re[500]:.4f}  |  gap R@50={rt[50]-re[50]:+.3f}',flush=True)
m=load('/tmp/best_matryoshka_triplet_train80.pt')
print('\n== train80 triplet Matryoshka ==',flush=True)
line('first-4096 (NO loss)',tap(m,3)); line('output-4096 (LOSS)',tap(m,5))
line('random-proj-4096 (no train)',randproj(4096))
print('done',flush=True)
