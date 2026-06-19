#!/usr/bin/env python3
"""Build the pk-real50k0.002 dataset: parse the real_*.txt encodings into a matrix,
parse the similarityMap GT into a lookup, and run a GT-metric oracle (raw vs
normalized vector-WJ) so we set the distillation target correctly.
Corpus=0..39999, Query=40000..49999."""
import os, glob, pickle, time, random
import numpy as np, pandas as pd, torch
ENC='/raid/ssEncodingData/encoding/pk-real50k0.002'
GT ='/raid/ssEncodingData/warehouse/pk-query-50k'
QS=40000; DEV='cuda:0'

def kf(f): return int(f.split('real_')[1].split('.txt')[0])
files=sorted(glob.glob(ENC+'/real_*.txt'),key=kf)
print(f"{len(files)} encoding files",flush=True); t0=time.time()
parts=[pd.read_csv(f,delim_whitespace=True,header=None,dtype=np.float32).values for f in files]
qt=np.ascontiguousarray(np.vstack(parts),dtype=np.float32)
print(f"qt {qt.shape} built in {(time.time()-t0)/60:.1f}min",flush=True)
np.save('/tmp/qt_50k.npy',qt)

gt={}
for f in sorted(glob.glob(GT+'/similarityMap_*')):
    for line in open(f):
        p=line.strip().split(',')
        if len(p)<2: continue
        gt[int(p[0])]=[int(x) for x in p[1:] if x.strip()!=''][:1000]
print(f"GT queries={len(gt)} depth~{len(next(iter(gt.values())))}",flush=True)
pickle.dump(gt,open('/tmp/gt_50k.pkl','wb'))

# GT-metric oracle: raw-vector-WJ vs normalized-vector-WJ vs the GT ranking
QTN=qt/np.clip(qt.sum(1,keepdims=True),1e-10,None)
Cr=torch.tensor(qt[:QS],device=DEV); Cn=torch.tensor(QTN[:QS],device=DEV)
rng=random.Random(0); sample=rng.sample(range(QS,qt.shape[0]),40)
def oracle(C,Q):
    R={50:[],500:[]}
    for qi in sample:
        g=gt.get(qi,[])
        if len(g)<50: continue
        q=torch.tensor(Q[qi],device=DEV).unsqueeze(0)
        s=torch.minimum(q,C).sum(1)/torch.maximum(q,C).sum(1).clamp(min=1e-10)
        top=torch.topk(s,500).indices.cpu().numpy()
        for k in (50,500): R[k].append(len(set(g[:k])&set(top[:k].tolist()))/len(set(g[:k])))
    return {k:float(np.mean(v)) for k,v in R.items()}
raw=oracle(Cr,qt); nrm=oracle(Cn,QTN)
print(f"RAW-vector-WJ  oracle vs GT : R@50={raw[50]:.4f} R@500={raw[500]:.4f}",flush=True)
print(f"NORM-vector-WJ oracle vs GT : R@50={nrm[50]:.4f} R@500={nrm[500]:.4f}",flush=True)
print(f"=> GT metric = {'RAW-vector WJ' if raw[500]>nrm[500] else 'NORMALIZED-vector WJ'}"
      f"{'  (WARNING: both low -> likely SHAPELY-WJ, vector gap)' if max(raw[500],nrm[500])<0.85 else ''}",flush=True)
print("done",flush=True)
