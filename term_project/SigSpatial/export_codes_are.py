#!/usr/bin/env python3
"""2026-10-12 (Prasad: throughput + index memory on Overture 1M/10M vs ShapeToVec+).
Exports our 8-bit signatures as raw uint8 codes + per-dimension scales so they can be indexed with
ShapeToVec+'s own SIMD integer kernel (patched NMSLIB space 'WeightedJaccardRatio', weights_file = scales).
That kernel computes sum_i w_i*min(qa_i,qb_i) / sum_i w_i*max(qa_i,qb_i) -- exactly WJ on our decoded
values q*s/255 (the 1/255 cancels). Quantization is IDENTICAL to eval_quant.py.quantize (99.9th-pct scale,
rint, clip, dead-vector fix). Usage: python export_codes_are.py <cfg> --prefixes 64,256"""
import sys, time, numpy as np, torch
import run_fulls_listwise_train as T
from fulls_common import CONFIGS, raw_matrix, rows

CFG = CONFIGS[sys.argv[1]]
PREF = [int(x) for x in sys.argv[sys.argv.index('--prefixes') + 1].split(',')]
OUT = '/raid/ruban/hpmlproj/term_project/SigSpatial/are_codes'

def codes(zc, zq, bits=8):
    L = 2 ** bits - 1
    s = np.percentile(zc, 99.9, axis=0).astype(np.float32); s[s <= 0] = zc.max(0)[s <= 0]; s[s <= 0] = 1.0
    res = []
    for z in (zc, zq):
        c = np.clip(np.rint(z / s * L), 0, L)
        dead = c.sum(1) == 0
        if dead.any(): c[dead, np.argmax(z[dead] / s, 1)] = 1
        res.append(c.astype(np.uint8))
    return res[0], res[1], s

t0 = time.time(); DEV = torch.device('cuda:0')
X = raw_matrix(CFG); IN = X.shape[1]
cids = np.arange(CFG['corpus_n']); qids = np.arange(CFG['corpus_n'], CFG['total'])
enc = T.Net(IN).to(DEV); enc.load_state_dict(torch.load(CFG['ckpt'], map_location=DEV, weights_only=True)); enc.eval()
maxp = max(PREF)
def embed(ids):
    Z = np.empty((len(ids), maxp), dtype=np.float32); te = time.time()
    with torch.no_grad():
        for i in range(0, len(ids), 8192):
            Z[i:i + 8192] = enc.embed(T.gnorm(torch.from_numpy(rows(X, ids[i:i + 8192])).to(DEV)))[:, :maxp].cpu().numpy()
            if (i // 8192) % 100 == 0:
                print(f"  embedded {min(i+8192, len(ids)):,}/{len(ids):,} ({(time.time()-te)/60:.1f}min)", flush=True)
    return Z
Zc, Zq = embed(cids), embed(qids)
for m in PREF:
    zc = T.norm(torch.from_numpy(Zc[:, :m])).numpy(); zq = T.norm(torch.from_numpy(Zq[:, :m])).numpy()
    cc, cq, s = codes(zc, zq)
    np.save(f"{OUT}/{CFG['name']}_d{m}_corpus.npy", cc); np.save(f"{OUT}/{CFG['name']}_d{m}_query.npy", cq)
    np.save(f"{OUT}/{CFG['name']}_d{m}_scales.npy", s)
    print(f"[{CFG['name']} d={m}] corpus {cc.shape} query {cq.shape} -> {OUT} ({(time.time()-t0)/60:.1f}min)", flush=True)
print(f"EXPORT DONE {CFG['name']} {(time.time()-t0)/60:.1f}min", flush=True)
