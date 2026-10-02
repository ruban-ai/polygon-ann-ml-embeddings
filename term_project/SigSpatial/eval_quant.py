#!/usr/bin/env python3
"""Quantized-embedding evaluation (2026-10-02): does storing our learned signature in
8 or 4 bits per dimension keep recall? Motivation: ShapeToVec+'s 32x is relative to
3K-d float32 (BAE-1bit = ~384 B/polygon), while our 256-d float32 embedding is 1 KB.
Scheme (ShapeToVec+-style code + shared table): per-dimension scale s_j = 99.9th
percentile of that coordinate over the corpus embeddings; code = clip(round(z_j/s_j*L),0,L),
L = 2^bits-1; search uses code*s_j/L (WJ needs the per-dim scale, exactly like
ShapeToVec+'s LUT). Corpus AND queries are quantized (symmetric, conservative).
Stored bytes/polygon = d*bits/8 (+ one shared d-float table). No retraining.
Usage: python eval_quant.py <cfg> --prefixes 256[,64] [--bits 8,4] [--no-ref] [--threads N] [--smoke]"""
import sys, time, csv, json, datetime
import numpy as np, torch
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS, BASE, raw_matrix, rows, load_gt
from sota_experiment_common import nmslib_neighbors, eval_recall
import run_fulls_listwise_train as T

CFG = CONFIGS[sys.argv[1]]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PREF = [int(x) for x in arg('--prefixes', '256').split(',')]
BITS = [int(x) for x in arg('--bits', '8,4').split(',')]
THREADS = int(arg('--threads', '96')); REF = '--no-ref' not in sys.argv; SMOKE = '--smoke' in sys.argv
EF = 200; K = 500


def quantize(zc, zq, bits):
    L = 2 ** bits - 1
    s = np.percentile(zc, 99.9, axis=0).astype(np.float32); s[s <= 0] = zc.max(0)[s <= 0]; s[s <= 0] = 1.0
    out = []
    for z in (zc, zq):
        code = np.clip(np.rint(z / s * L), 0, L)
        dead = code.sum(1) == 0                                   # all coords rounded to 0 -> keep top coord
        if dead.any():
            code[dead, np.argmax(z[dead] / s, 1)] = 1
        out.append((code * (s / L)).astype(np.float32))
    return out[0], out[1], int(((code.sum(1) == 0)).sum())


def main():
    t0 = time.time(); DEV = torch.device('cuda:0')
    X = raw_matrix(CFG); IN = X.shape[1]; corpus_n, total = CFG['corpus_n'], CFG['total']
    gt = load_gt(CFG)
    cids = np.arange(20_000 if SMOKE else corpus_n); qids = np.arange(corpus_n, total)[:200 if SMOKE else None]
    if SMOKE:
        cids = np.unique(np.concatenate([cids] + [np.array(gt[q][:50]) for q in qids if q in gt]))
        pos = {int(c): i for i, c in enumerate(cids)}
        gt = {corpus_n + k: [pos[n] for n in gt.get(q, []) if n in pos] for k, q in enumerate(qids)}
    enc = T.Net(IN).to(DEV); enc.load_state_dict(torch.load(CFG['ckpt'], map_location=DEV, weights_only=True)); enc.eval()
    maxp = max(PREF)
    print(f"[{CFG['name']}] quant eval prefixes={PREF} bits={BITS} ref={REF} corpus={len(cids):,} "
          f"queries={len(qids):,} ckpt={CFG['ckpt'].split('/')[-1]}", flush=True)

    def embed(ids):
        Z = np.empty((len(ids), maxp), dtype=np.float32); te = time.time()
        with torch.no_grad():
            for i in range(0, len(ids), 8192):
                Z[i:i + 8192] = enc.embed(T.gnorm(torch.from_numpy(rows(X, ids[i:i + 8192])).to(DEV)))[:, :maxp].cpu().numpy()
                if (i // 8192) % 200 == 0:
                    print(f"  embedded {min(i+8192, len(ids)):,}/{len(ids):,} ({(time.time()-te)/60:.1f}min)", flush=True)
        return Z
    Zc, Zq = embed(cids), embed(qids)
    out = []
    for m in PREF:
        zc = T.norm(torch.from_numpy(Zc[:, :m])).numpy(); zq = T.norm(torch.from_numpy(Zq[:, :m])).numpy()
        for b in ([32] if REF else []) + BITS:
            tm = time.time()
            if b == 32:
                ce, qe, dead = zc, zq, 0
            else:
                ce, qe, dead = quantize(zc, zq, b)
            nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=K, threads=THREADS, query_params={"efSearch": EF})
            r = eval_recall(gt, nb, corpus_n, K)
            byt = m * 4 if b == 32 else m * b // 8
            print(f"[{CFG['name']} d={m} {'fp32' if b == 32 else f'int{b}'} {byt}B] R@10={r[10]:.4f} R@50={r[50]:.4f} "
                  f"R@100={r[100]:.4f} R@500={r[500]:.4f} QPS={info['qps']:.0f} dead_query_vecs={dead} "
                  f"({(time.time()-tm)/60:.1f}min)", flush=True)
            out.append(dict(dim=m, bits=b, bytes=byt, r10=r[10], r50=r[50], r100=r[100], r500=r[500], qps=info['qps']))
    if SMOKE:
        print("SMOKE OK", flush=True); return
    today = datetime.date.today().isoformat()
    note = (f"{CFG['dataset']} quantized learned signature (no retraining): per-dim 99.9pct scale table, symmetric "
            f"corpus+query quantization, bytes/polygon = d*bits/8; Stage-1, eval ALL held-out queries; M=20 efC=200 efS={EF}")
    for csvp in [f'{BASE}/NEW_RESULTS.csv', f'{BASE}/RESULTS_LOG.csv']:
        with open(csvp, 'a', newline='') as f:
            w = csv.writer(f)
            for o in out:
                tag = 'fp32' if o['bits'] == 32 else f"int{o['bits']}"
                w.writerow([today, f"{CFG['name']}_allq", f"{CFG['name']}-listwise-d{o['dim']}-{tag}", o['dim'], 'base', '',
                            round(o['r10'], 4), round(o['r50'], 4), round(o['r100'], 4), round(o['r500'], 4),
                            round(o['qps']), 'eval_quant.py', note + f"; {o['bytes']} B/polygon"])
    json.dump(out, open(f"{BASE}/results_{CFG['name']}_quant.json", 'w'), indent=1)
    print(f"QUANT DONE {CFG['name']} {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
