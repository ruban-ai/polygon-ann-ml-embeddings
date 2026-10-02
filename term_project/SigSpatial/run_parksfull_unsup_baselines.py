#!/usr/bin/env python3
"""Untrained baselines at FULL Parks scale (187,019 corpus / 46,754 held-out queries),
256-d, Stage-1: random projection, PCA, NMF. Each method is ported unchanged from the
10K/50K implementations so the full-scale table is comparable:
  rp  : run_50k_ddp.py --method rp  (Gaussian W/sqrt(m) on RAW vectors, shifted_l1_simplex, l1-norm)
  pca : 24_pca_wj_simplex_512.ipynb (IncrementalPCA fit on the first 50K corpus rows, shifted_l1_simplex)
  nmf : 25_nmf_wj_512.ipynb (MiniBatchNMF batch 512, max_iter 200, init random, fit on first 50K
        corpus rows of max(x,0), transform all, l1_simplex)
Usage: python run_parksfull_unsup_baselines.py rp,pca,nmf [--dim 256] [--threads N] [--smoke]"""
import sys, time, csv, json, datetime
import numpy as np, torch
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS, BASE, raw_matrix, load_gt
from sota_experiment_common import nmslib_neighbors, eval_recall, l1_simplex, shifted_l1_simplex

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
METHODS = sys.argv[1].split(','); DIM = int(arg('--dim', '256')); THREADS = int(arg('--threads', '64'))
SMOKE = '--smoke' in sys.argv; SEED = 42; EF = 200; K = 500
CFG = CONFIGS['parksfull']


def main():
    t0 = time.time()
    X = raw_matrix(CFG); corpus_n, total = CFG['corpus_n'], CFG['total']
    gt = load_gt(CFG)
    n_all = 30_000 if SMOKE else total
    cn = 25_000 if SMOKE else corpus_n
    if SMOKE:   # fake small split purely for plumbing
        gt = {cn + i: [n for n in gt.get(corpus_n + i, []) if n < cn] or [0] for i in range(n_all - cn)}
    print(f"[parksfull unsup] methods={METHODS} dim={DIM} rows={n_all:,} corpus={cn:,} smoke={SMOKE}", flush=True)
    out = []
    for meth in METHODS:
        tm = time.time()
        if meth == 'rp':
            rng = np.random.default_rng(SEED)
            W = torch.tensor((rng.standard_normal((X.shape[1], DIM)) / np.sqrt(DIM)).astype(np.float32), device='cuda:0')
            Z = np.concatenate([(torch.from_numpy(np.asarray(X[i:min(i + 4096, n_all)], dtype=np.float32)).to('cuda:0') @ W).cpu().numpy()
                                for i in range(0, n_all, 4096)])
            emb = l1_simplex(shifted_l1_simplex(Z))
        elif meth == 'pca':
            from sklearn.decomposition import IncrementalPCA
            fit_rows = min(cn, 50_000); bs = max(DIM, 2048)
            pca = IncrementalPCA(n_components=DIM, batch_size=bs)
            for s in range(0, fit_rows, bs):
                pca.partial_fit(np.asarray(X[s:min(s + bs, fit_rows)], dtype=np.float32))
                if (s // bs) % 5 == 0:
                    print(f"  pca partial_fit {min(s+bs, fit_rows):,}/{fit_rows:,} ({(time.time()-tm)/60:.1f}min)", flush=True)
            Z = np.concatenate([pca.transform(np.asarray(X[i:min(i + 8192, n_all)], dtype=np.float32)).astype(np.float32)
                                for i in range(0, n_all, 8192)])
            print(f"  explained variance ratio sum={pca.explained_variance_ratio_.sum():.4f}", flush=True)
            emb = shifted_l1_simplex(Z)
        elif meth == 'nmf':
            from sklearn.decomposition import MiniBatchNMF
            fit_rows = min(cn, 50_000)
            nmf = MiniBatchNMF(n_components=DIM, batch_size=512, max_iter=20 if SMOKE else 200, init='random',
                               random_state=SEED, verbose=1)
            nmf.fit(np.maximum(np.asarray(X[:fit_rows], dtype=np.float32), 0))
            print(f"  nmf fit done ({(time.time()-tm)/60:.1f}min), transforming {n_all:,} rows", flush=True)
            parts = []
            for i in range(0, n_all, 8192):
                parts.append(nmf.transform(np.maximum(np.asarray(X[i:min(i + 8192, n_all)], dtype=np.float32), 0)).astype(np.float32))
                if (i // 8192) % 5 == 0:
                    print(f"  nmf transform {min(i+8192, n_all):,}/{n_all:,} ({(time.time()-tm)/60:.1f}min)", flush=True)
            emb = l1_simplex(np.concatenate(parts))
        else:
            raise SystemExit(f'unknown method {meth}')
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        nb, info = nmslib_neighbors(emb[:cn], emb[cn:n_all], space="WeightedJaccard", k=K, threads=THREADS,
                                    query_params={"efSearch": EF})
        r = eval_recall(gt, nb, cn, K)
        print(f"[parksfull-{meth} d={DIM} base ALLq] R@10={r[10]:.4f} R@50={r[50]:.4f} R@100={r[100]:.4f} "
              f"R@500={r[500]:.4f} HNSW_QPS={info['qps']:.0f} ({(time.time()-tm)/60:.1f}min)", flush=True)
        out.append(dict(method=meth, dim=DIM, r10=r[10], r50=r[50], r100=r[100], r500=r[500], qps=info['qps']))
        if not SMOKE:
            today = datetime.date.today().isoformat()
            for csvp in [f'{BASE}/NEW_RESULTS.csv', f'{BASE}/RESULTS_LOG.csv']:
                with open(csvp, 'a', newline='') as f:
                    csv.writer(f).writerow([today, 'parksfull_allq', f'parksfull-{meth}-d{DIM}', DIM, 'base', '',
                        round(r[10], 4), round(r[50], 4), round(r[100], 4), round(r[500], 4), round(info['qps']),
                        'run_parksfull_unsup_baselines.py',
                        f"Parks FULL {meth} baseline, ported unchanged from the 10K/50K implementation; Stage-1; "
                        f"eval ALL held-out queries; M=20 efC=200 efS={EF}"])
    if SMOKE:
        print("SMOKE OK", flush=True); return
    json.dump(out, open(f"{BASE}/results_parksfull_unsup_{'_'.join(METHODS)}.json", 'w'), indent=1)
    print(f"UNSUP DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
