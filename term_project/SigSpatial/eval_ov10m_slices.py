#!/usr/bin/env python3
"""Overture recall vs INDEX SIZE with one fixed encoder (2026-10-02 probe + the
dataset-size scaling curve Prof. Prasad asked for). Uses the 10M encoder
(best_ov10m_listwise.pt, trained on the first 1M corpus rows) and indexes the first
S corpus rows for S in {1M, 2.5M, 5M, 9.999M}; same 1,000 held-out queries.
GT for a slice = the 10M GT list restricted to ids < S (the list is ranked
closest-first over the whole corpus, so the restriction is exactly the true ranking
inside the slice). R@k is computed only over queries with >= k in-slice GT
neighbours (count reported), so shallow slices do not distort it.
If recall at S=1M is close to the 1M-encoding result while it falls with S, the
1M->10M drop is an index-size (HNSW) effect; if it is already low at 1M, the
encoder (trained on 10% of the corpus) is the cause.
Usage: python eval_ov10m_slices.py [--cfg ov10m] [--prefixes 64,256] [--threads N] [--smoke]"""
import sys, time, csv, json, datetime
import numpy as np, torch
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS, BASE, raw_matrix, rows, load_gt
from sota_experiment_common import nmslib_neighbors
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
CFGN = arg('--cfg', 'ov10m'); sys.argv = [sys.argv[0], CFGN] + sys.argv[1:]   # train module reads argv[1]
import run_fulls_listwise_train as T

CFG = CONFIGS[CFGN]; SMOKE = '--smoke' in sys.argv
PREF = [int(x) for x in arg('--prefixes', '64,256').split(',')]
THREADS = int(arg('--threads', '96')); EF = 200; K = 500
SLICES = [20_000, 50_000] if SMOKE else [1_000_000, 2_500_000, 5_000_000, 9_999_000]


def main():
    t0 = time.time(); DEV = torch.device('cuda:0')
    X = raw_matrix(CFG); IN = X.shape[1]; corpus_n, total = CFG['corpus_n'], CFG['total']
    gt = load_gt(CFG)
    enc = T.Net(IN).to(DEV); enc.load_state_dict(torch.load(CFG['ckpt'], map_location=DEV, weights_only=True)); enc.eval()
    maxp = max(PREF); smax = max(SLICES)
    ids = np.concatenate([np.arange(smax), np.arange(corpus_n, total)])
    Z = np.empty((len(ids), maxp), dtype=np.float32); te = time.time()
    with torch.no_grad():
        for i in range(0, len(ids), 8192):
            Z[i:i + 8192] = enc.embed(T.gnorm(torch.from_numpy(rows(X, ids[i:i + 8192])).to(DEV)))[:, :maxp].cpu().numpy()
            if (i // 8192) % 200 == 0:
                print(f"  embedded {min(i+8192, len(ids)):,}/{len(ids):,} ({(time.time()-te)/60:.1f}min)", flush=True)
    Zc, Zq = Z[:smax], Z[smax:]; qkeys = list(range(corpus_n, total)); out = []
    for S in SLICES:
        gS = [[n for n in gt[q] if n < S] for q in qkeys]
        for m in PREF:
            tm = time.time()
            ce = T.norm(torch.from_numpy(Zc[:S, :m])).numpy(); qe = T.norm(torch.from_numpy(Zq[:, :m])).numpy()
            nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=K, threads=THREADS, query_params={"efSearch": EF})
            res = {}
            for k in (10, 50, 100):
                v = [len(set(g[:k]) & set(list(r)[:k])) / k for g, r in zip(gS, nb) if len(g) >= k]
                res[k] = (float(np.mean(v)) if v else float('nan'), len(v))
            print(f"[{CFGN} slice S={S:,} d={m}] R@10={res[10][0]:.4f} (n={res[10][1]}) R@50={res[50][0]:.4f} "
                  f"(n={res[50][1]}) R@100={res[100][0]:.4f} (n={res[100][1]}) QPS={info['qps']:.0f} "
                  f"build={info['build_s']/60:.1f}min ({(time.time()-tm)/60:.1f}min)", flush=True)
            out.append(dict(S=S, dim=m, **{f'r{k}': res[k][0] for k in res}, **{f'n{k}': res[k][1] for k in res},
                            qps=info['qps']))
    if SMOKE:
        print("SMOKE OK", flush=True); return
    today = datetime.date.today().isoformat()
    note = (f"Overture recall vs index size, one encoder ({CFG['ckpt'].split('/')[-1]}, trained on "
            f"{CFG['train_n']:,} corpus rows), first S corpus rows indexed, same 1K queries, GT restricted to the "
            f"slice, R@k over queries with >=k in-slice GT; Stage-1; M=20 efC=200 efS={EF}")
    for csvp in [f'{BASE}/NEW_RESULTS.csv', f'{BASE}/RESULTS_LOG.csv']:
        with open(csvp, 'a', newline='') as f:
            w = csv.writer(f)
            for o in out:
                w.writerow([today, f"{CFGN}_slice{o['S']}", f"{CFGN}-listwise-d{o['dim']}-S{o['S']}", o['dim'], 'base', '',
                            round(o['r10'], 4), round(o['r50'], 4), round(o['r100'], 4), '', round(o['qps']),
                            'eval_ov10m_slices.py', note + f"; n@50={o['n50']}"])
    json.dump(out, open(f"{BASE}/results_{CFGN}_slices.json", 'w'), indent=1)
    print(f"SLICES DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
