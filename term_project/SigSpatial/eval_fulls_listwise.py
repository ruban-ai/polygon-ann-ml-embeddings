#!/usr/bin/env python3
"""Stage-1 (no rerank) evaluation for the 2026-10-01 full-scale runs. Same HNSW
(M=20, efConstruction=200, efSearch=200, WeightedJaccard space) and eval_recall
as run_wbfull_listwise.py; the rerank stage is dropped per the 2026-10-01 scope
decision (recall-focused paper). QPS is still recorded for completeness.
Appends to NEW_RESULTS.csv (listwise convention) and RESULTS_LOG.csv (master log).
Usage: python eval_fulls_listwise.py <cfg> [--smoke] [--threads N]"""
import sys, os, time, csv, json, datetime
import numpy as np, torch
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS, BASE, raw_matrix, rows, load_gt
from sota_experiment_common import nmslib_neighbors, eval_recall
import run_fulls_listwise_train as T   # Net / norm / gnorm, same definitions as training

CFG = CONFIGS[sys.argv[1]]; SMOKE = '--smoke' in sys.argv
THREADS = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 96
EF = 200; K = 500
CKPT = CFG['ckpt'].replace('.pt', '_SMOKE.pt') if SMOKE else CFG['ckpt']


def main():
    t0 = time.time(); DEV = torch.device('cuda:0')
    X = raw_matrix(CFG); IN = X.shape[1]
    corpus_n, total = CFG['corpus_n'], CFG['total']
    gt = load_gt(CFG)
    if SMOKE:   # small corpus + its own queries, only checks the plumbing end to end
        q_ids = np.arange(corpus_n, total)[:200]
        c_ids = np.unique(np.concatenate([np.arange(20_000)] + [np.array(gt[q][:50]) for q in q_ids]))
    else:
        q_ids = np.arange(corpus_n, total); c_ids = None
    enc = T.Net(IN).to(DEV); enc.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); enc.eval()
    maxp = max(CFG['eval_prefixes'])
    print(f"[{CFG['name']}] eval rows={total:,} corpus={corpus_n:,} queries={total-corpus_n:,} dim={IN} "
          f"eval_prefixes={CFG['eval_prefixes']} threads={THREADS} smoke={SMOKE}", flush=True)

    def embed(ids):
        Z = np.empty((len(ids), maxp), dtype=np.float32); CH = 8192; te = time.time()
        with torch.no_grad():
            for i in range(0, len(ids), CH):
                xb = T.gnorm(torch.from_numpy(rows(X, ids[i:i + CH])).to(DEV))
                Z[i:i + CH] = enc.embed(xb)[:, :maxp].float().cpu().numpy()
                if (i // CH) % 100 == 0:
                    el = time.time() - te
                    print(f"  embedded {min(i+CH, len(ids)):,}/{len(ids):,} ({el/60:.1f}min, "
                          f"ETA {el/max(i+CH,1)*(len(ids)-i-CH)/60:.1f}min)", flush=True)
        return Z

    cids = c_ids if c_ids is not None else np.arange(corpus_n)
    Zc = embed(cids); Zq = embed(q_ids)
    print(f"embedding done {(time.time()-t0)/60:.1f}min", flush=True)
    if c_ids is not None:   # remap GT ids into the smoke corpus positions
        pos = {int(c): i for i, c in enumerate(c_ids)}
        gt = {corpus_n + k: [pos[n] for n in gt[q] if n in pos] for k, q in enumerate(q_ids)}

    today = datetime.date.today().isoformat(); out = []
    for m in CFG['eval_prefixes']:
        tm = time.time()
        ce = T.norm(torch.from_numpy(Zc[:, :m])).numpy(); qe = T.norm(torch.from_numpy(Zq[:, :m])).numpy()
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=K, threads=THREADS,
                                    query_params={"efSearch": EF})
        r = eval_recall(gt, nb, corpus_n, K)
        print(f"[{CFG['name']}-listwise d={m} base ALLq] R@10={r[10]:.4f} R@50={r[50]:.4f} R@100={r[100]:.4f} "
              f"R@500={r[500]:.4f} HNSW_QPS={info['qps']:.0f} build={info['build_s']/60:.1f}min "
              f"(width total {(time.time()-tm)/60:.1f}min)", flush=True)
        out.append(dict(dim=m, r10=r[10], r50=r[50], r100=r[100], r500=r[500], qps=info['qps'],
                        build_min=info['build_s'] / 60))
    if SMOKE:
        print(f"SMOKE OK {(time.time()-t0)/60:.1f}min", flush=True); return

    note = (f"{CFG['dataset']} FULL-SCALE listwise (2026-10-01 queue), trained on {CFG['train_n']:,} corpus rows, "
            f"indexed {corpus_n:,}, eval ALL {total-corpus_n:,} held-out queries; Stage-1 only (no rerank, scope "
            f"decision 2026-10-01); prefixes={CFG['prefixes']}; tau=0.1; M=20 efC=200 efS={EF}")
    if CFG['name'] == 'ov1m':
        note += "; GT depth ~100 (min 71) so R@500 = coverage of the full GT list"
    for csvp in [f'{BASE}/NEW_RESULTS.csv', f'{BASE}/RESULTS_LOG.csv']:
        with open(csvp, 'a', newline='') as f:
            w = csv.writer(f)
            for o in out:
                w.writerow([today, f"{CFG['name']}_allq", f"{CFG['name']}-listwise-d{o['dim']}", o['dim'], 'base', '',
                            round(o['r10'], 4), round(o['r50'], 4), round(o['r100'], 4), round(o['r500'], 4),
                            round(o['qps']), 'eval_fulls_listwise.py', note])
    json.dump(out, open(f"{BASE}/results_{CFG['name']}_listwise.json", 'w'), indent=1)
    print(f"EVAL DONE {CFG['name']} -> NEW_RESULTS.csv, RESULTS_LOG.csv, results_{CFG['name']}_listwise.json "
          f"total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
