#!/usr/bin/env python3
"""2026-10-26 v2 (Prasad: 1M 13.8K < 10M 14.5K QPS looked odd; 2026-10-12 runs varied 8.4K-16.5K because one
1,000-query batch at 64 threads lasts ~0.07 s). Same build/kernel/params as bench_are.py; throughput now = 1 discarded
warm-up + 20 timed runs (median, IQR) + a long run over the query set tiled 20x (20,000 queries). Writes results_bench_are_v2_*.json.
2026-10-12 (Prasad): query throughput/latency and index memory of our 8-bit signatures on Overture,
measured with ShapeToVec+'s own SIMD integer kernel (patched NMSLIB space 'WeightedJaccardRatio', the ARE
space) so the comparison with ShapeToVec+ uses the same code, machine, and HNSW parameters.
Run with Buddhi's env: /home/buddhi/anaconda3/envs/quant-nmslib2/bin/python bench_are.py <cfg> <d>
HNSW M=20, efConstruction=200, post=1 (as in all our evals), efSearch=200, k=50.
Throughput: knnQueryBatch over all held-out queries with 64 threads (ShapeToVec+ uses 64), mean of 5 runs.
Latency: single-thread knnQuery per query, mean and p95. Memory: saved index file (graph + data)."""
import os, sys, time, pickle, json, numpy as np, nmslib

BASE = '/raid/ruban/hpmlproj/term_project/SigSpatial'
cfg, d = sys.argv[1], int(sys.argv[2])
GT = {'ov1m': f'{BASE}/gt_ov1m.pkl', 'ov10m': f'{BASE}/gt_ov10m.pkl'}[cfg]
CORPUS_N = {'ov1m': 999_000, 'ov10m': 9_999_000}[cfg]
THREADS, EF, K = 64, 200, 50
OUT = f'{BASE}/are_codes'

def pack(u8):
    pad = (-u8.shape[1]) % 4
    if pad: u8 = np.pad(u8, ((0, 0), (0, pad)))
    return np.ascontiguousarray(u8).view(np.float32)

t0 = time.time()
cc = np.load(f'{OUT}/{cfg}_d{d}_corpus.npy'); cq = np.load(f'{OUT}/{cfg}_d{d}_query.npy'); s = np.load(f'{OUT}/{cfg}_d{d}_scales.npy')
assert cc.shape[0] == CORPUS_N, cc.shape
w = np.pad(s.astype(np.float32), (0, (-len(s)) % 4)); wfile = f'{OUT}/{cfg}_d{d}_weights.bin'; w.tofile(wfile)
pc, pq = pack(cc), pack(cq)
print(f"[{cfg} d={d}] corpus {cc.shape} -> packed {pc.shape} float32 ({pc.nbytes/2**20:.1f} MiB payload), queries {cq.shape[0]}", flush=True)

idx = nmslib.init(method='hnsw', space='WeightedJaccardRatio', space_params={'weights_file': wfile},
                  data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
idx.addDataPointBatch(pc)
tb = time.time(); idx.createIndex({'M': 20, 'indexThreadQty': THREADS, 'efConstruction': 200, 'post': 1}, print_progress=False)
build_s = time.time() - tb
print(f"  built in {build_s/60:.1f} min", flush=True)
idx.setQueryTimeParams({'efSearch': EF})

idx.knnQueryBatch(pq, k=K, num_threads=THREADS)   # warm-up, discarded
qps = []
for r in range(20):
    tq = time.perf_counter(); res = idx.knnQueryBatch(pq, k=K, num_threads=THREADS); qps.append(len(pq) / (time.perf_counter() - tq))
    print(f"  run {r+1}/20: {qps[-1]:.0f} QPS", flush=True)
q25, q50, q75 = np.percentile(qps, [25, 50, 75])
print(f"  QPS (64 threads, 20 runs of 1,000): median {q50:.0f}  IQR {q25:.0f}-{q75:.0f}  min {min(qps):.0f}  max {max(qps):.0f}", flush=True)
pl = np.ascontiguousarray(np.tile(pq, (20, 1)))
long_qps = []
for r in range(3):
    tq = time.perf_counter(); idx.knnQueryBatch(pl, k=K, num_threads=THREADS); long_qps.append(len(pl) / (time.perf_counter() - tq))
    print(f"  long run {r+1}/3 (20,000 queries): {long_qps[-1]:.0f} QPS", flush=True)
print(f"  QPS long (64 threads, 20,000 queries, 3 runs): mean {np.mean(long_qps):.0f}  runs {[round(x) for x in long_qps]}", flush=True)

lat = []
for i in range(len(pq)):
    tl = time.perf_counter(); idx.knnQuery(pq[i], k=K); lat.append((time.perf_counter() - tl) * 1e3)
lat = np.array(lat)
print(f"  latency (1 thread): mean {lat.mean():.3f} ms  p95 {np.percentile(lat,95):.3f} ms", flush=True)

gt = pickle.load(open(GT, 'rb'))
rec = {}
for k in (10, 50):
    vals = []
    for i in range(len(pq)):
        g = gt.get(CORPUS_N + i, [])[:k]
        if g: vals.append(len(set(g) & set(res[i][0][:k].tolist())) / len(g))
    rec[k] = float(np.mean(vals))
print(f"  recall R@10 {rec[10]:.4f}  R@50 {rec[50]:.4f}", flush=True)

ipath = f'{OUT}/{cfg}_d{d}_v2.hnsw'; idx.saveIndex(ipath, save_data=True)
size = sum(os.path.getsize(f) for f in [ipath, ipath + '.dat'] if os.path.exists(f))
print(f"  index on disk (graph + data): {size/2**30:.2f} GiB  = {size/CORPUS_N:.0f} B/polygon; signature payload {cc.shape[1]} B/polygon", flush=True)
json.dump(dict(cfg=cfg, d=d, build_min=build_s/60, qps_median=float(q50), qps_q25=float(q25), qps_q75=float(q75), qps_runs=qps, qps_long_mean=float(np.mean(long_qps)), qps_long_runs=long_qps, lat_ms_mean=float(lat.mean()),
               lat_ms_p95=float(np.percentile(lat, 95)), r10=rec[10], r50=rec[50], index_bytes=size, bytes_per_polygon=size/CORPUS_N),
          open(f'{BASE}/results_bench_are_v2_{cfg}_d{d}.json', 'w'), indent=1)
for f in [ipath, ipath + '.dat']:
    if os.path.exists(f): os.remove(f)
print(f"BENCH DONE {cfg} d={d} {(time.time()-t0)/60:.1f}min", flush=True)
