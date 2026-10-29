#!/usr/bin/env python3
"""Shared config + data access for the 2026-10-01 full-scale runs (Overture 1M,
Overture 10M, Sports full). Protocol mirrors run_wbfull_listwise.py /
run_full187k_listwise.py exactly (same MLP, optimizer, listwise loss, tau,
MAX_POS, batch, HNSW params, eval_recall). Differences, all forced by the data:
  * inputs are memory-mapped (Sports is 87 GB; 8 DDP ranks x full copy would
    not fit), and l1-normalization is done on the GPU per batch / per tile,
    which is numerically identical to sota_experiment_common.l1_simplex;
  * Overture has D~3K, so its Matryoshka prefixes are {64..1024} (EMB=1024)
    instead of {256..4096} -- widths >= D are meaningless, and <=96-d is needed
    to beat ShapeToVec+'s 32x on bytes;
  * Overture 10M trains on the first 1M corpus rows of the 10M encoding (exact
    corpus kNN over 10M is infeasible) and then indexes all 9,999,000;
  * evaluation is Stage-1 only (no rerank) per the 2026-10-01 scope decision.
"""
import glob, os, pickle
import numpy as np

BASE = '/raid/ruban/hpmlproj/term_project/SigSpatial'
OV1M_X = '/raid/ssEncodingData/encoding/ov-raster512-encodings-1M/ov1m_X.npy'
OV10M_DIR = '/raid/ruban/Buddhi/encoding/ov-raster512-10M'
SPORTS_QT = f'{BASE}/qt_sportsfull.npy'

CONFIGS = {
    'ov1m': dict(dataset='overture_1M', train_n=999_000, corpus_n=999_000, total=1_000_000,
                 prefixes=[64, 128, 256, 512, 1024], emb=1024, eval_prefixes=[64, 128, 256, 512, 1024],
                 epochs=5, gt=f'{BASE}/gt_ov1m.pkl'),
    'ov10m': dict(dataset='overture_10M', train_n=1_000_000, corpus_n=9_999_000, total=10_000_000,
                  prefixes=[64, 128, 256, 512, 1024], emb=1024, eval_prefixes=[64, 128, 256],
                  epochs=5, gt=f'{BASE}/gt_ov10m.pkl'),
    'sportsfull': dict(dataset='sports_full', train_n=1_403_192, corpus_n=1_403_192, total=1_753_989,
                       prefixes=[256, 512, 1024, 2048, 4096], emb=4096,
                       eval_prefixes=[256, 512, 1024, 2048, 4096],
                       epochs=5, gt=f'{BASE}/gt_sportsfull.pkl'),
}
# eval/baseline-only configs for the earlier full-scale runs (same data those runs used)
CONFIGS['parksfull'] = dict(dataset='parks_full', train_n=187_019, corpus_n=187_019, total=233_773,
                            prefixes=[256, 512, 1024, 2048, 4096], emb=4096, eval_prefixes=[256, 4096],
                            epochs=5, gt=f'{BASE}/gt_lookup_full.pkl')
CONFIGS['wbfull'] = dict(dataset='wb_full', train_n=358_840, corpus_n=358_840, total=448_550,
                         prefixes=[256, 512, 1024, 2048, 4096], emb=4096, eval_prefixes=[256, 4096],
                         epochs=5, gt=f'{BASE}/gt_wbfull.pkl')
# 2026-10-03 Parks full: single-width controls (joint-vs-independent) and a {64..4096} Matryoshka run
_pk = {k: v for k, v in CONFIGS['parksfull'].items()}
CONFIGS['parksfull_i256'] = dict(_pk, prefixes=[256], emb=256, eval_prefixes=[256])
CONFIGS['parksfull_i4096'] = dict(_pk, prefixes=[4096], emb=4096, eval_prefixes=[4096])
CONFIGS['parksfull_m64'] = dict(_pk, prefixes=[64, 128, 256, 512, 1024, 2048, 4096], emb=4096,
                                eval_prefixes=[64, 128, 256, 4096])
# 2026-10-09 input-resolution sweep (Parks at ~3K/6K/12K quadtree dims; identical recipe to parksfull_m64)
for _t in ('3k', '6k', '12k', '18k'):   # 18k = qtree_vectors_full.npy (symlink), 7-GPU rerun of m64
    CONFIGS[f'parksres_{_t}'] = dict(_pk, prefixes=[64, 128, 256, 512, 1024, 2048, 4096], emb=4096,
                                     eval_prefixes=[64, 128, 256, 1024, 4096])
# 2026-10-10: Water bodies + Sports at ~3K input (same resolution as ShapeToVec+), widths 64..4096 like Parks
CONFIGS['wbres_3k'] = dict(CONFIGS['wbfull'], prefixes=[64, 128, 256, 512, 1024, 2048, 4096], emb=4096,
                           eval_prefixes=[64, 128, 256, 1024, 4096])
CONFIGS['sportsres_3k'] = dict(CONFIGS['sportsfull'], prefixes=[64, 128, 256, 512, 1024, 2048, 4096], emb=4096,
                               eval_prefixes=[64, 128, 256, 1024, 4096])
# 2026-10-02 Overture probes
CONFIGS['ov1m_e10'] = dict(CONFIGS['ov1m'], epochs=10)
CONFIGS['ov10m_t3m'] = dict(CONFIGS['ov10m'], train_n=3_000_000)

for k, c in CONFIGS.items():
    c['name'] = k
    c['knn'] = f'{BASE}/corpus_knn_{k}.npy'
    c['ckpt'] = f'{BASE}/best_{k}_listwise.pt'
CONFIGS['ov1m_e10']['knn'] = CONFIGS['ov1m']['knn']            # same corpus, reuse mined kNN
for _k in ('parksfull_i256', 'parksfull_i4096', 'parksfull_m64'):
    CONFIGS[_k]['knn'] = CONFIGS['parksfull']['knn']           # same corpus, reuse the existing kNN
CONFIGS['parksres_18k']['knn'] = CONFIGS['parksfull']['knn']    # same 18K vectors, reuse mined kNN
CONFIGS['parksfull']['ckpt'] = f'{BASE}/best_full187k_listwise.pt'   # run_full187k_listwise.py output
PARKS_QT = f'{BASE}/qtree_vectors_full.npy'
WB_QT = f'{BASE}/qt_wbfull.npy'


class ShardedRows:
    """Row-indexable view over several memory-mapped .npy shards (in id order)."""
    def __init__(self, paths):
        self.parts = [np.load(p, mmap_mode='r') for p in paths]
        self.starts = np.cumsum([0] + [p.shape[0] for p in self.parts])
        self.shape = (int(self.starts[-1]), self.parts[0].shape[1])

    def __len__(self): return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            a, b, _ = idx.indices(self.shape[0])
            return self.take(np.arange(a, b))
        return self.take(np.asarray(idx))

    def take(self, idx):
        out = np.empty((len(idx), self.shape[1]), dtype=np.float32)
        s = np.searchsorted(self.starts, idx, side='right') - 1
        for k in np.unique(s):
            m = s == k
            out[m] = self.parts[k][idx[m] - self.starts[k]]
        return out


def raw_matrix(cfg):
    """Raw (unnormalized) ShapeToVec vectors for ALL rows, id order, memory-mapped."""
    if cfg['name'] in ('ov1m', 'ov1m_e10'):
        return np.load(OV1M_X, mmap_mode='r')
    if cfg['name'].startswith(('parksres_', 'wbres_', 'sportsres_')):
        return np.load(f"{BASE}/qt_{cfg['name']}.npy", mmap_mode='r')
    if cfg['name'].startswith('parksfull'):
        return np.load(PARKS_QT, mmap_mode='r')
    if cfg['name'] == 'wbfull':
        return np.load(WB_QT, mmap_mode='r')
    if cfg['name'] in ('ov10m', 'ov10m_t3m'):
        fs = sorted(glob.glob(f'{OV10M_DIR}/real_*.npy'),
                    key=lambda f: int(os.path.basename(f)[5:-4]))
        return ShardedRows(fs)
    if cfg['name'] == 'sportsfull':
        return np.load(SPORTS_QT, mmap_mode='r')
    raise KeyError(cfg['name'])


def rows(X, idx):
    """Gather rows as contiguous float32 (sorted gather is faster on memmaps)."""
    idx = np.asarray(idx)
    order = np.argsort(idx, kind='stable')
    out = np.empty((len(idx), X.shape[1]), dtype=np.float32)
    out[order] = X[idx[order]] if not isinstance(X, ShardedRows) else X.take(idx[order])
    return out


def load_gt(cfg):
    return pickle.load(open(cfg['gt'], 'rb'))
