#!/usr/bin/env python3
"""Diagnostic retest of run_matdistill_listwise_curriculum_10k.py: that run
widened small->large (256 first, 4096 last) and 4096 collapsed (R@50 0.975 ->
0.870), while 256 gained a hair (+0.16pp). Hypothesis: it's not "small-first is
good," it's "whichever width trains LAST gets starved of epochs" -- a
structural property of curriculum scheduling on our shared-single-dense-layer
encoder (no separable per-width weights to freeze, unlike SMEC's stacked
adapters). This script flips the order: 4096 first, 256 last, same 5-stage/
8-epoch-per-stage shape. If the hypothesis is right, 4096 should recover to
~baseline (0.975) and 256 should now be the one that collapses.

Launch: CUDA_VISIBLE_DEVICES=3 python run_matdistill_listwise_curriculum_rev_10k.py"""
import sys, time, csv, datetime, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (load_dataset_normalized, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)
DEV = torch.device('cuda:0'); THREADS = 32; torch.set_num_threads(THREADS)
PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096; B = 512; EPOCHS = 40; LR = 1e-3; WD = 1e-4; MAX_POS = 100
TAU = 0.1
EPOCHS_PER_STAGE = EPOCHS // len(PREFIXES)
EF = 200; RB = 64; CAND_KS = [500, 1000]
SEED = 42
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_matdistill_listwise_curriculum_rev_10k.pt'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)
def emb_wj(a, b): l1 = torch.cdist(a, b, p=1); return (2.0 - l1) / (2.0 + l1)


@torch.no_grad()
def raw_wj(R, chunk=64):
    M = R.shape[0]; out = torch.empty(M, M, device=R.device)
    for i in range(0, M, chunk):
        ri = R[i:i + chunk]; mn = torch.minimum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2)
        mx = torch.maximum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2).clamp(min=1e-10); out[i:i + chunk] = mn / mx
    return out


NEG_INF = -1e9


def listwise_curriculum(z, tw, active_prefixes):
    N = z.shape[0]
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    tw_m = tw.masked_fill(diag, NEG_INF)
    target = F.softmax(tw_m / TAU, dim=1)
    tot = 0.
    for m in active_prefixes:
        zm = norm(z[:, :m])
        that = emb_wj(zm, zm).masked_fill(diag, NEG_INF)
        logp = F.log_softmax(that / TAU, dim=1)
        tot = tot + (-(target * logp).sum(1)).mean()
    return tot / len(active_prefixes)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))


class DS(Dataset):
    def __init__(s, gt, qs):
        s.p = [(q, n) for q, nb in gt.items() for n in nb[:MAX_POS] if q >= qs and n < qs]; random.shuffle(s.p)
        print(f"pairs={len(s.p):,} steps/epoch~{len(s.p)//B}", flush=True)

    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.p[i]


def main():
    qt, gt, qs, cq, qq, cs, qtn = load_dataset_normalized('10k'); IN = qtn.shape[1]
    print(f"10K corpus={qs} q={len(qq)} dim={IN} loss=listwise-curriculum-REVERSED tau={TAU} "
          f"stages={len(PREFIXES)} epochs/stage={EPOCHS_PER_STAGE} order=large->small", flush=True)
    RAW = torch.from_numpy(np.ascontiguousarray(qt, dtype=np.float32)).to(DEV)
    VN = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEV)
    m = Net(IN).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    loader = DataLoader(DS(gt, qs), batch_size=B, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        stage = min((ep - 1) // EPOCHS_PER_STAGE, len(PREFIXES) - 1)
        active_prefixes = PREFIXES[len(PREFIXES) - 1 - stage:]  # REVERSED: {4096} -> {2048,4096} -> ... -> all 5
        m.train(); tl = st = 0
        for ai, pi in tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS} stage{stage} w={active_prefixes}", ncols=110, mininterval=10):
            ids = torch.cat([ai, pi]).to(DEV)
            z = m(VN[ids])
            with torch.no_grad(): tw = raw_wj(RAW[ids])
            loss = listwise_curriculum(z, tw, active_prefixes)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl += loss.item(); st += 1
        sch.step()
        print(f"ep{ep:02d} stage{stage} active={active_prefixes} listwise_ce={tl/st:.6f} {(time.time()-t0)/60:.1f}min", flush=True)
        if tl / st < best: best = tl / st; torch.save(m.state_dict(), CKPT)

    m.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); m.eval()
    with torch.no_grad():
        Z = torch.cat([m(VN[i:i + 1024]) for i in range(0, len(VN), 1024)])
    today = datetime.date.today().isoformat(); rows = []
    Cr = qt[:qs].astype(np.float32); Cr_sum = Cr.sum(1).astype(np.float32); Qr = qt[qs:].astype(np.float32)
    for k in PREFIXES:
        emb = norm(Z[:, :k]).cpu().numpy().astype(np.float32); ce = emb[:qs]; qe = emb[qs:]
        mk = max(max(CAND_KS), 500)
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, qs, mk)
        print(f"[curriculum-rev d={k} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
        rows.append([k, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB); e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[curriculum-rev d={k} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([k, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = (f"SMRL-curriculum REVERSED-order diagnostic (large->small: 4096 first, 256 last), vs "
            f"ListwiseCurriculum10K (small->large) and MatDistillListwise10K (joint): tests whether "
            f"'whichever width trains last gets starved' explains the earlier 2048/4096 collapse; "
            f"RAW-WJ target; tau={TAU}; efSearch={EF}")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for k, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "10k", f"ListwiseCurriculumRev10K-d{k}", k, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_matdistill_listwise_curriculum_rev_10k.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
