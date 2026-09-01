#!/usr/bin/env python3
"""ADS (Adaptive Dimension Selection) on top of listwise (10K), candidate #4 from
the loss-function exploration (SMEC, EMNLP 2025). Previously deferred as low
priority (reasoning: conflicts with "just slice a prefix" deployment simplicity) --
this actually implements and tests it instead of leaving it untried.

Core idea: static prefix truncation (today's approach: width-m embedding = first m
raw dims, always) implicitly assumes dimension importance already matches raw
index order. ADS instead learns a single importance ranking over all EMB=4096
output dims jointly with the encoder, and for each Matryoshka width m selects the
(learned) top-m most important dims -- not necessarily the first m -- via a
differentiable Gumbel/straight-through top-k relaxation.

Adaptation notes vs. the literal SMEC ADS mechanism (not independently re-derived
here, adapted in spirit): one shared learnable importance vector `imp` (size EMB),
used across all widths (rather than per-width-specific importance). For width m,
during training: scores = imp + Gumbel noise; hard = top-m indicator of scores;
soft = sigmoid(sharpness * (scores - kth_largest)) as the continuous relaxation;
mask = hard + soft - soft.detach() (straight-through estimator -- forward pass
uses the hard 0/1 mask, backward pass routes gradient through the soft one, so
`imp` receives a training signal even for dims just outside the current top-m).
Sigmoid sharpness anneals from soft (5) to sharp (50) over training, standard
Gumbel-softmax practice. At eval, no noise / hard top-m by final `imp` values
(deterministic) -- the selected column indices per width are then fixed constants,
exactly like a prefix slice, so deployment cost is unchanged (store m columns per
width instead of "the first m columns per width").

Same architecture / data / total epoch budget (40) / optimizer / listwise loss /
tau as run_matdistill_listwise_10k.py -- only the per-width dimension *selection*
differs (learned top-m vs. fixed first-m).

Launch: CUDA_VISIBLE_DEVICES=5 python run_matdistill_listwise_ads_10k.py"""
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
GUMBEL_NOISE_SCALE = 0.5
SHARPNESS_START, SHARPNESS_END = 5.0, 50.0
EF = 200; RB = 64; CAND_KS = [500, 1000]; SEED = 42
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_matdistill_listwise_ads_10k.pt'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)
def norm_masked(x): return x / x.sum(1, keepdim=True).clamp(min=1e-4)  # ADS-only: with near-random early
    # selection (imp starts at 0), a row's chosen top-m columns can plausibly be ALL exactly zero
    # post-ReLU -- far more likely here than under the trained first-m prefix the other scripts use.
    # 1e-10 lets 1/denom explode to ~1e10 in the backward pass and NaN the run; 1e-4 keeps it finite.
def emb_wj(a, b): l1 = torch.cdist(a, b, p=1); return (2.0 - l1) / (2.0 + l1)


@torch.no_grad()
def raw_wj(R, chunk=64):
    M = R.shape[0]; out = torch.empty(M, M, device=R.device)
    for i in range(0, M, chunk):
        ri = R[i:i + chunk]; mn = torch.minimum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2)
        mx = torch.maximum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2).clamp(min=1e-10); out[i:i + chunk] = mn / mx
    return out


NEG_INF = -1e9


def gumbel_like(x): return -torch.log(-torch.log(torch.rand_like(x).clamp(min=1e-10, max=1 - 1e-10)).clamp(min=1e-10))


def st_topk_mask(imp, m, sharpness, training):
    """Straight-through top-m mask over `imp` (size EMB,). Returns (EMB,) mask."""
    if training:
        scores = imp + GUMBEL_NOISE_SCALE * gumbel_like(imp)
    else:
        scores = imp
    topk_vals, topk_idx = torch.topk(scores, m)
    hard = torch.zeros_like(scores); hard[topk_idx] = 1.0
    if not training:
        return hard
    kth = topk_vals.min()
    soft = torch.sigmoid(sharpness * (scores - kth))
    return hard + soft - soft.detach()


def listwise_ads(z, tw, imp, sharpness, training):
    N = z.shape[0]
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    tw_m = tw.masked_fill(diag, NEG_INF)
    target = F.softmax(tw_m / TAU, dim=1)
    tot = 0.
    for m in PREFIXES:
        mask = st_topk_mask(imp, m, sharpness, training)
        zm = norm_masked(z * mask.unsqueeze(0))
        that = emb_wj(zm, zm).masked_fill(diag, NEG_INF)
        logp = F.log_softmax(that / TAU, dim=1)
        tot = tot + (-(target * logp).sum(1)).mean()
    return tot / len(PREFIXES)


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
    print(f"10K corpus={qs} q={len(qq)} dim={IN} loss=listwise+ADS tau={TAU} "
          f"noise_scale={GUMBEL_NOISE_SCALE}", flush=True)
    RAW = torch.from_numpy(np.ascontiguousarray(qt, dtype=np.float32)).to(DEV)
    VN = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEV)
    m = Net(IN).to(DEV)
    imp = nn.Parameter(torch.zeros(EMB, device=DEV))  # neutral init: no prior over dim importance
    opt = torch.optim.AdamW([
        {"params": m.parameters(), "weight_decay": WD},
        {"params": [imp], "weight_decay": 0.0, "lr": LR * 0.1},  # imp has no BatchNorm to keep it bounded;
    ], lr=LR)                                                    # first run at full LR diverged to NaN within epoch 1
    loader = DataLoader(DS(gt, qs), batch_size=B, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        sharpness = SHARPNESS_START + (SHARPNESS_END - SHARPNESS_START) * (ep - 1) / max(EPOCHS - 1, 1)
        m.train(); tl = st = 0
        for ai, pi in tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS} sharp={sharpness:.1f}", ncols=110, mininterval=10):
            ids = torch.cat([ai, pi]).to(DEV)
            z = m(VN[ids])
            with torch.no_grad(): tw = raw_wj(RAW[ids])
            loss = listwise_ads(z, tw, imp, sharpness, training=True)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_([imp], 1.0)  # separate clip: imp's own gradient norm was getting
            opt.step()                                  # diluted by the encoder's much larger combined norm
            with torch.no_grad(): imp.clamp_(-15.0, 15.0)  # bounded range: unconstrained param, no BN to self-limit
            tl += loss.item(); st += 1
        sch.step()
        with torch.no_grad():
            sel_preview = {p: sorted(torch.topk(imp, p).indices.tolist())[:5] for p in [256]}
        print(f"ep{ep:02d} listwise_ce={tl/st:.6f} sharp={sharpness:.1f} imp_range=[{imp.min().item():.3f},{imp.max().item():.3f}] "
              f"top256_sample={sel_preview[256]} {(time.time()-t0)/60:.1f}min", flush=True)
        if tl / st < best: best = tl / st; torch.save({"model": m.state_dict(), "imp": imp.detach().cpu()}, CKPT)

    ckpt = torch.load(CKPT, map_location=DEV, weights_only=True)
    m.load_state_dict(ckpt["model"]); m.eval(); imp_final = ckpt["imp"].to(DEV)
    with torch.no_grad():
        Z = torch.cat([m(VN[i:i + 1024]) for i in range(0, len(VN), 1024)])
    today = datetime.date.today().isoformat(); rows = []
    Cr = qt[:qs].astype(np.float32); Cr_sum = Cr.sum(1).astype(np.float32); Qr = qt[qs:].astype(np.float32)
    for k in PREFIXES:
        with torch.no_grad():
            mask = st_topk_mask(imp_final, k, 0, training=False)
            Zk = norm(Z * mask.unsqueeze(0))
        emb = Zk.cpu().numpy().astype(np.float32); ce = emb[:qs]; qe = emb[qs:]
        mk = max(max(CAND_KS), 500)
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, qs, mk)
        print(f"[ads d={k} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
        rows.append([k, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB); e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[ads d={k} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([k, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = (f"ADS (learnable top-m dim selection, straight-through Gumbel) ablation vs MatDistillListwise10K: "
            f"same arch/data/40ep/listwise loss, one shared learnable importance vector replaces static "
            f"first-m prefix truncation (candidate #4, previously deferred, now tested); RAW-WJ target; "
            f"tau={TAU}; efSearch={EF}")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for k, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "10k", f"ListwiseADS10K-d{k}", k, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_matdistill_listwise_ads_10k.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
