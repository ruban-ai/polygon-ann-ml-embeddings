#!/usr/bin/env python3
"""RKD angle-wise term on top of listwise (10K), candidate #2 from the loss-function
exploration (Relational KD, Park et al. CVPR 2019). Previously reasoned-and-skipped
in LOSS_EXPLORATION_LOG.md sec 3.2 on the theory that full N x N pairwise matching
(what listwise already does) makes the extra angle-wise structure redundant --
this actually implements and tests that assumption instead of taking it on faith.

RKD angle potential: for a sampled triplet (i, j, k) with j as the vertex, take
unit vectors e_ij = (t_i - t_j)/||t_i - t_j|| and e_kj = (t_k - t_j)/||t_k - t_j||,
then psi_A(i,j,k) = <e_ij, e_kj> (cosine of the angle at vertex j). The distillation
term is Huber(psi_A_teacher, psi_A_student) -- this captures triplet *geometry*
(relative angles) that a purely pairwise-similarity match (listwise's full matrix,
or RKD's own distance-wise term) cannot see by construction, since three points'
angles carry information beyond their three pairwise distances/similarities alone
only in the sense of *how the embedding space linearly arranges them* -- distances
alone fix a triangle's shape but not its higher-dimensional embedding, so matching
angles in the raw/teacher space is an extra constraint on the student's geometry.

Adaptation notes vs. the literal RKD paper: RKD's potentials are Euclidean/dot-
product based (designed for L2 embedding spaces), which is what's used here for
both teacher (raw simplex-normalized vectors) and student (compact embeddings) --
there is no direct WJ-native angle analog, so this stays faithful to RKD's own
Euclidean formulation as an auxiliary regularizer layered on top of the WJ-native
listwise term (which already handles the primary WJ-ranking objective).

Same architecture / data / total epoch budget (40) / optimizer / listwise loss /
tau as run_matdistill_listwise_10k.py -- adds an angle-wise Huber term per step,
computed on a fixed-size random subset of in-batch triplets (full enumeration is
O(N^3), infeasible; NUM_TRIPLETS=2048 keeps the added cost small relative to the
existing full-matrix listwise computation).

Launch: CUDA_VISIBLE_DEVICES=4 python run_matdistill_listwise_rkd_10k.py"""
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
NUM_TRIPLETS = 2048   # sampled per step, not the full O(N^3) enumeration
LAMBDA_ANGLE = 300.0  # recalibrated from an initial guess of 5.0 after observing epoch-1 magnitudes on a
                       # real run: raw angle Huber ~0.003 vs listwise CE ~5.3 (cosine-bounded potentials are
                       # naturally small-scale) -- lambda=5 made the angle term contribute ~0.015, functionally
                       # inert. 300x brings it to ~0.9, a real ~15% regularizer contribution without dominating.
HUBER_DELTA = 1.0
EF = 200; RB = 64; CAND_KS = [500, 1000]; SEED = 42
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_matdistill_listwise_rkd_10k.pt'
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


def listwise_term(z, tw):
    N = z.shape[0]
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    tw_m = tw.masked_fill(diag, NEG_INF)
    target = F.softmax(tw_m / TAU, dim=1)
    tot = 0.
    for m in PREFIXES:
        zm = norm(z[:, :m])
        that = emb_wj(zm, zm).masked_fill(diag, NEG_INF)
        logp = F.log_softmax(that / TAU, dim=1)
        tot = tot + (-(target * logp).sum(1)).mean()
    return tot / len(PREFIXES)


def angle_potential(X, i_idx, j_idx, k_idx, eps=1e-8):
    """psi_A(i,j,k) = cosine of the angle at vertex j, for rows X[i],X[j],X[k]."""
    e_ij = X[i_idx] - X[j_idx]; e_kj = X[k_idx] - X[j_idx]
    e_ij = e_ij / e_ij.norm(dim=1, keepdim=True).clamp(min=eps)
    e_kj = e_kj / e_kj.norm(dim=1, keepdim=True).clamp(min=eps)
    return (e_ij * e_kj).sum(1)


def sample_triplets(N, n, device):
    i = torch.randint(0, N, (n,), device=device)
    j = torch.randint(0, N, (n,), device=device)
    k = torch.randint(0, N, (n,), device=device)
    return i, j, k


def rkd_angle_term(z, vn, i_idx, j_idx, k_idx):
    """Angle Huber loss between teacher (raw simplex-normalized input, vn) and
    student (embedding z), averaged over Matryoshka prefixes like listwise_term."""
    with torch.no_grad():
        psi_t = angle_potential(vn, i_idx, j_idx, k_idx)
    tot = 0.
    for m in PREFIXES:
        zm = norm(z[:, :m])
        psi_s = angle_potential(zm, i_idx, j_idx, k_idx)
        tot = tot + F.huber_loss(psi_s, psi_t, delta=HUBER_DELTA)
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
    print(f"10K corpus={qs} q={len(qq)} dim={IN} loss=listwise+RKD-angle tau={TAU} "
          f"num_triplets={NUM_TRIPLETS} lambda_angle={LAMBDA_ANGLE}", flush=True)
    RAW = torch.from_numpy(np.ascontiguousarray(qt, dtype=np.float32)).to(DEV)
    VN = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEV)
    m = Net(IN).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    loader = DataLoader(DS(gt, qs), batch_size=B, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        m.train(); tl = tl_lw = tl_ang = st = 0
        for ai, pi in tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=100, mininterval=10):
            ids = torch.cat([ai, pi]).to(DEV)
            z = m(VN[ids])
            with torch.no_grad(): tw = raw_wj(RAW[ids])
            lw = listwise_term(z, tw)
            i_idx, j_idx, k_idx = sample_triplets(z.shape[0], NUM_TRIPLETS, DEV)
            ang = rkd_angle_term(z, VN[ids], i_idx, j_idx, k_idx)
            loss = lw + LAMBDA_ANGLE * ang
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += loss.item(); tl_lw += lw.item(); tl_ang += ang.item(); st += 1
        sch.step()
        print(f"ep{ep:02d} total={tl/st:.6f} listwise_ce={tl_lw/st:.6f} angle_huber={tl_ang/st:.6f} "
              f"{(time.time()-t0)/60:.1f}min", flush=True)
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
        print(f"[rkd d={k} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
        rows.append([k, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB); e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[rkd d={k} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([k, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = (f"RKD angle-wise term ablation vs MatDistillListwise10K: same arch/data/40ep/listwise loss, "
            f"plus lambda={LAMBDA_ANGLE}*Huber(angle_teacher,angle_student) over {NUM_TRIPLETS} sampled "
            f"triplets/step (candidate #2, previously skipped-on-reasoning, now tested); RAW-WJ target; "
            f"tau={TAU}; efSearch={EF}")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for k, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "10k", f"ListwiseRKD10K-d{k}", k, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_matdistill_listwise_rkd_10k.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
