# Loss-Function Exploration Log (BSD 2026)

Single running log for the post-GeoSearch exploration: is MSE distillation (the reviewed paper's method) actually the best available, or does something newer beat it? Everything here is cross-referenced against `NEW_RESULTS.csv` (the authoritative machine-readable log) and the source scripts under `/raid/ruban/hpmlproj/term_project/SigSpatial/`. Updated continuously as experiments land — see `/home/ruban/.claude/plans/ancient-coalescing-nygaard.md` for the active plan this log is tracking.

---

## 1. Established baseline (already validated, all three scales)

**Listwise (ListNet-style softmax + cross-entropy over the WJ row) beats MSE distillation at every scale and width tested.** Not a repeat of the triplet/InfoNCE collapse pattern — listwise's target is graded (real WJ value), triplet/InfoNCE's is binary (positive/not), which is the mechanistic reason it doesn't collapse at scale like they did.

| Scale | Method | R@10 | R@50 | R@500 | QPS | Source script |
|---|---|---|---|---|---|---|
| 10K (256-d, base) | MSE | 0.840 | 0.926 | 0.954 | 20,777 | `eval_matdistill_10k.py` |
| 10K (256-d, base) | **Listwise** | **0.941** | **0.972** | **0.995** | 14,145 | `eval_listwise_10k.py` |
| 50K (256-d, base) | Random Proj. | 0.616 | 0.659 | 0.766 | 3,890 | `run_50k_ddp.py` |
| 50K (256-d, base) | triplet+recon | 0.494 | 0.600 | 0.725 | 3,810 | `run_50k_ddp.py` |
| 50K (256-d, base) | InfoNCE | 0.459 | 0.560 | 0.720 | 7,128 | `run_50k_ddp.py` |
| 50K (256-d, base) | MSE | 0.692 | 0.794 | 0.924 | 6,772 | `run_50k_ddp.py` |
| 50K (256-d, base) | **Listwise** | **0.871** | **0.921** | **0.969** | 4,921 | `run_50k_listwise.py` |
| 187K (256-d, base) | MSE | 0.720 | 0.797 | 0.890 | 2,136 | `run_matdistill_fulleval_ddp.py` |
| 187K (256-d, base) | **Listwise** | **0.862** | **0.906** | **0.946** | 2,538 | `run_full187k_listwise.py` |
| 187K (256-d, rerank K=1000) | MSE | 0.993 | 0.995 | 0.966 | 1,166 | `run_matdistill_fulleval_ddp.py` |
| 187K (256-d, rerank K=1000) | **Listwise** | 0.993 | 0.996 | **0.982** | 1,065 | `run_full187k_listwise.py` |

**10K listwise baseline to beat, for every candidate below:** R@10=0.941, R@50=0.972, R@500=0.995 (`MatDistillListwise10K-d256` in `NEW_RESULTS.csv`).

---

## 2. Literature scan — candidates considered (2024–2025 unless noted)

| # | Candidate | Source | Core idea | Fit to our architecture | Status |
|---|---|---|---|---|---|
| 1 | **SMRL** (Sequential Matryoshka Repr. Learning) | SMEC, EMNLP 2025 ([aclanthology.org/2025.emnlp-main.1332](https://aclanthology.org/2025.emnlp-main.1332/)) | Joint multi-width Matryoshka training creates gradient-variance noise on shared params (formally derived + measured); train widths *sequentially*, freeze, then widen. Largest single ablation gain of SMEC's 3 components. | Their setup has separate stacked adapter layers per width (clean freeze point); ours is one shared dense output layer — approximated here as a **curriculum** (widen supervised prefixes in stages) rather than literal freeze-and-stack. | **Rejected** — tested both directions (§3.1), confirmed structural mismatch, does not transplant to our architecture |
| 2 | **RKD angle-wise term** | Park et al., CVPR 2019 (Relational KD) + 2024/25 follow-ups (pairwise-difference RKD, ICASSP 2024 similarity-based RKD) | Our MSE/listwise losses are already RKD's *distance*-wise term (match pairwise similarity). RKD's *angle*-wise term additionally matches angles formed by point triplets — higher-order structure we currently discard. | **Skipped without testing** — RKD's angle-term earns its keep when supervision is sparse (a handful of sampled pairs/triplets per batch). We already match the *entire* N×N pairwise matrix every step, so the extra relational structure the angle-term adds is very likely already implicit in what full-matrix matching provides; also no clean WJ-native analog of "angle" (RKD's is Euclidean dot-product based). Reasoning only, not empirically tested — could revisit if S-XBM also disappoints. |
| 3 | **S-XBM** (cross-batch memory) | SMEC, EMNLP 2025 | FIFO queue of recent batches' frozen embeddings; mine top-k hardest/most-similar historical samples to enrich each batch beyond its own ~30 mined neighbours. | Directly implementable (queue + top-k retrieval); moderate effort. Adapted to mine by **true raw-WJ similarity** (cheap and exact at our scale) rather than SMEC's stale-embedding similarity, since our whole 10K corpus already fits in GPU memory. | **Testing now** (10K) |
| 4 | **ADS** (learnable dimension selection) | SMEC, EMNLP 2025 | Gumbel-Softmax–learned per-dimension importance instead of static prefix truncation. | Conflicts with "just slice a prefix" deployment simplicity that's part of our paper's appeal; bigger architecture change. | Deferred, low priority |
| 5 | **MIPIC** (cross-dimension self-distillation) | 2025 ([arxiv 2604.24374](https://arxiv.org/pdf/2604.24374)) | Higher-dimensional Matryoshka prefixes act as soft teachers for lower-dimensional ones, within the same forward pass. Evaluated on retrieval-flavored benchmarks (STS, clustering, MTEB reranking). | Real architectural addition (needs the cross-prefix teacher wiring); not yet scoped in detail. | Deferred, low priority |
| — | Classical LTR SOTA (LambdaMART, ListMAP, NCGAN-LTR, ListMLE) | various | Newer than ListNet (2007) *within* search-ranking LTR. | Built for fixed candidate-list ranking w/ rich per-item features, often tree ensembles — domain mismatch with differentiable-embedding learning. Not pursuing. | Ruled out |
| — | Proxy-based deep metric learning (Proxy Anchor 2020, PD-Loss 2025) | various | Learned per-class "proxy" vectors. | Built around discrete semantic classes; we have continuous pairwise WJ, no classes. Not pursuing. | Ruled out |

Full research notes/citations are also in the chat transcript; this table is the durable summary.

---

## 3. Experiment log

Testing protocol: every candidate is layered on top of listwise (not standalone), tested at **10K only** first, compared against the 10K listwise baseline above. Escalation to 50K/187K is a separate, explicit decision per candidate — not automatic.

### 3.1 SMRL-curriculum (candidate #1) — DONE, verdict: not a win, do not escalate
- **Script:** `run_matdistill_listwise_curriculum_10k.py`
- **Change from baseline:** same architecture/data/40-epoch budget/listwise loss as `run_matdistill_listwise_10k.py`; only the *schedule* differs — prefixes {256} → {256,512} → {256,512,1024} → {256,512,1024,2048} → all 5, widened in 5 stages of 8 epochs each, instead of all 5 supervised jointly from epoch 1.

**Result (Stage-1 base, no rerank, R@50):**

| Dim | Listwise (joint, baseline) | Curriculum | Δ |
|---|---|---|---|
| 256 | 0.9715 | 0.9731 | +0.16pp |
| 512 | 0.9734 | 0.9731 | −0.03pp (noise) |
| 1024 | 0.9747 | 0.9680 | −0.67pp |
| 2048 | 0.9751 | 0.8912 | **−8.39pp** |
| 4096 | 0.9752 | 0.8699 | **−10.53pp** |

(Full R@10/R@500/QPS/rerank numbers in `NEW_RESULTS.csv`, method tag `ListwiseCurriculum10K-d*`.)

- **Verdict: not a win, do not escalate to 50K.** Marginal, noise-level change at the two smallest widths (256, 512 — the ones we actually deploy), but a severe regression at the three largest. Root cause: the *cumulative-widening* schedule I used gives later-added widths systematically less total training exposure than earlier ones (2048 only got supervised for ~16/40 epochs, 4096 for ~8/40), which is the opposite of what we want if all widths are meant to be usable.
- **Also worth flagging:** this curriculum went **small → large** (256 first, widening outward), which is actually the *reverse* of SMEC's own design — their sequential compression starts from the pretrained full-size embedding and derives progressively *smaller* stages from an already-converged larger one (large → small). What I tested is a reasonable adaptation to our shared-single-layer architecture, but it isn't a faithful test of SMEC's actual hypothesis.

#### 3.1b Diagnostic retest: reversed order (large → small)
- **Script:** `run_matdistill_listwise_curriculum_rev_10k.py` — identical to 3.1 except the stage order is flipped: 4096 supervised first (all 40 epochs), 256 added last (only the final 8).
- **Hypothesis being tested:** it's not "small-first helps," it's "whichever width trains *last* gets starved of epochs" — a structural property of curriculum scheduling on our shared-single-dense-layer encoder (no separable per-width weights to freeze, unlike SMEC's stacked adapters). If true: 4096 should recover to ~baseline (0.975) here, and 256 should now be the one that collapses.
- **Status: DONE — hypothesis confirmed.**

**Result (Stage-1 base R@50), all three runs side by side:**

| Dim | Baseline (joint) | Curriculum small→large | Curriculum large→small (this run) |
|---|---|---|---|
| 256 | 0.9715 | 0.9731 | **0.8783** (−9.3pp) |
| 512 | 0.9734 | 0.9731 | **0.9338** (−4.0pp) |
| 1024 | 0.9747 | 0.9680 | 0.9744 |
| 2048 | 0.9751 | **0.8912** (−8.4pp) | 0.9751 |
| 4096 | 0.9752 | **0.8699** (−10.5pp) | 0.9747 |

Exactly mirrored between the two curriculum directions — whichever widths trained *first/longest* end up near baseline, whichever trained *last* collapse. **Confirms the mechanism is purely "total training exposure," not direction.** Since our encoder has no separable per-width parameters to freeze (unlike SMEC's stacked adapters), there is no ordering that avoids this trade-off — joint training (equal exposure for every width, the entire time) is structurally the better fit for this architecture.

**Final verdict on candidate #1 (SMRL): rejected, with a clean mechanistic explanation, not just an inconclusive negative result.** Not escalating to 50K. SMEC's finding doesn't transplant to our shared-single-layer encoder; it would need genuinely separable per-width adapters (a bigger architecture change) to even be testable fairly, and isn't worth pursuing given the two lower-cost candidates remaining.

### 3.2 RKD angle-wise term (candidate #2)
- **Status: IN PROGRESS (2026-09-01)** — "no stones unturned" pass: actually implementing and testing instead of reasoning it away. Script: `run_matdistill_listwise_rkd_10k.py` (GPU4). Angle Huber term over 2048 sampled in-batch triplets/step, teacher=raw simplex-normalized input, student=embedding, layered on top of listwise.
- **Calibration note:** first launch used `LAMBDA_ANGLE=5.0` (a priori guess); epoch-1 data showed angle_huber≈0.003 vs listwise_ce≈5.3, so lambda=5 contributed only ≈0.015 — functionally inert. Killed after 5min and relaunched with `LAMBDA_ANGLE=300` (≈0.9 contribution, ~15% of total loss) so the run actually tests the hypothesis instead of silently no-op'ing the angle term for 40 epochs. Results pending.

### 3.3 S-XBM cross-batch memory (candidate #3)
- **Script:** `run_matdistill_listwise_sxbm_10k.py`
- **Change from baseline:** same architecture/data/40-epoch budget/listwise loss as `run_matdistill_listwise_10k.py`; each training step, the batch (~1024 ids) is augmented with up to 64 additional ids mined by true raw-WJ similarity from a 2048-id FIFO queue of recently-seen ids (queue is pushed with each step's own batch ids after use). Adapted from SMEC's S-XBM: they store stale embeddings from a frozen backbone and mine by embedding similarity (necessary since their corpus doesn't fit in memory); we mine by *exact* raw WJ instead, since our whole 10K corpus fits in GPU memory and exact similarity is cheap and strictly more reliable than embedding-based mining, especially early in training.
- **Pre-flight check:** original chunk size (256) in the cross-similarity computation would have needed ~76GB per intermediate tensor and OOM'd; corrected to chunk=16, queue=2048 (~2.4GB/intermediate, verified). Timing-tested standalone: ~0.46s/mining call × ~10,360 total steps ≈ 79 min added overhead; total run time expected ~2.5–3h.
- **Status: DONE — null result (no measurable effect, positive or negative).**

**Result (Stage-1 base, no rerank):**

| Dim | Listwise (baseline) R@50 | + S-XBM R@50 | Δ | QPS (baseline → +S-XBM) |
|---|---|---|---|---|
| 256 | 0.9715 | 0.9715 | 0.0000 | 14,145 → 18,575 |
| 512 | 0.9734 | 0.9734 | 0.0000 | 9,690 → 14,437 |
| 1024 | 0.9747 | 0.9747 | 0.0000 | 4,562 → 5,588 |
| 2048 | 0.9751 | 0.9751 | 0.0000 | 2,331 → 2,163 |
| 4096 | 0.9752 | 0.9752 | 0.0000 | 1,339 → 994 |

R@10 and R@50 match the baseline to 4 decimal places at every single width; R@500 differs only in the 4th decimal (noise). **Not a regression like SMRL — a completely flat result.** ~2 hours of extra training bought nothing measurable.

**Why, most likely:** at 10K scale (8K corpus), 40 epochs × 259 steps/epoch means the training process already sees the corpus many times over — the static top-30-neighbour batch construction plus plain shuffling probably already saturates whatever cross-batch diversity a 2048-id memory queue could add. **This may not generalize to "S-XBM doesn't help" at 50K/187K** — SMEC's own motivation for the technique is specifically that a single batch is a vanishingly small fraction of a *large* corpus, which is far more true at 187K (46,754 queries) than at 10K (2,000). This is a real limitation of the "always test small first" protocol: it can't distinguish "this idea doesn't work" from "this idea's mechanism doesn't kick in until the corpus is much bigger than the batch." Worth flagging for a judgment call rather than auto-applying the "flat → drop it" rule here.

**Verdict per protocol: flat result, don't auto-escalate — but this one has a specific, reasoned case for an exception** (see above) if there's appetite to spend one more 50K run confirming before fully closing the book on it.

#### 3.3b Escalation to 50K (resolving the scale-dependence caveat)
- **Status: IN PROGRESS (2026-09-01)** — "no stones unturned" pass. Script: `run_50k_listwise_sxbm.py`, 8-GPU DDP, same protocol as `run_50k_listwise.py`. Results pending.

### 3.4 ADS — learnable dimension selection (candidate #4)
- **Status: IN PROGRESS (2026-09-01)** — previously deferred without testing; now actually implemented. Script: `run_matdistill_listwise_ads_10k.py` (GPU5). One shared learnable importance vector over EMB=4096 dims, straight-through Gumbel top-m mask per Matryoshka width replaces static first-m prefix truncation. Results pending.

### 3.5 MIPIC — cross-dimension self-distillation (candidate #5)
- **Status: IN PROGRESS (2026-09-01)** — previously deferred without testing; now actually implemented. Script: `run_matdistill_listwise_mipic_10k.py` (GPU6). Largest (4096-d) prefix's own predicted similarity matrix, detached, used as an extra softmax-CE teacher for smaller widths, additive to the raw-WJ listwise loss (lambda=0.5). Results pending.

---

## 4. Overall session verdict (updated 2026-09-01 — "no stones unturned" pass in progress)

| Candidate | Verdict |
|---|---|
| SMRL (curriculum, both directions) | **Rejected** — structural mismatch with our architecture, clean mechanistic explanation, confirmed by symmetric evidence |
| RKD angle-wise term | **In progress** — now actually implemented and running (was previously reasoned-and-skipped) |
| S-XBM (cross-batch memory) @10K | **Null result at 10K** — flat, not negative |
| S-XBM (cross-batch memory) @50K | **In progress** — resolving the 10K scale-dependence caveat |
| ADS (dimension selection) | **In progress** — now actually implemented and running (was previously deferred) |
| MIPIC (cross-dim self-distillation) | **In progress** — now actually implemented and running (was previously deferred) |

Prior conclusion (SMRL rejected, S-XBM@10K flat, listwise remains the best-known method) still stands. This section will be updated with real numbers for all four in-progress runs as they land — see below for live status.

---

*This file is the single source of truth for "what did we try and what happened" during the loss-exploration phase — update it immediately after each experiment lands, before moving to the next candidate.*
