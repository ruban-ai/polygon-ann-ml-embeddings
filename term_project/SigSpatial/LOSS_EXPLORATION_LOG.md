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
| 2 | **RKD angle-wise term** | Park et al., CVPR 2019 (Relational KD) + 2024/25 follow-ups (pairwise-difference RKD, ICASSP 2024 similarity-based RKD) | Our MSE/listwise losses are already RKD's *distance*-wise term (match pairwise similarity). RKD's *angle*-wise term additionally matches angles formed by point triplets — higher-order structure we currently discard. | Additive loss term on the existing batch structure; moderate effort. | Queued (after SMRL verdict) |
| 3 | **S-XBM** (cross-batch memory) | SMEC, EMNLP 2025 | FIFO queue of recent batches' frozen embeddings; mine top-k hardest/most-similar historical samples to enrich each batch beyond its own ~30 mined neighbours. | Directly implementable (queue + top-k retrieval); moderate effort. | Queued |
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
- **Status:** not started (queued behind 3.1's verdict)

### 3.3 S-XBM cross-batch memory (candidate #3)
- **Status:** not started

---

*This file is the single source of truth for "what did we try and what happened" during the loss-exploration phase — update it immediately after each experiment lands, before moving to the next candidate.*
