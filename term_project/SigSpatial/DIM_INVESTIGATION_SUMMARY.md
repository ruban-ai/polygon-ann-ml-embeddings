# Why higher embedding dimension does NOT improve WJ candidate-generation recall — full investigation

**Date:** 2026-06-16  ·  **Dataset:** `full` (boi/ssEncodingData polygons)  ·  **Context:** SIGSPATIAL 2026 short paper on Weighted-Jaccard (WJ) ANN over polygons. Two-stage retrieval: Stage-1 = cheap learned embedding + HNSW(WeightedJaccard) candidate generation; Stage-2 = exact-WJ GPU rerank on raw 18,220-d vectors.

This doc is a self-contained handoff (limit may run out; may continue in Codex/Cursor).

---

## 0. TL;DR — what we proved

1. **Higher *output* dimension does not help the learned embeddings** (triplet R@500 = 0.654 at 512-d AND 2048-d; InfoNCE tracks ~0.73 at 512-d and 2048-d; effective rank ~37–240 regardless of output width). Confirmed on BOTH full (187K) and 10K.
2. **It is NOT the index, NOT precision, NOT the recon term, NOT BatchNorm, NOT raw capacity** — all ruled out by controlled experiments.
3. **★ THE BIG ONE (added later): it's the OUTPUT COMPRESSION, not the objective.** Tapping the model's *intermediate* layers as the embedding shows recall RISES as you go wider/earlier: triplet full-set R@500 = **0.654 (512 output) → 0.777 (1024 layer) → 0.818 (4096 layer)**. The narrow output bottleneck (where the loss is applied) is what destroys recall. The wide 4096 layer of the **triplet+recon** model is the **best** candidate generator we have (R@500 0.818, beating InfoNCE-4096 0.778 and random-4096 0.772).
4. **Mechanism:** recall ∝ Spearman(emb-WJ, raw-18220-WJ). The loss, applied at the narrow output, *damages* that layer's WJ ordering (triplet output Spearman 0.62–0.82) while upstream wide layers keep it (0.89–0.94). Proof that dimension itself is useful: a *training-free random projection* improves with dim (R@500 0.689→0.772 over 256→4096) — so a metric-preserving wide map benefits from width; the narrow learned output does not.
5. **Both objectives "work" — triplet+recon is NOT inferior.** Earlier "InfoNCE is the method / triplet doesn't scale" was an artifact of comparing only the *narrow outputs* (where triplet's hard margin damages geometry most). At the wide layer, **triplet+recon wins** (10K R@50 0.85–0.88 vs InfoNCE 0.70–0.75; full 4096 R@500 0.818 vs 0.778). The real lever is **width / which layer**, not the objective.
6. **★★ DEEPEST finding (see §4c): the loss DAMAGES the layer it sits on.** Within one model, the layer with NO loss beats the loss layer by 8–10 pts; the trained OUTPUT is even *worse than an untrained random projection*; the no-loss early layer *beats* random projection. Best embedding found = **InfoNCE Matryoshka's first-4096 layer = R@500 0.832** (a layer with no loss on it). Triplet/InfoNCE are proxy objectives misaligned with broad WJ recall; the layer they're applied to becomes a distorted "task-head."
7. **Fix / path forward:** (a) deploy the early wide layer, not the output; (b) **switch to a WJ-distillation objective** (make emb-WJ match raw-WJ ordering) so the OUTPUT is no longer distorted — user has `*wjdistill*` checkpoints to test; (c) distillation+Matryoshka for truncatable dims with no damage. Matryoshka with the *contrastive* loss exposes dims but still damages each output.
8. **Separate critical bug (root-caused earlier):** the first 2048 triplet run *collapsed* because it used `max_pos=256` (from the InfoNCE runner). Triplet's hard margin is unsatisfiable with broad positives → identical embeddings. **Triplet needs `max_pos=30`.** Fixed; the 2048 numbers are post-fix and healthy. InfoNCE *does* tolerate max_pos=256.

---

## 1. Data provenance (VERIFIED bit-identical to source)

| | size | our cache | source (verified identical) |
|---|---|---|---|
| Corpus (indexed) | **187,019** polygons (80%) | `/tmp/qtree_vectors_full.npy` rows `[:187019]` | `/raid/ssEncodingData/encoding/pk-real0.002/` (`real_*.txt`, dense 18,220-d) |
| Queries | **46,754** (20%) | rows `[187019:]` | same encoding dir |
| Total | 233,773 × 18,220-d | `/tmp/qtree_vectors_full.npy` (17 GB, raw quadtree) | — |
| Ground truth | dict, 44,666 queries w/ GT, **mean 4,867 neighbors/query** (max 17,325) | `/tmp/gt_lookup_full.pkl` (930 MB) | `/raid/ssEncodingData/warehouse/pk-query-187019/similarityMap_*` |

- Input encoding = quadtree resolution **real0.002 → 18,220-d** (smaller res value = finer grid → more dims). Alternatives exist for the SAME dataset/GT: `real0.003`=12,060-d, `real0.006`=6,004-d (in `/raid/ruban/boi_multi_index/Dataset_FloatingPoint/boi_cache/`). GT is grid-independent (true geometric shape similarity), so coarser grids only need retraining.
- **Verified:** `real_0.txt` line-1 == `qt[0]` exactly (all 18,220 dims, 0 diff); `gt_lookup_full.pkl` == `similarityMap` neighbor lists exactly for qids 187019/187020/187418 (incl. a 15,029-element list). The pipeline data IS the ssEncodingData source.
- `qt_norm` (used by models) = `l1_simplex(qt)` (non-negative, rows sum to 1 → the natural WJ simplex). Cached `/tmp/qt_norm_full.npy`.

---

## 2. The collapse bug (root-caused earlier) and the fix

**Symptom:** triplet+recon @2048 trained with `max_pos=256` → loss pinned at exactly the margin (0.3000), `viol=1.000` every step, embedding row-diversity → ~0 (identical embeddings), recall ~random.

**Cause:** the margin triplet needs `WJ(anchor,pos) > WJ(anchor,hardest_neg)+margin` for *every* pair. With `max_pos=256`, "positives" include weakly-similar 31st–256th neighbors → constraint unsatisfiable → only loss-reducing move is to collapse all embeddings together.

**Ruled out (controlled runs, all still collapsed):** output dim (512 collapses too), cdist-vs-min/max loss (forward+gradient identical, cosine 0.9999), bf16, recon/decoder (lam=0 collapses), BatchNorm batching (separate fwd collapses), lr/wd/margin.

**Fix = `max_pos=30`** (the original healthy nb28 setting). Verified: 512-d max_pos=256 → embdiv 0.02 (collapse); max_pos=30 → embdiv 0.92, viol<1, loss<margin (healthy). InfoNCE *does* tolerate max_pos=256 (softmax pulls positives relatively, not absolutely) — do NOT copy max_pos between the two objectives.

Memory note saved: `~/.claude/.../memory/project_triplet_maxpos_collapse.md`.

---

## 3. Headline result — 2048 vs 512 (post-fix, healthy), full corpus

Method = triplet+recon AE (`AE-WJ-tri+recon`). Same architecture, only OUT_DIM differs.

| dim | base R@50 | base R@500 | HNSW QPS | rerank@1k R@50 | rerank@2k R@50 |
|---|---|---|---|---|---|
| 512 | 0.656 | 0.656 | 1188 | 0.989 | 0.991 |
| **2048** | 0.666 | **0.667** | 1107 | 0.990 | 0.992 |

→ **~1 point difference = noise. Quadrupling dimension does nothing.** (Rerank fixes top-50 to ~0.99 regardless, because Stage-2 uses exact raw-WJ.)

For reference (512-d, full): `InfoNCE base R@500=0.727`, `RandomProjection base R@500=0.722` — both **beat** the learned triplet (0.656). i.e., our trained embedding loses to a random projection at Stage-1.

---

## 4. The analysis — proof it's embedding quality (3 stages)

All probes: `/tmp/analysis_stage{1,2,3}.py`, run on a spare GPU with `CUDA_VISIBLE_DEVICES=1`. Recall = rank-matched `|gt[:k] ∩ retrieved[:k]| / k` (matches pipeline `eval_recall`).

### Stage 1 — is the ceiling the embedding or the HNSW index? + capacity
| config | dim | exact R@500 | HNSW R@500 | eff. rank | dims@90%var |
|---|---|---|---|---|---|
| triplet-2048 | 2048 | 0.654 | 0.667 | 240.7 | 683 |
| triplet-512 | 512 | 0.654 | 0.656 | 151.9 | 268 |
| infonce-512 | 512 | 0.732 | 0.727 | **39.2** | 63 |

- **Index innocent:** exact brute-force kNN ≈ HNSW everywhere → the ceiling is in the embedding, not the index.
- **Capacity innocent (anti-correlated!):** the BEST method (InfoNCE) uses the FEWEST effective dims (39); the worst (triplet-2048) uses the most (241). More dims ≠ better.

### Stage 2 — WHY: WJ rank-preservation (Spearman of embedding-WJ vs true 18,220-d WJ)
| config | exact R@500 | eff rank | Spearman(emb-WJ, raw-WJ) | AUC(true>rand) |
|---|---|---|---|---|
| triplet-2048 | 0.654 | 240.7 | **0.43** | 0.998 |
| triplet-512 | 0.654 | 151.9 | **0.45** | 0.998 |
| infonce-512 | 0.732 | 39.2 | **0.80** | 0.999 |
| random-512 | 0.723 | 26.1 | **0.86** | 0.998 |

- Recall tracks Spearman almost perfectly. Triplet preserves the true WJ ordering **worse than a random projection** (0.44 vs 0.86) → its embedding is actively worse than doing nothing.

### Stage 3 — the decisive proof + ceiling + cause
- **(A) Ceiling:** exact raw-18,220-d WJ kNN vs GT → **R@50=0.996, R@500=0.998**. Task fully solvable at full dim; all loss is compression loss.
- **(B) Dimension HELPS a metric-preserving embedding but NOT the learned one:**

| embedding | dim | R@500 | broad Spearman |
|---|---|---|---|
| random-proj | 256 | 0.689 | 0.93 |
| random-proj | 512 | 0.723 | 0.93 |
| random-proj | 1024 | 0.746 | 0.94 |
| random-proj | 2048 | 0.758 | 0.94 |
| random-proj | 4096 | **0.772** | 0.95 |
| triplet (learned) | 512 | 0.654 | 0.82 |
| triplet (learned) | 2048 | **0.654** | 0.82 |

  Random projection climbs with dim; learned triplet is flat. **This is the proof: the limiter is the learned geometry, not the dimension.** (Random eff-rank stays ~25 regardless of dim — JL: more dims = less distortion = better recall even at constant rank.)
- **(C) Cause — Spearman by neighborhood depth:**

| embedding | sp@30 | sp@100 | sp@500 |
|---|---|---|---|
| triplet-512 | 0.14 | 0.40 | 0.82 |
| triplet-2048 | 0.12 | 0.39 | 0.82 |
| infonce-512 | 0.73 | 0.81 | 0.92 |

  Triplet's WJ-preservation is poor (and worst in the tight neighborhood); identical at 512 and 2048 (dims don't fix it). InfoNCE preserves it well at all depths.

---

## 4b. ★ BREAKTHROUGH — it's the OUTPUT COMPRESSION; wide intermediate layers win

Architecture is a funnel `18220 → 4096 → 1024 → OUT_DIM`. We tapped each layer (relu'd, L1-normalized → a WJ embedding) and measured recall. Probe: `/tmp/probe_layers.py` (full), `/tmp/analysis_10k.py` (10K).

**Full set (187K), R@500 / Spearman@500:**
| model | 4096 layer | 1024 layer | output | 
|---|---|---|---|
| triplet-512 | **0.818** / 0.935 | 0.777 / 0.919 | 0.654 / 0.820 |
| InfoNCE-512 | 0.778 / 0.942 | 0.775 / 0.959 | 0.731 / 0.920 |
(ref: random-4096 0.772, ceiling 0.998)

→ Recall climbs monotonically as you go wider/earlier. The **narrow output (where the loss is applied) is what destroys recall**; the loss damages that layer's WJ ordering. **triplet+recon's 4096 layer (0.818) is the best embedding found.**

**10K (8K corpus, easy → R@500 saturates ~0.98; read R@50):**
| model | out | 4096 R@50 | output R@50 | output Spearman |
|---|---|---|---|---|
| triplet | 256 | 0.875 | 0.853 | **0.62** |
| triplet | 1024 | 0.874 | 0.854 | **0.62** |
| InfoNCE | 512 | 0.747 | 0.698 | 0.84 |

→ **10K's "stagnant recall across output dims" is the SAME mechanism:** output dim 256 vs 1024 gives identical R@50 (0.853 vs 0.854); the output Spearman is damaged to 0.62; the easy task just *masks* it (R@500 saturates). On 10K triplet (0.85–0.88) beats InfoNCE (0.70–0.75) — **triplet+recon is not inferior; it's better.**

**Implication:** the objective debate was a red herring at the output layer. Both objectives build good wide representations; the funnel's narrow output is the damage. Use a wide embedding. Trade-off: wider = slower/bigger HNSW (the recall↔throughput knob).

## 4c. ★★ DEEPEST FINDING — the loss DAMAGES the layer it is applied to (deploy the representation, not the task-head)

Probe: `/tmp/analysis_loss_damage.py` → `/tmp/loss_damage_analysis.log`. Full set; R@500 + Spearman(emb-WJ, raw-18220-WJ) at neighborhood depths 30/100/500.

| embedding / layer | dim | loss here? | R@500 | sp@30 | sp@100 | sp@500 |
|---|---|---|---|---|---|---|
| random-proj (no training) | 1024 | — | 0.744 | 0.854 | 0.896 | 0.938 |
| random-proj | 2048 | — | 0.757 | 0.873 | 0.911 | 0.942 |
| random-proj | 4096 | — | 0.770 | 0.874 | 0.914 | 0.946 |
| funnel-tri512: 4096 | 4096 | NO | 0.816 | 0.756 | 0.830 | 0.937 |
| funnel-tri512: 1024 | 1024 | NO | 0.775 | 0.675 | 0.779 | 0.919 |
| funnel-tri512: **512 OUTPUT** | 512 | **YES** | 0.652 | 0.143 | 0.405 | 0.817 |
| matry-tri: first-4096 | 4096 | NO | 0.797 | 0.476 | 0.637 | 0.895 |
| matry-tri: **out-4096** | 4096 | **YES** | 0.692 | 0.112 | 0.390 | 0.823 |
| matry-inf: first-4096 | 4096 | NO | **0.832** | 0.832 | 0.887 | 0.949 |
| matry-inf: **out-4096** | 4096 | **YES** | 0.752 | 0.710 | 0.796 | 0.921 |

**Three facts:**
1. **The loss layer is always the worst layer in its own model.** funnel declines toward the loss: 0.816 (4096) → 0.775 (1024) → 0.652 (512-OUTPUT). Both Matryoshkas drop ~0.08–0.10 from first→output. Same model, same depth-class → it's the **loss position**, not depth.
2. **The trained OUTPUT is worse than an UNTRAINED random projection of the same dim:** matry-inf out-4096 = 0.752 < random-4096 0.770; matry-tri out-4096 = 0.692 ≪ 0.770. The loss makes the deployed readout *worse than no learning at all*.
3. **The no-loss EARLY layer BEATS random projection:** matry-inf first-4096 = **0.832** > 0.770; funnel 4096 = 0.816 > 0.770. So learning helps — but only where the loss doesn't directly sit.

**Where the damage lives:** worst in the TIGHT neighborhood — sp@30 collapses at the output (funnel 0.143, matry-tri 0.112) vs the wide layer (0.48–0.83). Even broad sp@500 degrades (0.82 vs 0.94). The hard-margin **triplet damages far more** than **InfoNCE** (out sp@30: 0.112 vs 0.710) — its margin reshuffles the local geometry hardest.

**Mechanism:** triplet/InfoNCE are PROXY objectives rewarding *local separation* (positive vs hardest negative); recall needs *global WJ-rank preservation*. The layer the loss sits on becomes a **task-head specialized to the proxy** — it distorts the WJ geometry to win the margin/softmax. Earlier layers are the **representation** (rich, metric-preserving, better than random). For retrieval you want the representation, not the task-head. (Classic "the last layer is too task-specific" — and here the task is *misaligned* with retrieval, so the last layer is actively bad.)

**Answer to "what was missing/not right":** we deployed the WRONG layer (the proxy-distorted output) and, with Matryoshka/plain, put the loss ON the embedding we deploy. Best embedding found = matry-inf **first-4096 = 0.832** (a layer with no loss on it).

**Fixes (ranked):**
- **(now, free) Deploy the early wide layer, not the output** → 0.832.
- **(principled) WJ-distillation objective**: train the embedding so its WJ matrix matches the raw-18220 WJ ordering. Then the objective IS metric preservation → the OUTPUT stops being distorted → the deployable layer becomes the best layer, at any chosen dim. **User has `/tmp/best_compressor_*wjdistill*_10k.pt`, `*wj_native*`, `*listwise_wjdistill*` — TEST THESE NEXT** (does their OUTPUT avoid the damage?).
- **Distillation + Matryoshka** → truncatable dims AND no damage → whole frontier from one deployable model.
- Or add a metric-preservation regularizer to the contrastive loss.

**Caveat (generalization):** these recalls are evaluated on the SAME queries used in training → possible memorization. Random projection (0.770) cannot memorize, so the learned early layer's +0.06 over random is the quantity to confirm on HELD-OUT queries → the running **train80/eval20** split runs (`run_matryoshka.py --split-file /tmp/query_split_80_20.pkl`, logs `..._train80.log`) will report this.

## 5. Root cause (final)

Recall is governed by **how faithfully the embedding's WJ ordering matches the true 18,220-d WJ ordering** (Spearman → R@500, near-linear). The **hard-margin triplet objective optimizes only a local margin** (top-`max_pos` positive vs single hardest negative) — it never constrains the global WJ ranking, so the learned geometry is *less* metric-faithful than a random projection and **cannot use extra dimensions** to improve. It is an **objective–metric misalignment**, not a capacity/architecture/index problem.

InfoNCE works because its **WJ-softmax over a broad neighborhood** (max_pos=256) directly trains the embedding to rank the whole neighborhood in WJ → high Spearman → high recall, and it should benefit from more dims like random projection does.

Note: our InfoNCE is **WJ-native** — the contrastive softmax logits ARE Weighted-Jaccard similarities (cdist `(2-L1)/(2+L1)` identity on L1-simplex embeddings), not cosine. Same metric in loss, HNSW index, and rerank.

---

## 6. How to make it better (UPDATED with the layer-tap finding)

1. **★ Use a WIDE embedding, stop funneling to a narrow output.** Best result so far = the **triplet+recon 4096 layer (R@500 0.818)**. Either tap the 4096 layer of an existing model, or train `18220→4096→4096` (no 1024/512 bottleneck) so the loss is applied at the wide layer.
2. **★ Matryoshka for the whole frontier from ONE model.** Train a single wide (e.g., 4096-d) WJ embedding with the loss summed over nested prefixes {512,1024,2048,4096}; at inference TRUNCATE to any dim. Gives a good embedding at every dim → pick recall vs speed per query without retraining. (Kusupati et al. 2022, "Matryoshka Representation Learning".)
3. **Objective is secondary to width.** triplet+recon and InfoNCE both build good wide layers; triplet+recon's is currently best. Pick whichever; the lever is width.
4. **Trade-off = recall ↔ throughput.** Wider embedding (4096) → higher recall but slower/bigger HNSW. This IS the paper's frontier; quantify QPS at each width.
5. Coarser input grid is an orthogonal lever (real0.003=12k, real0.006=6k available; GT carries over) — untested.
6. Stage-2 exact rerank lifts top-50 to ~0.99 regardless of Stage-1; ceiling is 0.998 (raw WJ). Any d-dim embedding pays a JL-style compression tax; rerank closes the top-k.

**Probe scripts (all reusable, run `CUDA_VISIBLE_DEVICES=1`):** `/tmp/analysis_stage{1,2,3}.py`, `/tmp/probe_layers.py` (layer taps, full), `/tmp/analysis_10k.py` (10K), `/tmp/probe_infonce2048.py`.

---

## 7. Running jobs, files, repro

**RUNNING (as of 2026-06-16 ~late):**
- `InfoNCE@2048` — `python run_infonce_2048.py` (DataParallel 8 GPUs), PID was 1011589. Log: `/tmp/infonce_2048_full.log`. At ep09/18, acc=0.568 rising, healthy. On finish it appends base+rerank rows (method `InfoNCE(Recall-CG)-d2048`) to `NEW_RESULTS.csv`. **Compare its base R@500 to InfoNCE-512 (0.727) — if it rises, confirms dim helps the good objective.**

**Key files (all under `/raid/ruban/hpmlproj/term_project/SigSpatial/` unless noted):**
- `sota_experiment_common.py` — `load_dataset_normalized`, `eval_recall`, `nmslib_neighbors`, `rerank_wj_gpu`, `build_fn_mask`, `l1_simplex`, `shifted_l1_simplex`. `QUERY_START_FULL=187019`.
- `run_infonce_2048.py` — InfoNCE@2048 runner (cdist WJ loss, max_pos=256, DataParallel, CSV logging).
- `28_triplet_autoencoder_wj_2048.ipynb` — triplet+recon @2048 notebook (max_pos=30, cdist loss). Generator: `/tmp/make_nb_2048.py`.
- `28_triplet_autoencoder_wj_512.ipynb` — original healthy triplet (max_pos=30, 75 epochs, DataParallel).
- Analysis: `/tmp/analysis_stage1.py`, `/tmp/analysis_stage2.py`, `/tmp/analysis_stage3.py`, `/tmp/diag_collapse.py` (collapse repro: `--out_dim --maxpos --lam --sep`).
- Results log: `NEW_RESULTS.csv` (forward-looking), `RESULTS_LOG.csv` (older). Schema: `date,dataset,method,dim,stage,cand_k,R@10,R@50,R@100,R@500,QPS,source,notes`.

**Checkpoints (/tmp):**
- `best_triplet_autoencoder_wj_2048_full.pt` (triplet+recon 2048, max_pos=30, healthy)
- `best_sota_triplet_autoencoder_wj_512_full.pt` (triplet 512, healthy)
- `best_filter_recall_infonce_512_full.pt` (InfoNCE 512)
- `best_filter_recall_infonce_2048_full.pt` (being written by the running job)

**Repro analysis:** `CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /tmp/analysis_stage3.py`

**Eval settings (for comparable numbers):** nmslib HNSW WeightedJaccard, `threads=150`, `efSearch=200`, rerank batch=64, candidate K∈{1000,2000}. (Original 10K table used default efSearch — watch when comparing.)

---

## 8. Open items / next steps (REPRIORITIZED after the layer-tap finding)

- [ ] **Eval the triplet+recon 4096 layer through the FULL pipeline** (HNSW WJ + exact rerank) and log to NEW_RESULTS.csv — it's likely our best Stage-1 candidate generator (exact R@500 0.818). Measure its HNSW QPS (4096-d is slower).
- [ ] **Train `18220→4096→4096` (no funnel)** — confirm the wide embedding is the win when the loss is applied at the wide layer (vs tapping a funnel trained for a narrow output).
- [ ] **Build a Matryoshka WJ model** (loss on prefixes {512,1024,2048,4096} of one 4096-d embedding) → one model spanning the recall↔throughput frontier; the paper's dimension figure falls out of it.
- [ ] Quantify recall↔QPS at each width {512,1024,2048,4096} for the frontier plot (wider = higher recall, lower QPS).
- [ ] (Lower priority) InfoNCE@2048 currently running is BOTTLENECKED (`…→1024→2048`) AND mid-training; early read = tracks InfoNCE-512 (no gain), effRank 37. Confounded — consider killing in favor of the clean `→4096→4096` / Matryoshka runs.
- [ ] Reframe the paper around: **recall is set by WJ-rank-preservation, which the narrow output bottleneck destroys; a wide embedding (either objective) recovers it; Matryoshka exposes the frontier.** (Not "InfoNCE beats triplet" — that was an output-layer artifact.)
- [ ] Optional: per-query failure breakdown (extreme-area / near-duplicate queries).

## 9. One-paragraph narrative (for the paper / for a fresh agent)
On this polygon corpus, exact 18,220-d Weighted-Jaccard recovers the ground truth almost perfectly (R@500=0.998), so any learned Stage-1 embedding's recall gap is pure compression loss, and recall is governed by how faithfully the embedding preserves the global WJ rank-ordering (Spearman(emb-WJ, raw-WJ) predicts R@500 near-linearly). Increasing the *output* dimension of our encoders does not help (triplet R@500 = 0.654 at both 512-d and 2048-d; InfoNCE ≈ 0.73 at both; effective rank stays low), and the same stagnation appears on the 10K benchmark (output dim 256 ≈ 1024) — there it is merely *masked* because the small corpus saturates recall. The cause is the **funnel architecture's narrow output**: the contrastive/triplet loss, applied at the bottleneck, damages that layer's WJ ordering (output Spearman 0.62–0.82), whereas the model's **wide intermediate layer preserves it** (Spearman ≈ 0.94) — so tapping the 4096-d layer lifts R@500 from 0.654 to **0.818** (the best embedding we found, and from the triplet+recon model, beating InfoNCE and a random projection). That the *dimension itself* is useful is shown by a training-free random projection, whose recall rises monotonically with width (0.689→0.772 over 256→4096-d). The conclusion is therefore architectural, not an objective failure: compress less. Practically, train a wide WJ embedding (`18220→4096→4096`) or a Matryoshka embedding (WJ loss on nested prefixes {512…4096}) to expose the entire recall↔throughput frontier from a single model, with Stage-2 exact rerank closing the top-k to ~0.99.
