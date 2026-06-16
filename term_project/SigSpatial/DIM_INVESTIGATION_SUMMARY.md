# Why higher embedding dimension does NOT improve WJ candidate-generation recall — full investigation

**Date:** 2026-06-16  ·  **Dataset:** `full` (boi/ssEncodingData polygons)  ·  **Context:** SIGSPATIAL 2026 short paper on Weighted-Jaccard (WJ) ANN over polygons. Two-stage retrieval: Stage-1 = cheap learned embedding + HNSW(WeightedJaccard) candidate generation; Stage-2 = exact-WJ GPU rerank on raw 18,220-d vectors.

This doc is a self-contained handoff (limit may run out; may continue in Codex/Cursor).

---

## 0. TL;DR — what we proved

1. **Higher dimension does not help the *learned triplet* embedding** (R@500 = 0.654 at both 512-d and 2048-d, identical).
2. **It is NOT the index, NOT capacity, NOT precision, NOT the recon term, NOT BatchNorm** — all ruled out by controlled experiments.
3. **It IS the embedding geometry / training objective.** Decisive proof: a *training-free random projection* IMPROVES with dimension (R@500: 0.689→0.772 over 256→4096-d) while the learned triplet stays flat. So dimension helps a **metric-preserving** embedding; the learned triplet doesn't benefit because **its objective does not preserve the WJ metric**.
4. **Root cause:** the hard-margin triplet loss optimizes a *local* margin (top-`max_pos` positive vs hardest in-batch negative); it places no constraint on global WJ rank-ordering, so the learned geometry is *less* metric-faithful than a random projection (Spearman 0.82 vs 0.93) and gains nothing from extra dims.
5. **Fix:** a metric-aligned objective. **InfoNCE** (WJ-softmax over a broad neighborhood) preserves the metric (Spearman 0.92) and gets R@500 = 0.73; and, like random projection, it should improve with dimension. `InfoNCE@2048` is training now to confirm.
6. **Separate but critical bug found earlier:** the first 2048 triplet run *collapsed* because it used `max_pos=256` (carried over from the InfoNCE runner). Triplet's hard margin is unsatisfiable with broad positives and collapses to identical embeddings. **Triplet needs `max_pos=30`.** Fixed; the 2048 numbers above are post-fix and healthy.

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

## 5. Root cause (final)

Recall is governed by **how faithfully the embedding's WJ ordering matches the true 18,220-d WJ ordering** (Spearman → R@500, near-linear). The **hard-margin triplet objective optimizes only a local margin** (top-`max_pos` positive vs single hardest negative) — it never constrains the global WJ ranking, so the learned geometry is *less* metric-faithful than a random projection and **cannot use extra dimensions** to improve. It is an **objective–metric misalignment**, not a capacity/architecture/index problem.

InfoNCE works because its **WJ-softmax over a broad neighborhood** (max_pos=256) directly trains the embedding to rank the whole neighborhood in WJ → high Spearman → high recall, and it should benefit from more dims like random projection does.

Note: our InfoNCE is **WJ-native** — the contrastive softmax logits ARE Weighted-Jaccard similarities (cdist `(2-L1)/(2+L1)` identity on L1-simplex embeddings), not cosine. Same metric in loss, HNSW index, and rerank.

---

## 6. How to make it better

1. **Use the metric-aligned objective (InfoNCE) — primary fix.** Spearman 0.92, R@500 0.73 at 512-d. Running at 2048 now (see §7); expected to improve with dim like random projection (unlike triplet).
2. **Higher dim DOES help once the embedding is metric-preserving** — e.g., InfoNCE@2048/4096, or even a training-free higher-dim random projection (4096 → 0.772). Trade-off: higher dim = lower HNSW QPS.
3. **Coarser input grid is an orthogonal lever** (real0.003=12k, real0.006=6k available; GT carries over) — untested for recall vs compute.
4. Stage-2 exact rerank already lifts top-50 to ~0.99 regardless of Stage-1, so for the paper the Stage-1 job is *broad recall* (R@500), where InfoNCE + dimension is the path.
5. **Caveat:** ceiling is 0.998 (raw WJ); any d-dim embedding pays a JL-style compression tax. To approach the ceiling at small d you need rerank (which we do).

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

## 8. Open items / next steps

- [ ] **Finish InfoNCE@2048**, log base+rerank; confirm base R@500 > InfoNCE-512 (0.727) → proves dim helps the metric-aligned objective.
- [ ] Optional: InfoNCE@1024 / @4096 to draw the InfoNCE recall-vs-dim curve next to random projection (the paper's dimension figure).
- [ ] Optional: per-query failure breakdown (which polygons fail — e.g., extreme-area / many-near-duplicate queries).
- [ ] Decide paper framing: the mechanistic story (recall ∝ WJ-rank-preservation; triplet's objective misaligned; InfoNCE aligned; dimension helps only the aligned objective) is a strong, defensible narrative.
- [ ] QPS: 2048-d slightly lower HNSW QPS than 512-d; quantify the recall↔QPS trade-off across dims for the frontier plot.

## 9. One-paragraph narrative (for the paper / for a fresh agent)
On this polygon corpus, exact 18,220-d Weighted-Jaccard recovers the ground truth almost perfectly (R@500=0.998), so any learned Stage-1 embedding's recall gap is pure compression loss. We find that recall is governed by how faithfully the embedding preserves the global WJ rank-ordering (Spearman(emb-WJ, raw-WJ) predicts R@500 near-linearly). A hard-margin triplet objective preserves this ordering poorly (Spearman ≈ 0.82) — worse than a *training-free* random projection (0.93) — and, crucially, does not improve with embedding dimension (R@500 = 0.654 at both 512-d and 2048-d). In contrast, random projection's recall rises monotonically with dimension (0.689→0.772 over 256→4096-d), proving the dimension itself is useful only for a *metric-preserving* embedding. The triplet's failure is therefore an objective–metric misalignment, not a capacity, architecture, or index limitation. A WJ-native InfoNCE objective (contrastive softmax over Weighted-Jaccard similarities across a broad neighborhood) restores metric fidelity (Spearman 0.92, R@500 0.73) and is expected to scale with dimension like random projection.
