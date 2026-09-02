# Paper Review & Improvements Log — BSD 2026 full paper

Started 2026-09-01, after the loss-function exploration (see `LOSS_EXPLORATION_LOG.md`)
concluded and the paper was ported from the ACM template (`full_paper/sigconf.tex`) to
IEEE's official conference template (`IEEE/conference_101719.tex`, mirrors what should
be pasted into the BSD Overleaf project). This log tracks the reviewer-style critique
pass and the experiments it triggered.

---

## 1. Reviewer critique — summary of findings

Full critique given in-conversation; key points, ranked by severity:

1. **QPS regression, unexplained** — listwise (best-recall method) is measurably
   *slower* than MSE distillation at matched dimensions in every table (e.g. 50K/256-d
   base: MSE 6,772 QPS vs listwise 4,921 QPS). No explanation currently in the paper.
   **Status: under investigation, see §2.**
2. **Contribution framing** — the three listed contributions (encoder, loss, Matryoshka)
   read as equally weighted; the real claim is the loss-function result, with the
   encoder/Matryoshka choices as supporting design, not co-equal contributions.
   **Status: pending rewrite, see §3 for the sharper framing the author wants.**
3. **10K scale** — Buddhi flagged 10K as not a credible scale to lean on. Verified (not
   assumed) that PCA/NMF/ICWS(brute)/Matryoshka-MLP baselines exist *only* at 10K, never
   rerun at 50K/187K. Decision: keep 10K (it's the only place the full baseline suite +
   ablations exist) but reframe explicitly in the text as a baseline/ablation testbed,
   not a scale-appropriate headline result.
4. Sports-field mention in the Intro has no results and invites an easy "why didn't you
   just run it" review comment — recommend cutting or explaining the format-incompatibility
   reason it was deferred.
5. PQ (product quantization) dismissal is secondhand (cites ShapeToVec's own claim, not
   verified independently).
6. Table redundancy: `tab:full50k` and `tab:dim` both report the 256-d 50K row.
7. No repeated-seed variance anywhere; some "flat across dimension" deltas are small
   enough that noise can't be ruled out without repeats.
8. The 0.999 R@500 quadtree-WJ-vs-geometric-Jaccard approximation claim is stated but
   not demonstrated in this paper (likely inherited from ShapeToVec's own validation).

## 2. QPS investigation (in progress)

**Hypothesis:** the MSE and listwise checkpoints' QPS numbers were measured on a
shared, non-dedicated multi-GPU node at different points across this long session, so
the "regression" could be measurement noise rather than a real property of the method.
Cannot rule this in or out without a clean, back-to-back, same-GPU re-measurement.

**Problem hit immediately:** the MSE 50K checkpoint (`best_50k_distill.pt`) no longer
exists — `run_50k_ddp.py` originally saved it to `/tmp/`, which is not durable (this is
the same class of failure flagged in `feedback_nohup_master_log` memory). No re-eval-only
comparison was possible without retraining.

**Action taken:**
- Patched `run_50k_ddp.py` to save checkpoints to the project directory instead of `/tmp`.
- Retrained MSE at 50K from scratch (`run_50k_ddp.py --method distill`, GPUs 1-7).
- Immediately after, re-evaluated the *existing* listwise checkpoint (no retraining,
  `eval_50k_listwise_recheck.py`) on an idle GPU, minutes after the MSE eval, for a
  genuinely back-to-back comparison.

**Result: DONE — the effect is real, but the "regression" framing was incomplete.**

| Dim | MSE QPS (fresh) | Listwise QPS (fresh) | Listwise vs MSE |
|---|---|---|---|
| 256  | 6,799 | 5,599 | **−18%** |
| 512  | 3,675 | 2,766 | **−25%** |
| 1024 | 3,403 | 1,939 | **−43%** |
| 2048 | 1,685 | 1,393 | **−17%** |
| 4096 | 1,075 | 1,286 | **+20%** |

Two findings:
1. **The MSE-vs-listwise QPS gap is real, not measurement noise.** The fresh MSE numbers
   landed close to the original logged ones (256-d: 6,799 vs 6,772, within 0.4%), and the
   listwise-slower-at-small-dims pattern reproduced independently minutes later on a
   different, idle GPU. Confirmed, not an artifact of a shared/contended node.
2. **But "listwise is slower" is the wrong generalization — it's dimension-dependent, and
   this was already visible in the original numbers, just not stated.** MSE's QPS falls
   sharply with dimension (6,772→967 in the original log, 7x); listwise's falls much more
   gently (4,921→1,493, 3.3x) — they cross over between 1024-d and 2048-d, and at 4096-d
   listwise is *faster*. This is the same fact the paper's dim-tradeoff paragraph already
   states in a different form ("listwise's recall is essentially flat across dimension
   while QPS falls only ~3.3x") — it was never connected to the MSE comparison.

**Interpretation:** MSE's HNSW search cost scales worse with nominal dimension than
listwise's does. A plausible reading: listwise's graded-softmax objective only supervises
*relative* order, not absolute magnitude, so it may pack useful signal into a lower
*effective* dimensionality even at large nominal widths, making search cost grow more
slowly with `d`; MSE's absolute-value regression has more pressure to use the full nominal
width, so its search cost scales more steeply. Not independently verified beyond the
reproducibility check (e.g. no direct effective-rank/intrinsic-dimensionality measurement
was taken) — stated as a reasoned hypothesis, not a proven mechanism.

**Action for the paper:** replace the "unexplained regression" framing with the correct,
dimension-dependent one — state the crossover explicitly (listwise faster at 2048/4096-d,
MSE faster at 256/512/1024-d) rather than letting the tables show an apparently
inconsistent, unexplained pattern.

## 3. Contribution reframing — author's explicit direction

Author's own framing (verbatim intent, 2026-09-01): **the core contribution is
compression while retaining recall relative to ShapeToVec**, not "our loss function
beats other loss functions" (that's supporting evidence for *how* we retain recall,
not the headline itself).

**Gap found:** the paper currently cites ShapeToVec's own externally-reported 97% R@50
(on their own up-to-1.7M-polygon setup) as Intro motivation, but never directly compares
it to our own numbers on the *same* benchmark/protocol — an apples-to-apples comparison
doesn't exist yet within the paper.

**Better option identified, not yet run:** rather than lean on ShapeToVec's external
number, reproduce ShapeToVec's own method (full-D-dimensional vectors, no compression,
straight into HNSW under the WeightedJaccard space) on our exact 50K/187K benchmark and
protocol. This gives a same-benchmark "uncompressed ceiling" to state the real headline
claim against: *"a 72x-compressed embedding retains N% of the uncompressed accuracy."*
Checked: **no such baseline exists in the results logs yet** (`grep` for full-dimensional
HNSW runs returned nothing) — this would be a new experiment, not a repositioning of
existing numbers.

**Status: DONE (2026-09-01) — result is stronger than expected, changes the framing.**
`run_uncompressed_baseline_50k.py` completed in 1.9 min total (HNSW build+query: 1.7 min).

**Result — full D=18,382-dim vectors, no compression, direct HNSW under WeightedJaccard,
same 50K corpus/queries/GT as every other result in the paper:**

| Method | Dim | R@10 | R@50 | R@500 | QPS |
|---|---|---|---|---|---|
| Uncompressed (ShapeToVec's own method, reproduced) | 18,382 | 0.6486 | 0.6990 | 0.8166 | 1,102 |
| **Ours (listwise distill)** | **256** | **0.871** | **0.921** | **0.969** | **4,921** |

**This is not "retains most of the accuracy while compressing" — the compressed
embedding beats the uncompressed one, by +22.2pp R@50 and +15.2pp R@500, while also
running ~4.5x faster.** Compression is not a tradeoff here, it's a net win on both axes.

**Mechanism (plausible, not yet independently verified):** this is the curse of
dimensionality hitting *approximate* graph search (HNSW) specifically, not the WJ metric
itself — brute-force/exact WJ on the raw vectors is still near-ceiling (the paper's
existing 0.999 R@500 quadtree-vs-geometric-Jaccard claim), but HNSW's approximate greedy
traversal degrades in very high-dimensional spaces (distance concentration reduces the
signal available for graph navigation). A well-shaped, lower-dimensional learned
embedding is not just cheaper to search, it is an *easier* search problem for HNSW.
Since candidate_k=500=top_k here, note R@500 is capped by Stage-1 alone — Stage-2 rerank
cannot improve it further for the uncompressed pipeline, since reranking only reorders
within a fixed candidate pool, it cannot add missed items back in.

**Action needed:** this is a much stronger and more citable core-contribution claim than
originally discussed ("beats an untrained random projection") — recommend leading the
abstract/intro with this same-benchmark uncompressed comparison instead of (or in
addition to) the RP comparison. Pending: rewrite abstract/intro/conclusion around this;
consider running the same uncompressed baseline at 187K for confirmation at the largest
scale too (same script pattern, CPU-only, cheap).

## 4. ICWS-brute at 50K (in progress → nearly done)

Fills the "no near-exact accuracy ceiling at a credible scale" gap (critique point 3
material). Original notebook (`26_icws_weighted_minhash_512.ipynb`) computed this at
10K and once, expensively, at full 187K scale (signing alone: **542.2 min / ~9h**, pure
NumPy with a Python loop over 512 samples; ranking: ~16,122s for 46,754×187,019).

**Fix:** rewrote signing as a GPU-vectorized op (`run_icws_brute_50k.py`), chunking over
samples instead of looping one at a time. **Verified bit-identical to the original
NumPy implementation** on a synthetic test with shared random parameters before trusting
it on real data (`idx match: True`, `t match: True`).

**Result at 50K: signing took 0.4 minutes** (vs. ~116 min extrapolated for the original
CPU implementation at this scale) — a ~290x speedup. Ranking phase in progress
(~10,000 queries × 40,000 corpus, estimated ~12-15 min based on the full-scale timing).
Will log final R@10/R@50/R@500/QPS here and to `NEW_RESULTS.csv`
(tag `50k-ICWSbrute-d512`) once complete.

## 5. Data reliability fix (2026-09-01, per explicit user instruction)

User instruction: stop using `/tmp` for anything that needs to persist, log everything
to files for later reference.

Migrated the following out of `/tmp` (which is not durable — this already caused the
lost MSE-50K-checkpoint problem in §2) into the project directory, with `/tmp` symlinks
left in place pointing to the new locations so every existing script's hardcoded
`/tmp/...` path keeps working without edits:

| File | Size | New location |
|---|---|---|
| `qt_10k.npy` | 706M | `SigSpatial/qt_10k.npy` |
| `qt_50k.npy` | 3.5G | `SigSpatial/qt_50k.npy` |
| `qtree_vectors_full.npy` | 16G | `SigSpatial/qtree_vectors_full.npy` |
| `corpus_knn_50k.npy` | 9.2M | `SigSpatial/corpus_knn_50k.npy` |
| `gt_lookup_10k.pkl` | 1.2M | `SigSpatial/gt_lookup_10k.pkl` |
| `gt_50k.pkl` | 17M | `SigSpatial/gt_50k.pkl` |
| `gt_lookup_full.pkl` | 888M | `SigSpatial/gt_lookup_full.pkl` |

`/raid` has 1.3TB free (vs `/`'s tighter headroom) so this is a safe home. New
checkpoints/outputs from this point on should be written directly to the project
directory, never `/tmp`, per standing project convention.

---

*Update this file as each in-progress item lands, before moving to the next one —
same convention as `LOSS_EXPLORATION_LOG.md`.*
