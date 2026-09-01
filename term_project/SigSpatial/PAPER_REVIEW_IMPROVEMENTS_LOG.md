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
- Patched `run_50k_ddp.py` to save checkpoints to the project directory instead of `/tmp`
  (line ~184/187).
- Retraining MSE at 50K from scratch (`torchrun --standalone --nproc_per_node=7
  run_50k_ddp.py --method distill`, GPUs 1-7, log `logs_50k_mse_retrain.log`).
- Plan once training completes: evaluate the fresh MSE checkpoint and the existing
  listwise checkpoint (`best_50k_listwise.pt`) back-to-back on the same idle GPU, to get
  a genuinely controlled QPS comparison. Result to be written up here and used to either
  (a) confirm it's noise and re-measure the paper's QPS numbers cleanly, or (b) find a
  real mechanistic explanation (e.g. embedding-value distribution differences affecting
  HNSW graph traversal cost) and add one sentence to the paper.

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

**Status: not started.** Next step once the QPS/ICWS work below is clear.

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
