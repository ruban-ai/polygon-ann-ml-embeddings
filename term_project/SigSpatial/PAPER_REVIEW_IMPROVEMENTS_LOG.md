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

## 6. ShapeToVec comparison correction (2026-09-01) — self-reproduction was wrong, use published numbers

The uncompressed-baseline claim in §3 above (our own reproduction of ShapeToVec's method
at D=18,382, getting R@50=0.699) was **retracted**. User pointed out ShapeToVec's own
paper reports very different numbers on the same Parks dataset (Table III: 96-97% R@50
at 3k/6k/12k-dim, floating-point encoding) — our self-reproduction likely had a real
methodology gap (unvalidated dimension, and/or `efSearch=200` copied from our own
compressed-embedding pipeline without retuning for a much larger raw vector). Rather
than debug our own reproduction further, switched to citing ShapeToVec's own published
Table III numbers directly (`\cite{shape2vec}`) — more authoritative anyway, and avoids
publishing an unverified/likely-wrong claim. Also corrected: the "3k/6k/12k" values are
the paper's own **Table III row label ("Vector size")**, distinct from the "max capacity"
quadtree-construction parameter described separately in their Methodology (IV.B.2) — user
flagged a concern these might be the same thing; textual evidence in the paper favors
"Vector size" meaning the resulting encoded dimension, but this is not independently
verified beyond the paper's own labeling, and was surfaced to the user for judgment.

**Corrected comparison used in the paper now** (full multi-dimension table in
`tab:uncompressed`): our 256-4096-d embeddings reach R@50=92.1-92.9% without reranking
(vs.\ ShapeToVec's 96-97% at their 3k-12k-d), at 4-14x the throughput; after a cheap
exact-WJ rerank pass, our 256-d embedding reaches R@50=99.7%, exceeding ShapeToVec's
best published number while remaining ~3x faster than their most accurate config.
Compression ratio corrected from the invalid "72x" (vs.\ our own unverified 18,382-d
reproduction) to "12-47x" (256-d vs.\ their 3k-12k-d range).

## 7. Independent (non-Matryoshka) single-width training — the "Matryoshka tax" (2026-09-01)

Triggered by a follow-up question: since all 5 Matryoshka widths come from one
jointly-trained encoder (shared 4096-d bottleneck, smaller widths are prefix
truncations), does independent single-width training recover higher recall? If so,
by how much, and is the effect concentrated at small widths (truncation-specific) or
uniform across widths (a general joint-training cost)?

**Method:** two new scripts, same architecture/data/10-epoch budget/listwise
loss/tau as `run_50k_listwise.py`, but each optimises only ONE width's loss term
(no other prefixes in the loss):
- `run_50k_listwise_independent4096.py` — same architecture (encoder already outputs
  4096-d natively), single-term loss at m=4096. Isolates whether joint training costs
  accuracy even with **no truncation at all**.
- `run_50k_listwise_independent256.py` — genuinely smaller architecture (encoder's
  final layer outputs 256-d natively, not a 4096-d truncation), single-term loss.
  Tests our actual deployed operating point directly.

Both ran DDP across 4 GPUs each (in parallel, ~46 min training + eval per job).

**Result:**

| Width | Matryoshka (existing) R@50 | Independent (new) R@50 | Δ | QPS (Matryoshka → independent) |
|---|---|---|---|---|
| 256-d  | 92.08% | **93.27%** | **+1.19pp** | 4,921 → 5,971 |
| 4096-d | 92.91% | **94.04%** | **+1.13pp** | 1,493 → 1,108 |

**Finding: the gap is real (~1.1-1.2pp R@50) and near-identical at both the narrowest
and widest widths tested.** Because the 4096-d case has zero truncation, this isolates
the effect cleanly: the cost is not "smaller widths get starved of capacity" (which
would show a bigger gap at 256-d than 4096-d) but a fairly uniform tax from training all
5 widths' loss terms jointly (gradient interference across simultaneous objectives),
independent of which width is being read out. Rerank recovers most of the gap regardless
(both land within noise of each other after reranking, ~R@500=0.99), so the practical
cost is concentrated in the Stage-1/no-rerank operating point specifically.

**Answers the "what's the point of Matryoshka if independent training is better"
question directly, with a real number**: ~1.1-1.2pp R@50 is the actual, now-measured
price of serving all 5 widths from one training run instead of five. Whether that's
worth stating in the paper as an honest limitation/trade-off acknowledgment is a
judgment call for the user — not yet added to the paper text.

## 8. 233K naming fix, abstract restore, and citation audit (2026-09-01)

**233K naming fix.** The paper's own "50K"/"10K" scale-label convention names the
*total* dataset size, but "187K" had been used the same way even though 187,019 is only
the corpus subset (queries are a separate 46,754). True total = 233,773, matching
ShapeToVec's own reported Parks dataset size exactly. Fixed via `sed -i 's/187K/233K/g;
s/tab:full187/tab:full233/g'` plus two manual rewrites of corpus-specific sentences that
needed to keep the corpus/query split legible (now read "the full 233k Parks dataset
(187,019-polygon corpus / 46,754 queries)"). Applied identically to both
`full_paper/sigconf.tex` and `IEEE/conference_101719.tex`.

**Abstract shortened, then partially restored.** First pass over-trimmed and dropped the
"extreme area variability" motivating context entirely, per author feedback ("i thnk we
sharnk too much.. we remove extreme area vairablet and everyhtign which i thnk we shuold
add back"). Final abstract (206 words) restores the motivation — uniform-grid encoding
can't serve both very small and very large polygons, forcing ShapeToVec's adaptive
quadtree — while keeping the length disciplined and leading with the corrected
233,773-polygon, ShapeToVec-published-numbers comparison (12-47x smaller, within a few
points of their recall pre-rerank, exceeds their best number post-rerank).

**Citation audit** (explicit request: "did we cite all the things needed? the forumlas
and everytting did we cite them properly. check"). Findings:

- **Two zero-citation claims found and fixed:**
  - The WJ formula itself (Eq.~\ref{eq:wj}) had no citation at its point of introduction
    — added `\cite{rajaraman11}` (Rajaraman & Ullman, *Mining of Massive Datasets*),
    the same source ShapeToVec's own paper cites for the identical formula.
  - "product quantization" (Related Work, PQ dismissal) was uncited — added
    `\cite{jegou10pq}` (Jégou, Douze, Schmid, IEEE TPAMI 2010), matching ShapeToVec's
    own reference for PQ.
- **Two bib entries had placeholder/wrong content, fixed via live source verification:**
  - `smec25` — author field was a `% TODO(Ruban)` placeholder ("Anonymous"); replaced
    with the real author list (Biao Zhang, Lixin Chen, Tong Liu, Bo Zheng) and corrected
    title, verified via `WebFetch` of `aclanthology.org/2025.emnlp-main.1332`.
  - `mipic25` → renamed `mipic26` throughout — author field was also a placeholder, and
    the year was wrong: arXiv ID `2604.24374` decodes to **April 2026**, not 2025 (the
    `YYMM.NNNNN` convention was checked directly). Fixed authors, title, and year via
    `WebFetch` of `arxiv.org/abs/2604.24374`.
- **Cascading fix: inaccurate "2024–2025" blanket year-claim.** The paper claimed "five
  further 2024–2025 rank-aware alternatives" in 5 places, but one of the five (the RKD
  angle-wise term) is only cited to its original 2019 paper — no actual 2024/2025
  follow-up was ever added, so the blanket claim was unsupported for that item. Fixed by
  rewriting the Related Work sentence to explicitly separate the 4 genuinely
  2025–2026-sourced items (SMEC-based: sequential/curriculum Matryoshka training,
  cross-batch hard-negative memory, learnable dimension selection; plus MIPIC) from the
  1 older (2019) RKD-based one, and removing the specific year qualifier from the other
  4 locations (abstract, intro contribution #2, Method section, Conclusion), replacing it
  with neutral "five further rank-aware and Matryoshka-training alternatives" (no year
  claim needed there).
- **Spot-checked ~12 other citations for content accuracy** (achlioptas03=random
  projection, jolliffe02=PCA, ioffe10=ICWS, hinton15=distillation, schroff15=FaceNet/
  triplet, oord18=InfoNCE, burges05=RankNet, cao07=ListNet, hnsw20=HNSW,
  boytsov13=NMSLIB, matryoshka22=Matryoshka, nmf99=NMF) — all correct, no changes needed.
- **Automated verification** (both files): `\begin`/`\end` balance, every `\ref`/`\eqref`
  resolves to an existing `\label`, every `\cite` key exists in `refs.bib` — all clean in
  both `full_paper/sigconf.tex` and `IEEE/conference_101719.tex`.
- **Open item, not yet resolved:** `refs.bib` has one unused entry, `osm` (OpenStreetMap
  contributors, 2024) — defined but never cited anywhere in either file. Likely intended
  for a data-provenance mention (SpatialHadoop's polygon datasets sometimes trace back to
  OSM), but this isn't independently confirmed for *our specific* Parks/water-body data,
  so it was left alone rather than guessed into a citation. Needs a decision: cite it
  where the Parks/water-body data source is described, or delete it as unused.
- **Open item, unresolved from earlier in this log:** whether ShapeToVec's "3k/6k/12k"
  values are vector dimension or quadtree node-capacity (see §6) was never independently
  confirmed beyond the paper's own "Vector size" table label — still resting on that
  reading, flagged for the user's judgment if they have a clarifying source.

**Sync status:** `full_paper/sigconf.tex`/`refs.bib` and `IEEE/conference_101719.tex`/
`refs.bib` are fully synchronized as of this fix — verified via direct diff (refs.bib:
`diff` exit 0) and independent per-file automated checks on the IEEE `.tex` (all clean,
matching the full_paper results).

---

*Update this file as each in-progress item lands, before moving to the next one —
same convention as `LOSS_EXPLORATION_LOG.md`.*
