Subject: WJ candidate generator — why triplet/InfoNCE underperformed, and a fix (WJ-distillation)

Dear Dr. Prasad,

Quick update on the learned Stage-1 candidate generator for Weighted-Jaccard (WJ) polygon retrieval. We found *why* the triplet and InfoNCE embeddings were underperforming, and a principled fix that already works — now validated at scale. Three figures attached.

— — —

## 1. The problem
Our learned Stage-1 embeddings (triplet+reconstruction, and InfoNCE) underperformed:
- On the full set they were **even slightly below a training-free random projection** at Stage-1.
- **Adding embedding dimensions did not help** (512 → 2048 gave essentially no gain).

This was puzzling — more capacity and a learned model should beat a random projection. We also ran the held-out protocol directly on the 47K — **train on 80% of the queries, evaluate the held-out 20%** — and the contrastive methods *still* did not improve with dimension:

| 47K held-out — base R@500 | 1024-d | 2048-d | 4096-d |
|---|---|---|---|
| triplet + recon | 0.651 | 0.666 | 0.673 |
| InfoNCE | 0.724 | 0.727 | 0.729 |

Recall is essentially flat from 1024→4096 — more dimensions buy almost nothing.

## 2. Root cause: the training objective does not match the metric we grade on
We grade on **recall**, which needs the embedding to **preserve the true WJ ranking** of neighbors. But triplet/InfoNCE optimize a **separation proxy** — "push each positive above the negatives." Those are *not* the same goal: you can separate positives from negatives while badly **distorting** the global WJ geometry. We verified this with controlled experiments:

- **Recall is governed by metric preservation.** How faithfully an embedding's WJ ordering matches the true 18,220-d WJ ordering (a rank-correlation) predicts recall almost linearly.
- **The loss damages the exact layer it is applied to.** Reading the network's *deployed output* gives the **worst** recall in the whole model — *below a random projection* — while an *untouched intermediate layer* gives the best (Figure 1). This mirrors a known effect in self-supervised learning (e.g., SimCLR), where the contrastive "projection head" is discarded and the layer *before* it is used as the representation.
- **Capacity was never the limiter** — so more dimensions can't help; the objective is the problem.

| layer (one trained triplet model) | Stage-1 recall R@500 |
|---|---|
| 1st layer (4096-d) | **0.816** |
| 2nd layer (1024-d) | 0.775 |
| **output (512-d — loss applied here)** | **0.652** |
| random projection (no training) | 0.770 |

**[Figure 1 — fig1_layer_tapping.png]** plots the same numbers: recall *increases* the further you read from the loss; the deployed output sits *below* a random projection, while the untouched first layer is best.

## 3. The fix: WJ-native distillation (align the objective with the metric)
Instead of a separation proxy, we train the embedding so that **its Weighted-Jaccard directly matches the true Weighted-Jaccard** — a distillation/regression objective (loss = MSE between the embedding's WJ and the raw 18,220-d WJ). Now **the objective *is* the metric**, so there is nothing to misalign — the output stops being distorted and becomes **directly deployable** (no discarded head, no layer-tapping tricks).

The more aligned the objective, the less the output is damaged (Figure 2): triplet (pure separation) ruins the output; InfoNCE (softmax, partially aligned) is better; WJ-distillation (the metric itself) leaves it essentially undamaged.

We also make it **Matryoshka**: one model produces a single embedding from which we can **truncate to any dimension {256, 512, 1024, 2048, 4096}** at query time — giving the whole recall-vs-throughput frontier from a single training, all WJ-native (no cosine anywhere).

**[Figure 2 — fig2_objective_alignment.png]:** how well the deployed output preserves the true WJ ranking, for the three objectives — the "alignment gradient."

## 4. Results so far (validated on the 10K benchmark)
The WJ-distillation output is now the **best** layer (the opposite of triplet/InfoNCE) and beats a random projection — the first learned WJ embedding to do so cleanly:

| stage | R@50 | R@500 |
|---|---|---|
| Stage-1 (HNSW, no rerank) | **0.905** | 0.992 |
| + exact-WJ rerank | **0.999** | 0.995 |

It also **generalizes** — on a held-out split (queries never seen in training) it scores R@50 = 0.92 / R@500 = 0.995, matching the train-set numbers (i.e. no memorization). The contrastive models, by contrast, lost top-50 recall sharply on unseen queries.

## 5. Full-scale on the 47K (held-out)
Per your suggestion, we ran the same WJ-distillation on the **47K pool, split 80% corpus (37,403) / 20% held-out queries (9,351)** — area-stratified (the area distributions of the two partitions match to ~0.6 percentage points). The model trains only on the corpus; the held-out queries are searched against it. One Matryoshka model, truncated to each dimension at query time.

| dim | base R@50 | base R@500 | HNSW QPS | reranked R@50 | reranked R@500 |
|---|---|---|---|---|---|
| **256** | 0.742 | 0.846 | **6,072** | 0.993 | 0.921 |
| 512 | 0.750 | 0.851 | 3,147 | 0.993 | 0.921 |
| 1024 | 0.755 | 0.854 | 1,976 | 0.994 | 0.922 |
| 2048 | 0.757 | 0.855 | 1,296 | 0.994 | 0.921 |
| 4096 | 0.756 | 0.854 | 1,154 | 0.994 | 0.919 |

*(rerank at K=1000, exact WJ; held-out 20% queries vs the official GT.)*

**Takeaways:** Stage-1 recall is strong on **unseen** queries (R@500 ≈ 0.85); the exact-WJ rerank lifts top-50 to **≈ 0.994**; and **256-d is a sweet spot** — essentially the same recall as 4096-d at **~5× the throughput**. The entire curve comes from a *single* training.

**[Figure 3 — fig3_47k_frontier.png]:** the recall–throughput frontier — recall stays ~0.85 while throughput spans 1,150 → 6,070 QPS across the truncation dimensions.

## 6. Next steps
With the 47K confirmed, we next scale to the full 187K corpus and the larger Overture sets, and finalize the recall–throughput frontier (dimension as the tunable knob) for the paper.

Happy to walk through any of this in person.

Best regards,
Ruban
