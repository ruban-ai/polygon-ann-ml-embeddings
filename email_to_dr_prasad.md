Subject: WJ candidate generator — why triplet/InfoNCE underperformed, and a fix (WJ-distillation)

Dear Dr. Prasad,

Quick update on the learned Stage-1 candidate generator for Weighted-Jaccard (WJ) polygon retrieval. We found *why* the triplet and InfoNCE embeddings were underperforming, and a principled fix that already works. Two figures attached.

— — —

## 1. The problem
Our learned Stage-1 embeddings (triplet+reconstruction, and InfoNCE) underperformed:
- On the full set they were **even slightly below a training-free random projection** at Stage-1.
- **Adding embedding dimensions did not help** (512 → 2048 gave essentially no gain).

This was puzzling — more capacity and a learned model should beat a random projection.

## 2. Root cause: the training objective does not match the metric we grade on
We grade on **recall**, which needs the embedding to **preserve the true WJ ranking** of neighbors. But triplet/InfoNCE optimize a **separation proxy** — "push each positive above the negatives." Those are *not* the same goal: you can separate positives from negatives while badly **distorting** the global WJ geometry. We verified this with controlled experiments:

- **Recall is governed by metric preservation.** How faithfully an embedding's WJ ordering matches the true 18,220-d WJ ordering (a rank-correlation) predicts recall almost linearly.
- **The loss damages the exact layer it is applied to.** Reading the network's *deployed output* gives the **worst** recall in the whole model — *below a random projection* — while an *untouched intermediate layer* gives the best (Figure 1). This mirrors a known effect in self-supervised learning (e.g., SimCLR), where the contrastive "projection head" is discarded and the layer *before* it is used as the representation.
- **Capacity was never the limiter** — so more dimensions can't help; the objective is the problem.

**[Figure 1 — fig1_layer_tapping.png]:** within one trained triplet model, recall *increases* the further you read from the loss; the deployed 512-d output (0.65) sits below a random projection (0.77), while the untouched first layer (0.82) is best.

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

It also **generalizes** — on held-out (unseen) queries the recall holds (no memorization), unlike the contrastive models, whose top-50 recall dropped sharply on unseen queries.

## 5. Full-scale on the 47K (in progress)
Per your suggestion, we are running the same WJ-distillation on the **47K pool, split 80% corpus / 20% held-out queries** (area-stratified — the area distributions of the two partitions match to ~0.6 percentage points). Eval is the held-out 20% queries searched against the 80% corpus, using the official GT.

**[RESULTS PLACEHOLDER — per-dimension recall + QPS]**

| dim | base R@50 | base R@500 | HNSW QPS | reranked R@50 | reranked R@500 |
|---|---|---|---|---|---|
| 256 | … | … | … | … | … |
| 512 | … | … | … | … | … |
| 1024 | … | … | … | … | … |
| 2048 | … | … | … | … | … |
| 4096 | … | … | … | … | … |

## 6. Next steps
Once the 47K confirms, we will scale to the full 187K corpus and the larger Overture sets, and finalize the recall–throughput frontier (dimension as the tunable knob) for the paper.

Happy to walk through any of this in person.

Best regards,
Ruban
