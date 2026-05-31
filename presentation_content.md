# Learning Compact Embeddings for Polygon Similarity Search
### MLP Compressor · Neural MinHash · GPU Two-Stage Retrieval
**OpenStreetMap Park Polygons · 233,773 polygons · 8× A100-80GB**

---

## Slide 1 — Title

**Title:** Learning Compact Embeddings for Polygon Similarity Search

**Subtitle:** MLP Compressor · Neural MinHash · GPU Two-Stage Retrieval

**Supporting line:** OpenStreetMap Park Polygons · 233,773 polygons · 8× A100-80GB

---

## Slide 2 — The Problem

**Title:** Finding Similar Polygons at Scale is Expensive

- 233,773 park polygons from OpenStreetMap
- Finding similar polygons requires comparing every pair using WeightedJaccard similarity
- At 18,220 dimensions per polygon: **25 GB RAM, 560 seconds to build index**

| Metric | Baseline |
|--------|----------|
| Vector memory | 12,999 MB |
| Index memory | 11,519 MB |
| Total memory | ~25 GB |
| Build time | 560s |
| QPS | 611 |

> **Speaker note:** "The baseline works — but it's too expensive for standard deployment. Most servers don't have 25 GB free just for one index."

---

## Slide 3 — What is a Polygon? What is WeightedJaccard?

**Title:** How Polygons Are Represented

**Quadtree encoding:**
- Each polygon is rasterized onto a geographic grid
- Each cell records fractional area overlap with the polygon
- Result: sparse float vector, 18,220 dimensions
- 73% zeros, values in [0, 0.079], sum = 1.0

**WeightedJaccard similarity:**

```
WJ(a,b) = Σ min(aᵢ,bᵢ) / Σ max(aᵢ,bᵢ)
```

- Measures how much two polygons spatially overlap
- Output: 0.0 (no overlap) → 1.0 (identical)
- Global statistic — depends on ALL 18,220 dimensions simultaneously

> **Speaker note:** "Think of it as laying a polygon on graph paper and recording how much ink falls in each square. WeightedJaccard then measures how similarly two polygons filled their squares."

---

## Slide 4 — The Goal

**Title:** Can We Learn a Smaller Vector That Preserves Similarity?

**The compression goal:**
```
18,220 dims  →  [MLP]  →  512 dims   (36× smaller)
```

**Three constraints the compressed vector must satisfy:**
- Non-negative (all values ≥ 0)
- L1-normalized (values sum to 1.0)
- Similarity preserved — similar polygons stay close

**Why it's hard:**
- WJ is a global statistic — cannot process dimensions locally
- Input values are near zero — causes numerical collapse
- Standard neural architectures all fail

> **Speaker note:** "36× smaller vector. If we get this right, the index drops from 25 GB to under 1 GB."

---

## Slide 5 — Why Everything Else Failed

**Title:** Every Existing Architecture Fails — Here's Why

**Three root causes:**

**1. Train/Index Space Mismatch**
Trained in cosine space → indexed in WJ space → metric structure destroyed

**2. Local Decomposition**
Transformers chunk input into 64 tokens. WJ needs all 18,220 dims at once. Result: Recall@50 = 0.006

**3. Input Scale Collapse**
Raw values mean ≈ 10⁻⁹ → first layer std = 0.0000003 → numerically zero → model learns nothing

| Architecture | R@50 | Failure Mode |
|---|---|---|
| Transformer (L2) | 0.006 | Local decomposition |
| Transformer (WJ) | 0.020 | Space mismatch |
| Chi-Sq (no log1p) | 0.126 | Scale collapse |
| Deep Sets | ~0.01 | Collapses epoch 2 |

> **Speaker note:** "We didn't guess these failures — we ran every experiment and traced exactly why each one failed."

---

## Slide 6 — Fix 1: Input Preprocessing

**Title:** Fix 1 — Log1p Scaling

**The problem:**
```
Raw quadtree values:    mean ≈ 0.000000001
First layer output std: 0.0000003  ← numerically zero
```

**The fix:**
```
x̃ = log(1 + x × 1,000,000)
```

**The result:**
```
After fix — first layer std: 0.0023  ← 7,000× improvement
```

| | R@50 |
|--|--|
| Without log1p | 0.126 |
| With log1p | **0.593** |

> **Speaker note:** "One line of code. 4.7× improvement in recall. This was the first breakthrough."

---

## Slide 7 — The MLP Compressor

**Title:** Our Architecture: MLP Compressor

**Architecture:**
```
Input:   18,220 dims  (after log1p)
              ↓
Layer 1: Linear(18220 → 4096) → BatchNorm → ReLU
              ↓
Layer 2: Linear(4096  → 1024) → BatchNorm → ReLU
              ↓
Layer 3: Linear(1024  →  512) → BatchNorm
              ↓
Output:    512 dims
```

**Three key design choices:**

**Global projection (Layer 1)**
All 18,220 dims seen simultaneously — necessary because WJ is a global statistic

**BatchNorm**
Normalizes each dimension to [−3, +3] — without it model cannot distinguish any two polygons

**ReLU**
Kills ~50% of neurons — adds non-linearity so 3 layers beat 1

**Stats:** 79.4M parameters · 36× compression

> **Speaker note:** "The single most important choice is that first layer. It sees everything at once. That's what Transformers don't do — and why they fail."

---

## Slide 8 — Why Cosine Space, Not WJ Space?

**Title:** Training in Cosine Space Using WJ Labels

**The question:** GT is WJ-based. Why train in cosine space?

**Three reasons:**

**1. WJ has non-smooth gradients**
min/max operations → sparse gradients → flat loss landscape → optimizer stalls

**2. Cosine is smooth everywhere**
Dot product → clean gradient through all 79.4M weights → optimizer makes progress every step

**3. Cosine and WJ are correlated**
For non-negative L1-normalized vectors: Pearson correlation ≈ 0.87
They rank polygon pairs in nearly the same order

**The key insight:**
GT provides *ordering* — not exact similarity values. Cosine loss preserves this ordering. Stage 2 reranking corrects the 13% where they disagree.

> **Speaker note:** "We're not asking the model to output WJ scores. We're asking it to rank polygons correctly. Cosine space does that more cleanly."

---

## Slide 9 — Training the Model

**Title:** How We Trained the MLP

**Dataset:** 1.28M anchor-positive pairs from GT neighbor lists

**Loss function — In-Batch Hard Triplet Loss:**
```
L = max(0,  d(a,p)  −  d(a,n)  +  0.3)

a = anchor polygon
p = its GT neighbor (positive)
n = most similar non-neighbor in batch (hardest negative)
d = Euclidean distance after L2 normalization
```

**The counterintuitive finding:**

| Strategy | R@50 |
|---|---|
| **In-batch hard negatives** | **0.8016** |
| Precomputed hard negatives | 0.3562 |
| Curriculum (random → hard) | 0.7768 |

Hard negatives hurt recall even though they improve discriminability.

**Setup:** AdamW · lr=1e-3 · cosine annealing · 50 epochs · batch 1024 · 8× A100 · best loss 0.1423

> **Speaker note:** "Hard negatives feel like they should help — they force the model to work harder. But they actually break the global ordering WJ needs. Random in-batch negatives preserve the global structure better."

---

## Slide 10 — Two-Stage Retrieval Pipeline

**Title:** The Two-Stage Pipeline

**The insight that makes it work:**
- Missed GT neighbors score HIGHER in cosine space (0.9882) than false positives (0.9805)
- True GT neighbors ARE in the top-500 cosine candidates
- WJ reranking on GPU recovers them

**Stage 1 — Fast cosine search:**
```
Query polygon → MLP → 512-dim embedding → HNSW cosine → top-500 candidates
Speed: 30,150 QPS
```

**Stage 2 — Exact WJ reranking on GPU:**
```
500 candidates → exact WJ on original 18,220-dim vectors → re-sorted by true WJ
Corrects metric mismatch completely
```

**Result:** R@10 = 0.9966 — matches baseline exactly at 4.8× higher QPS

> **Speaker note:** "Stage 1 is fast but approximate. Stage 2 is exact but cheap — only 500 comparisons per query instead of 187,019."

---

## Slide 11 — Neural MinHash

**Title:** Second Architecture: Neural MinHash

**Classical MinHash idea:**
K random hash functions → K-dimensional signature
Probability of signature match = Jaccard similarity

**Our version:** Replace random hash functions with learned ones

**Architecture:**
```
log1p → Linear(18220→1024) → LayerNorm → GELU
       → Linear(1024→1024)  → LayerNorm → GELU
       → hash_proj(1024→512) → BatchNorm → ReLU → L1-norm
```

**Key differences from MLP:**
- LayerNorm (not BatchNorm) in extractor — per-sample, stable for sparse inputs
- Trained with WJ-space triplet loss directly
- Must use WJ HNSW index — train/index space must match
- 20.5M params — 4× more parameter-efficient than MLP

**10k results vs MLP:**

| | MLP | Neural MinHash |
|---|---|---|
| R@500 | 0.9540 | **0.9571** |
| Parameters | 79.4M | **20.5M** |
| Index memory | 44.8 MB | **16.6 MB** |

> **Speaker note:** "At 10k scale MinHash actually beats MLP at R@500 with 4× fewer parameters. At full 233k scale MLP wins — MinHash scales less well."

---

## Slide 12 — Full Results

**Title:** Results: 10k and 233k Scale

**10k dataset (8,000 corpus · 2,000 queries):**

| Method | R@10 | R@50 | R@500 | QPS |
|---|---|---|---|---|
| Baseline | 0.9966 | 0.9986 | 0.9974 | 246 |
| MLP+Cosine (no rerank) | 0.6655 | 0.8080 | 0.9540 | 30,150 |
| MLP K=500 + GPU rerank | **0.9966** | 0.9984 | 0.9540 | **2,904** |
| Neural MinHash | 0.6586 | 0.7964 | 0.9571 | 18,605 |

**Full 233k dataset (187,019 corpus · 46,754 queries):**

| Method | R@10 | R@50 | QPS | Memory |
|---|---|---|---|---|
| Baseline | 0.9925 | 0.9953 | 611 | ~25 GB |
| MLP K=1000 + GPU rerank | **0.9927** | 0.9952 | **986** | **~943 MB** |
| Neural MinHash | 0.5821 | 0.6604 | 1,822 | ~558 MB |

> **Speaker note:** "At full scale — R@10 = 0.9927 vs baseline 0.9925. Gap of 0.0002. Essentially identical recall. 26× less memory."

---

## Slide 13 — Efficiency Gains

**Title:** 26× Smaller. 4.5× Faster to Build. Same Recall.

| Metric | Baseline | Ours | Improvement |
|---|---|---|---|
| Vector memory | 12,999 MB | 365 MB | **35.6× smaller** |
| Index memory | 11,519 MB | 578 MB | **19.9× smaller** |
| Total memory | ~25,000 MB | ~943 MB | **26.5× smaller** |
| Build time | 560s | 124s | **4.5× faster** |
| QPS | 611 | 986 | **1.6× faster** |
| Recall@10 | 0.9925 | 0.9927 | **+0.0002** |

> **Speaker note:** "The baseline needs 25 GB. A standard server has 16-32 GB total RAM. Our method needs under 1 GB — it runs anywhere."

---

## Slide 14 — Adaptive K

**Title:** Bonus: Adaptive-K Reranking

**The insight:** Not all queries need the same K
- Easy queries (clear neighborhood) → K=100 is enough
- Hard queries (many similar polygons) → need K=1000

**How it works:**
Use cosine distance gap between rank-1 and rank-100 as a proxy for query difficulty.
Calibrate thresholds on held-out data. Assign K per query automatically.

**Results (10k):**

| Method | R@10 | R@50 | Avg K |
|---|---|---|---|
| Fixed K=1000 | 0.9966 | 0.9986 | 1000 |
| **Adaptive K** | **0.9966** | **0.9976** | **280** |

Same R@10. **3.6× fewer candidates on average.**

**K distribution across queries:**
- K=100 → 182 queries (easy)
- K=200 → 1,397 queries (most queries)
- K=500 → 317 queries
- K=1000 → 104 queries (hardest)

> **Speaker note:** "Most queries are easy — they only need 200 candidates. A few hard queries need 1000. Adaptive K gives each query exactly what it needs."

---

## Slide 15 — Limitations

**Title:** Limitations and Honest Caveats

- **Dataset-specific:** Model trained on parks GT only — not tested on lakes, buildings, or administrative boundaries
- **Grid dependency:** Input dimension (18,220) is hardcoded — new dataset with different geographic extent requires full retrain from scratch
- **GT bottleneck:** Brute-force WJ GT generation is O(N²) — the slow step is not ML training but GT generation (~8-10 hours for 233k polygons)
- **Reranking overhead:** Current Python NumPy reranking is the speed bottleneck — GPU C++ implementation needed for full production throughput
- **No cross-dataset generalization:** One polygon type tested only — true generalization requires multi-dataset training

---

## Slide 16 — Future Work

**Title:** What's Next

- **Listwise WJ distillation** — optimize full ranking order instead of pairwise triplets (code written, not yet executed)
- **Adaptive-K at full 233k scale** — currently validated on 10k only
- **C++ WJ reranking** — estimated 10× speedup → ~1,400 QPS, strictly faster than baseline on all metrics
- **Multi-dataset training** — parks + lakes + buildings simultaneously for true cross-domain generalization
- **Grid-agnostic architecture** — variable input dimension, works on any quadtree encoding regardless of geographic extent
- **Theoretical WJ embeddability bounds** — what is the minimum distortion possible when compressing WJ similarity to ℝᵐ?

---

## Slide 17 — Conclusion

**Title:** Summary

**Three key takeaways:**

**1. We found why everything fails**
Three root causes: space mismatch, local decomposition, scale collapse.
Architectural — not hyperparameter issues.

**2. We built the first working WJ-compatible neural compression**
MLP compressor + Neural MinHash.
Both satisfy non-negativity, L1-normalization, global structure preservation.

**3. Near-baseline recall at 26× less memory**
R@10 = 0.9927 vs 0.9925.
~943 MB vs ~25 GB.
Runs on a standard server.

---

## Slide 18 — Q&A

**Title:** Questions?

*(Place the three-pipeline system diagram here — index build + query flow for Baseline, MLP+Cosine+Rerank, Neural MinHash)*

**Anticipated questions and where to find answers:**

| Question | See |
|---|---|
| Why cosine not WJ training? | Slide 8 |
| Why not Transformers? | Slide 5 |
| Will it generalize to other datasets? | Slide 15 |
| Why do hard negatives hurt? | Slide 9 |
| What is inference? | It is just running the model forward — no learning, no weight updates |
| What if input dimension changes? | Slide 15 — grid dependency limitation |

---

*Total: 18 slides · Estimated presentation time: 20–25 minutes*
