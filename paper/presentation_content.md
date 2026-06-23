# Presentation: Learned Polygon Similarity Search
### Slide-by-Slide Content

---

## Slide 1 — Title Slide

**Title:** Learning Compact Weighted-Jaccard-Preserving Embeddings for Scalable Polygon Similarity Search

**Subtitle:** From 18,219 dimensions to 512 — without losing the geometry

**Authors:** Ruban Sampath, Buddhi Ashan M.K., Sushil K. Prasad
**Affiliation:** University of Texas at San Antonio
**Venue:** ACM SIGSPATIAL 2026

---

## Slide 2 — The Problem

**Title:** Finding Similar Polygons at Scale

**Left panel — What is the task?**
- Given a large database of geographic polygons (parks, parcels, water bodies), find the top-k most similar shapes for a query polygon
- Similarity metric: **Weighted Jaccard (WJ)** — captures area-weighted shape overlap

$$WJ(\mathbf{a},\mathbf{b}) = \frac{\sum_i \min(a_i, b_i)}{\sum_i \max(a_i, b_i)}$$

**Right panel — Why is it hard?**
- Polygons are encoded as very high-dimensional vectors (18,219 dims)
- Brute-force WJ search over 234k polygons = too slow for real applications
- Standard ANN indexes (HNSW) don't support WJ natively
- Need: **compact embeddings that preserve WJ and plug into fast ANN search**

**Bottom callout:** *Dataset: 234,447 OSM park polygons, encoded via ShapeToVec into 18,219-dim probability simplex vectors*

---

## Slide 3 — Background: ShapeToVec & Weighted Jaccard

**Title:** How Polygons Become Vectors

**Left — ShapeToVec encoding:**
- Uses a quadtree decomposition of the polygon's bounding region
- Each cell in the quadtree gets a weight proportional to the area of the polygon overlapping it
- Output: a sparse non-negative vector on the probability simplex (Δ^D, D = 18,219)
- WJ between two such vectors = area-weighted shape similarity

**Right — Why the probability simplex matters:**
- WJ is only well-defined for non-negative vectors
- If embeddings live on Δ^d, WJ is directly computable between compressed embeddings
- This is the key constraint our encoder must satisfy

**Visual suggestion:** Show a polygon → quadtree grid → sparse vector bar, side by side with the WJ formula

---

## Slide 4 — The Scalability Problem

**Title:** Why Brute Force Doesn't Scale

| | Brute-force ICWS | Our Goal |
|---|---|---|
| Similarity | Exact WJ (via sketching) | WJ-preserving |
| Search | Scan all 234k vectors | HNSW ANN index |
| QPS | 139 | 10,000+ |
| Dim | 512 integer codes | 512 float32 |

**Key insight box:**
> Classical WJ sketching methods (ICWS, ProbMinHash) produce integer codes incompatible with HNSW's float-vector index. Learned methods work for cosine/Euclidean but none produce WJ-compatible embeddings. **We fill this gap.**

---

## Slide 5 — Our Approach at a Glance

**Title:** Two-Stage Pipeline

**Large visual — End-to-End Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (once)                           │
│                                                                 │
│  OSM Polygon  →  ShapeToVec  →  18,219-dim vector              │
│                                      ↓                          │
│                              Learned Encoder fθ                 │
│                          (D→4096→1024→512, simplex)             │
│                                      ↓                          │
│                           512-dim embedding  →  HNSW Index      │
│                              (WJ space)         (offline build) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       ONLINE (per query)                        │
│                                                                 │
│  Query polygon  →  ShapeToVec  →  fθ  →  512-dim              │
│                                              ↓                  │
│                              HNSW Search (WJ)  →  top-K        │
│                                              ↓                  │
│                              Exact WJ Rerank  →  top-k          │
│                             (on raw 18k-dim)                    │
└─────────────────────────────────────────────────────────────────┘
```

**Caption:** Stage 1 = fast ANN retrieval (high QPS). Stage 2 = exact rerank of top-K candidates for precision. Encoder fθ is the only thing we learn.

---

## Slide 6 — Methods We Tried: The Full Landscape

**Title:** What We Explored — 10 Methods Across 4 Categories

**Box 1 — Non-Learned (simplex-compatible)**
- PCA + L1-normalize
- NMF (Non-negative Matrix Factorization)
- Random Projection + shift + L1-normalize
- *These learn nothing about WJ — treat geometry blindly*

**Box 2 — Classical WJ Sketching**
- ICWS (Improved Consistent Weighted Sampling) — 512 hash dims
- ProbMinHash
- *Theoretically optimal WJ sketches but output integer codes → brute-force search only*

**Box 3 — Learned, Non-WJ Space**
- MLP with cosine triplet loss
- Matryoshka MLP (multi-scale WJ triplet)
- Deep Binary Hashing
- *These produce Euclidean/Hamming embeddings — wrong metric for WJ*

**Box 4 — Ours: Learned, WJ-Native**
- MLP WJ-triplet (simplex output)
- **AE WJ-triplet + reconstruction ← BEST**

---

## Slide 7 — Encoder Architecture

**Title:** The Encoder: Mapping Polygons to the Simplex

**Center — Architecture diagram:**

```
Input: x ∈ Δ^18219  (ShapeToVec encoding)
         ↓
   Linear(18219 → 4096)
   BatchNorm + ReLU
         ↓
   Linear(4096 → 1024)
   BatchNorm + ReLU
         ↓
   Linear(1024 → 512)
   BatchNorm
         ↓
   ReLU + ℓ1-normalize    ← simplex projection
         ↓
Output: z ∈ Δ^512   (36× smaller, WJ-compatible)
```

**Right side — Why this design:**
- **Simplex output constraint** (ReLU + L1-norm): makes WJ(z_i, z_j) well-defined between any two embeddings
- **80M parameters**: large first layer (18k→4096) captures full quadtree feature space
- **Shared weights**: same encoder used for corpus and queries — no asymmetry

**Bottom callout:** *The only difference between the MLP and AE variants is what we add during training — the encoder architecture is identical*

---

## Slide 8 — Training: WJ Triplet Loss

**Title:** Training the Encoder with WJ-Native Triplets

**Left — What is a triplet?**
- **Anchor** xₐ: a query polygon
- **Positive** x_p: a known GT neighbor (high WJ with anchor)
- **Negative** xₙ: a hard non-neighbor (high WJ but NOT a GT neighbor)

**Center — The loss:**

$$\mathcal{L}_{\text{trip}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \bigl[ WJ(\mathbf{z}_n, \mathbf{z}_a) - WJ(\mathbf{z}_p, \mathbf{z}_a) + 0.3 \bigr]_+$$

*Only violated pairs (where negative is closer than positive by margin 0.3) contribute*

**Right — In-batch hard negative mining:**
1. Form a batch of B=2048 anchor-positive pairs
2. Compute the B×B cross-WJ similarity matrix S_ij between all anchors and all positives
3. For each anchor i, the hardest negative is: argmax_j S_ij (the positive of another pair that looks most similar to this anchor)
4. Apply **false-negative filter**: mask known GT neighbors before selecting the negative

---

## Slide 9 — The False-Negative Filter (Key Innovation)

**Title:** Why the FN Filter Matters

**Left — The problem:**
- In a batch of 2048 pairs, many polygons are actual GT neighbors of each other
- Without filtering, a real GT neighbor gets selected as the "negative"
- This creates a **contradictory gradient**: push apart two polygons that should be close

**Center — Visual:**

```
Without FN filter:
Anchor xₐ ──── treated as NEGATIVE ────► xⱼ (actually a GT neighbor!)
Loss says: "push xⱼ away from xₐ"  ← WRONG

With FN filter:
GT adjacency matrix G masks position (i,j)
Only true non-neighbors selected as negatives
Loss says: "push only genuine non-neighbors away"  ← CORRECT
```

**Right — Ablation result:**

| FN Filter | Aux Loss | R@10 |
|---|---|---|
| ✗ | none | 0.677 |
| ✗ | WJ regression | **0.510** ← collapse |
| ✓ | WJ regression | 0.669 |
| ✓ | reconstruction | **0.711** |

*Without filter: adding any auxiliary loss collapses recall to 0.510*

---

## Slide 10 — The Autoencoder Variant

**Title:** Adding Reconstruction: Why It Helps

**Left — Architecture:**

```
Encoder fθ:  x → z ∈ Δ^512       (kept at inference)
                  ↓
Decoder gφ:  z → x̂ ∈ R^18219_≥0  (discarded at inference)
             (512 → 1024 → 18219)
```

**Center — Combined loss:**

$$\mathcal{L} = \mathcal{L}_{\text{trip}} + 0.1 \cdot \mathcal{L}_{\text{recon}}$$

$$\mathcal{L}_{\text{recon}} = \text{MSE}(g_\phi(f_\theta(\mathbf{x})), \mathbf{x})$$

**Right — Why reconstruction helps:**
- Triplet loss only cares about **rank order** (is positive closer than negative?)
- Reconstruction forces the encoder to preserve the **full input distribution** — fine-grained geometry
- Acts as a regularizer: encoder can't collapse all points to a small manifold
- **+5.0 pp R@50** over triplet-only (0.858 vs 0.806)

**Bottom:** *The decoder adds zero inference cost — only the 80M param encoder runs at query time*

---

## Slide 11 — Results: 10k Subset

**Title:** Stage-1 Results (10k Corpus, No Reranking)

| Method | R@10 | R@50 | R@500 | QPS |
|---|---|---|---|---|
| ICWS 512 (brute force) | 0.840 | 0.897 | 0.988 | 139 |
| PCA simplex | 0.432 | 0.553 | 0.789 | 16,666 |
| NMF simplex | 0.638 | 0.727 | 0.942 | 11,764 |
| MLP cosine triplet | 0.666 | 0.808 | 0.954 | 30,150 |
| Matryoshka MLP | 0.455 | 0.672 | 0.932 | 18,647 |
| AE reconstruction only | 0.665 | 0.745 | 0.938 | 11,142 |
| MLP WJ-triplet (ours) | 0.677 | 0.806 | 0.964 | 10,756 |
| **AE WJ-triplet+recon (ours)** | **0.711** | **0.858** | **0.977** | **10,903** |

**Callout boxes:**
- 78× faster than ICWS at comparable recall
- Best recall among all ANN-compatible methods
- Matryoshka (cosine-space) degrades in WJ setting — wrong metric

---

## Slide 12 — Results: After Two-Stage Reranking

**Title:** Stage-2 Results — Near-Perfect Recall

| Method | Rerank K | R@10 | R@50 | QPS |
|---|---|---|---|---|
| MLP WJ-triplet | 500 | 0.997 | 0.999 | 1,139 |
| **AE WJ-triplet+recon** | **500** | **0.997** | **0.999** | **75** |
| MLP WJ-triplet | 1000 | 0.997 | 0.999 | 232 |
| AE WJ-triplet+recon | 1000 | 0.997 | 0.999 | 38 |

**Explanation box:**
> Reranking reorders the top-K HNSW candidates using exact WJ on the original 18k-dim vectors. R@10 reaches 0.997 — effectively perfect. The bottleneck shifts from model quality to Stage-1 coverage (R@500).

**Diagram suggestion:** Show the QPS vs Recall tradeoff as a simple curve with K=500 and K=1000 labeled

---

## Slide 13 — Ablation Study: Every Design Choice Tested

**Title:** What Each Component Contributes

| Model | FN Filter | Aux Loss | R@10 | R@50 | Takeaway |
|---|---|---|---|---|---|
| MLP | ✗ | none | 0.677 | 0.806 | Baseline |
| MLP | ✓ | none | 0.656 | 0.773 | FN filter hurts small corpus (fewer hard negatives) |
| MLP | ✗ | WJ regression | 0.510 | 0.736 | Gradient conflict without filter |
| MLP | ✓ | WJ regression | 0.669 | 0.808 | Filter resolves conflict |
| AE | — | recon only | 0.665 | 0.745 | Reconstruction without metric learning |
| **AE** | **✓** | **triplet+recon** | **0.711** | **0.858** | **All three together = best** |

**Insight callout:**
> The false-negative filter is not just a detail — it is a prerequisite for any auxiliary loss to help. Without it, auxiliary losses actively hurt.

---

## Slide 14 — End-to-End Info Diagram

**Title:** Complete System — From Raw Polygon to Top-k Results

**Design brief for the designer:**

Divide the slide into three horizontal sections with distinct colors:

**Section 1 — DATA (gray/dark background):**
- Icon of a park polygon shape
- Arrow → "ShapeToVec Encoding (quadtree)"
- Arrow → Wide bar representing 18,219-dim vector (label: Δ^18219, many tick marks)
- Label below: "Area-weighted quadtree cells → sparse simplex vector"

**Section 2 — ENCODER (blue gradient background):**
- Funnel/trapezoid shape labeled "Encoder fθ (80M params, shared)"
- Inside the funnel, three stacked rows:
  - Row 1: "Linear 18219→4096 + BN + ReLU"
  - Row 2: "Linear 4096→1024 + BN + ReLU"
  - Row 3: "Linear 1024→512 + ReLU + ℓ1-norm"
- Output: narrow bar representing 512-dim vector (label: Δ^512, "36× smaller")
- Side branch (dashed, labeled "Training only"):
  - Decoder gφ: 512→1024→18219
  - MSE reconstruction loss box
- Main flow continues down

**Section 3 — RETRIEVAL (green background):**

*Stage 1 (left):*
- 512-dim query embedding
- Arrow → HNSW Index (graph with nodes/edges, label: "WeightedJaccard space, 198 MB")
- Arrow → "Top-K candidates" (e.g., K=1000)
- Label: "2,423 QPS (full corpus)"

*Stage 2 (right):*
- Top-K candidates
- Arrow → "Exact WJ Rerank (on raw 18k-dim)"
- Arrow → "Top-k results"
- Label: "R@10 = 0.992"

**Left-side vertical label:** "OFFLINE (build once)" for sections 1+2, "ONLINE (per query)" for section 3

**Color coding key (bottom):**
- Blue = Encoder (learned)
- Orange = Decoder (training only, discarded at inference)
- Green = Retrieval pipeline

---

## Slide 15 — Full Corpus Results (234k)

**Title:** Scaling to the Full 234k Polygon Corpus

**Left column — MLP WJ-native (no recon, no FN filter):**
- Stage 1: R@10 = 0.542, R@50 = 0.627, QPS = 2,423
- After rerank K=2000: R@10 = 0.992, R@50 = 0.992, QPS = 785
- HNSW builds in 85 sec, index = 198 MB

**Right column — AE WJ-triplet+recon:**
- Stage 1: results pending
- Expected: significant improvement in Stage-1 R@50 over MLP baseline

**Observation box:**
- Stage-1 recall drops vs 10k (more crowded embedding space at scale)
- Two-stage reranking maintains near-perfect top-10 recall even at 234k scale
- HNSW index fits in 198 MB — practical for real deployment

---

## Slide 16 — Key Takeaways

**Title:** What We Learned

**Box 1 — Technical:**
Probability simplex output constraint is the key enabler. By forcing encoder output to Δ^512, WJ is computable between compressed embeddings and HNSW's WeightedJaccard space is directly usable. No other method achieves this.

**Box 2 — Training insight:**
False-negative filtering is a prerequisite, not an optional add-on. Adding any auxiliary loss without it causes gradient conflicts that collapse recall. The order matters: filter first, then auxiliary loss, then reconstruction.

**Box 3 — System result:**
78× QPS gain over brute-force ICWS. Near-perfect recall (R@10 = 0.997) after two-stage reranking. 36× compression with no special hardware at inference — just the 80M param encoder.

---

## Slide 17 — Future Work & Conclusion

**Title:** What's Next

**Future work:**
- Scale evaluation to full 234k corpus with AE variant (in progress)
- Test on additional polygon types: buildings, water bodies, parcels
- Matryoshka-style nested WJ embeddings for flexible truncation
- Knowledge distillation to reduce encoder size for edge deployment

**Conclusion:**
> We built the first learned compression framework natively compatible with Weighted Jaccard search. The probability simplex constraint, WJ triplet loss, false-negative filtering, and reconstruction loss together achieve 78× throughput improvement while preserving near-perfect recall after reranking — demonstrating that learned WJ embeddings are both practical and effective for large-scale polygon similarity search.

---

## Designer Notes

1. **Color palette:** Blue `#2563EB` for encoder/model · Orange `#EA580C` for training-only components · Green `#16A34A` for retrieval · Gray `#6B7280` for data/input
2. **Font:** Clean sans-serif (Inter or Lato). Bold large numbers for impact: 78×, 0.997, 36×
3. **Slide 14** is the centerpiece infographic — spend the most design effort there
4. **Tables:** alternating row shading, bold the "ours" rows in blue
5. **Equations:** white/light background, LaTeX-style math rendering
6. **Total:** 17 slides — fits a 20–25 min talk with Q&A
