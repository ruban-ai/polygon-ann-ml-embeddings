# Learned Embeddings for Polygon Similarity Search (HPML)

Machine learning pipeline for **large-scale polygon similarity search** on **quadtree spatial encodings**. Ground truth is **Weighted Jaccard (WJ)** overlap on high-dimensional cell vectors (~18k dims). The main idea is a **two-stage retriever**: a fast learned embedding stage for candidates, then **exact WJ** on the original vectors for accurate ranking.

Active development lives in **`term_project/`**. Older exploratory notebooks (GNN / PolyMP / Poly2Vec baselines) remain at the repo root for reference.

---

## Problem

Given a corpus of polygons and query polygons, return the top-*K* most similar corpus polygons per query.

| Piece | Detail |
|--------|--------|
| **Input** | Quadtree / spatial-cell encoding per polygon (~18,000 dimensions) |
| **Similarity (GT)** | Weighted Jaccard on nonnegative cell weights: \(J_w = \sum\min / \sum\max\) (equivalently intersection over union mass) |
| **Split** | Corpus = low IDs; queries = high IDs (e.g. 10k set: 8k corpus / 2k queries) |
| **Metric** | Recall@K against precomputed GT neighbor lists |

Exact WJ on 18k-D vectors is accurate but expensive at scale. This project learns **512-D embeddings** that approximate WJ structure for fast search, then **reranks** a short candidate list with exact WJ on GPU.

---

## Repository layout

```
hpmlproj/
├── term_project/          # Main HPML pipeline (notebooks + scripts)
│   ├── 00_cache_data.ipynb
│   ├── 01_baseline.ipynb … 10_results.ipynb
│   ├── 15_mlp_wj_native.ipynb      # WJ-ratio triplet + WJ HNSW
│   ├── 16_mlp_intersection_min.ipynb  # Intersection-only training (advisor direction)
│   ├── 17_eval_sports_50k_zeroshot.py # Cross-dataset eval (no retrain)
│   ├── readme.txt           # Detailed run order, paths, /tmp artifacts
│   └── requirements.txt
├── best_model/              # Saved checkpoints (when present)
├── MLP baseline.ipynb       # Early 50k sports-style weightint experiments
├── SOTA*.ipynb, polygon_gnn_*.ipynb  # Legacy GNN / PolyMP / Poly2Vec comparisons
├── presentation_content.md  # Talk slides / result tables
└── readme.md                # This file
```

For step-by-step reproduction, platform notes, and `/tmp` file names, see **`term_project/readme.txt`**.

---

## Datasets

| Dataset | Role | Encodings | Ground truth | Notes |
|---------|------|-----------|--------------|--------|
| **Parks 10k** | Primary dev / train | `pk-real10k0.002` (`real_*.txt`, dense float, **18499-D**) | `pk-query-10k` | 8k corpus / 2k queries; cached as `/tmp/qt_10k.npy` |
| **Parks full** | Scale experiments | Full quadtree cache | `pk-query-187019` | ~187k corpus / ~47k queries; dim can differ slightly from 10k |
| **Sports 50k** | Transfer / zero-shot eval | `sports-50k0.002` (`weightint_*.txt`, binary **18382-D**) | `sports-query-50k` | 40k corpus / 10k queries; see `17_eval_sports_50k_zeroshot.py` |
| **Overture 50k** | Transfer eval script | `overture-50k0.020` | `overture-query-50k` | `term_project/14_generalize_overture.py` |

**Orion paths (current):**

- Parks 10k encodings: `/mnt/data1/shapeSimilarity/encodings/pk-real10k0.002`
- Parks 10k GT: `/mnt/data1/shapeSimilarity/warehouse/pk-query-10k`
- Sports 50k encodings: `/mnt/data1/ruban/encodings/sports-50k0.002`
- Sports 50k GT: `/mnt/data1/ruban/groundtruth/sports-query-50k`

Large raw files are not in git. Run **`term_project/00_cache_data.ipynb`** once to build `/tmp` caches.

---

## Methods (current)

### 1. Baseline — exact WJ + HNSW

`01_baseline.ipynb`: **nmslib** HNSW with custom **`WeightedJaccard`** space on original quadtree vectors (CPU). This is the accuracy reference and defines GT alignment.

Typical **parks 10k**: Recall@10 ≈ **0.997** (order-of-magnitude; see `01_baseline` / `10_results` for your run).

Requires a WeightedJaccard-enabled nmslib build (see `term_project/readme.txt` and `install_weighted_jaccard/`).

### 2. Two-stage MLP (main line)

Train an MLP **18499 → 512**, search with **cosine HNSW** (or variants), then **GPU batched exact WJ rerank** on top-*K* candidates (`04`–`06`, `10_results`).

- Strongest historical variant in this line: **hard-negative WJ distillation** (`06_mlp_hard_negative_distillation.ipynb`).
- **Parks 10k** with K≈500–1000 rerank: Recall@10 ≈ **0.99+**, often matching baseline with much lower memory and higher effective QPS when amortizing rerank.

### 3. WJ-aligned embedding training (advisor experiments)

| Notebook | Training objective | Stage-1 search | Rerank |
|----------|-------------------|----------------|--------|
| **`15_mlp_wj_native`** | WJ **ratio** triplet on L1-simplex embeddings | HNSW **WeightedJaccard** | Optional raw WJ |
| **`16_mlp_intersection_min`** | **Intersection only** \(\sum\min\) triplet (no ratio in loss); scratch init | GPU **intersection** on 512-D | Raw WJ ratio for eval only |

**Parks 10k (`16`, scratch, no cosine warm-start):**

| Stage | Recall@10 (approx.) |
|-------|---------------------|
| 512-D intersection KNN only | **~0.65** |
| K=1000 candidates + raw WJ rerank | **~0.99** |

Reference cosine MLP (`02`, `best_model`): ~**0.67** no rerank, ~**0.997** with rerank on 10k.

Training still uses **GT neighbor lists** to sample (query, positive) pairs; it does not use GT scores as input features.

### 4. Cross-dataset zero-shot

**`17_eval_sports_50k_zeroshot.py`**: Load parks-trained **`/tmp/best_compressor_intersection_min_10k.pt`**, embed sports polygons (pad 18382 → 18499), evaluate **without retraining**.

| Stage | Sports 50k Recall@10 (reported run) |
|-------|-------------------------------------|
| 512-D intersection KNN | **0.26** |
| K=1000 + raw WJ rerank | **0.67** |

Shows **domain shift** (encoding format and dataset) but rerank still recovers substantial signal. Results: `/tmp/results_sports_50k_zeroshot.pkl`.

### 5. Legacy comparisons (repo root)

Notebooks such as **`SOTA2_PolyMP*.ipynb`**, **`polygon_gnn_jaccard.ipynb`**, and **`poly2vec/`** explore GNN / boundary-descriptor embeddings. Early runs showed **poor recall vs cell-based WJ** when GT is defined on quadtree overlap—not because GNNs are useless in general, but because **GT metric and representation were misaligned**. Those artifacts are kept for the project narrative; they are not the active retrieval path.

---

## Quick start

```bash
conda activate hpmlproj   # or env with torch, nmslib WeightedJaccard, etc.
cd term_project
pip install -r requirements.txt
```

1. Build caches: run **`00_cache_data.ipynb`**
2. Baseline: **`01_baseline.ipynb`** (10k)
3. Main learned + rerank: **`04_two_stage_mlp_wj_rerank.ipynb`** or **`06_mlp_hard_negative_distillation.ipynb`**
4. Summarize: **`10_results.ipynb`**
5. Advisor line: **`16_mlp_intersection_min.ipynb`**
6. Sports transfer:  
   `python 17_eval_sports_50k_zeroshot.py`

Use **`cuda:0`** where notebooks expect GPU reranking.

---

## Key artifacts (`/tmp`)

| File | Purpose |
|------|---------|
| `qt_10k.npy` | Parks 10k quadtree matrix |
| `gt_lookup_10k.pkl` | Query → GT neighbor IDs |
| `best_compressor_intersection_min_10k.pt` | Scratch intersection-min checkpoint |
| `best_compressor_v1_clean.pt` / `best_compressor_full_fixed.pt` | Cosine MLP checkpoints |
| `results_*.pkl` | Per-experiment recall / QPS summaries |
| `qt_sports_50k.npy`, `gt_lookup_sports_50k.pkl` | Sports cache (from script 17) |

Checkpoints may also live under **`best_model/`** depending on what was copied off the training machine.

---

## Design lessons (accurate summary)

1. **Align the learned space with the retrieval metric.** Cosine on unconstrained MLP outputs underperforms WJ-aware training or intersection-based search on simplex embeddings.
2. **Two-stage is practical.** Fast approximate stage + exact WJ rerank reaches baseline-level Recall@10 on parks while cutting index size and enabling GPU batching.
3. **Intersection-only training** (Dr. Prasad): optimizing \(\sum\min\) avoids ratio gradients; parks 10k intersection-only scratch model is competitive with cosine MLP before rerank.
4. **Transfer is hard.** Parks-trained models on sports drop sharply without finetuning; exact rerank on raw cells remains important on new domains.
5. **GT defines labels, not features.** Neighbor lists supervise contrastive/triplet training; evaluation still uses the same WJ GT as the baseline.

---

## Citation / context

HPML term project (UTA): learned retrieval for geospatial polygon similarity with Weighted Jaccard ground truth. Advisor direction includes WJ-native indexing, intersection-based losses, and generalization across datasets (parks, sports, Overture).

For tables and speaker notes, see **`presentation_content.md`**.
