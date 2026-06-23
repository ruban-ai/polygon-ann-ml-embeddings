# Weighted-Jaccard-Native Distillation for Compact Polygon Similarity Search

Learned, **Weighted-Jaccard-native** compression of high-dimensional ShapeToVec
polygon encodings into compact embeddings that stay directly indexable under
Weighted Jaccard (WJ) — for fast approximate nearest-neighbor (ANN) similarity
search over polygonal datasets with extreme area variability.

> **Paper:** *Weighted-Jaccard-Native Distillation for Compact Polygon Similarity
> Search under Extreme Area Variation* — Ruban Sampath, Buddhi Ashan M. K.,
> Sushil K. Prasad (University of Texas at San Antonio). Submitted to **ACM
> SIGSPATIAL 2026** (short paper). Sources in [`paper/`](paper/).

---

## Motivation

Finding similar shapes in large polygon datasets underpins deduplication,
conflation, and map integration. The natural overlap metric is **Weighted
Jaccard**:

```
WJ(a, b) = Σ min(aᵢ, bᵢ) / Σ max(aᵢ, bᵢ),   a, b ≥ 0
```

[**ShapeToVec**](https://doi.org/10.5281/zenodo.11522173) handles extreme area
variability with an adaptive *quadtree* grid, encoding each polygon as a
high-dimensional probability-simplex vector (~18k dims) whose WJ preserves shape
overlap, indexed with a custom WJ distance in HNSW (NMSLIB). The accuracy is
strong but the vectors are large — heavy to store and slow to search. **This
project learns a compact, WJ-compatible replacement.**

## Approach

- **WJ-native simplex encoder** — a 2-layer MLP (`D → 4096 → 4096`) with
  BatchNorm-before-ReLU and a `ReLU + ℓ1` output, so every embedding lies on the
  probability simplex and is indexable *directly* under WJ (no post-processing).
- **WJ-native distillation** — regress the *embedding* WJ onto the *true* WJ of
  the original D-dim vectors (`L = ‖ŴJ − WJ‖²`). The training objective **is** the
  retrieval metric, so the deployed output is exactly the optimized quantity.
- **Contrastive autoencoder (baseline route)** — encoder + decoder trained with a
  WJ triplet / InfoNCE loss (false-negative filtered) plus reconstruction. Strong
  at small scale, but collapses to random-projection level on held-out queries —
  it optimizes *separation*, not the global WJ *ranking* recall needs.
- **Matryoshka deployment** — one trained model serves every width
  `{256, 512, …, 4096}`; 256-d already dominates (72× compression).
- **Two-stage retrieval** — HNSW Stage-1 over the compact embeddings, then exact-WJ
  rerank on the original vectors (Numba/GPU).

## Headline results

**50K benchmark** (40k corpus / 10k held-out queries; geometric-Jaccard GT;
methods corpus-trained, shown at **256-d = 72× compression**):

| Method (256-d)      | Stage-1 R@500 | + rerank R@500 | Stage-1 QPS |
|---------------------|:-------------:|:--------------:|:-----------:|
| Random projection   | 0.766         | 0.955          | 3,890       |
| Triplet + recon     | 0.725         | 0.859          | 3,810       |
| InfoNCE–WJ          | 0.720         | 0.821          | 7,128       |
| **WJ-distillation** | **0.924**     | **0.976**      | **6,772**   |

- **10K subset:** WJ-distillation R@50 = 0.930 @ 14,552 QPS; R@10 > 0.996 after rerank.
- **Full 187K Parks corpus** (all 46,754 queries held out, same encoder unchanged):
  Stage-1 R@500 = 0.890 @ 2,136 QPS → 0.966 after rerank.

First learned WJ embedding to beat an untrained random projection on **both** the
near- and wide-net recall–throughput frontiers.

## Repository layout

```
.
├── paper/                      # SIGSPATIAL '26 LaTeX (main_distill.tex, refs.bib, tables)
├── figures/                    # paper figures (+ plotting fonts)
├── logs/                       # run logs (+ logs/scripts analysis helpers)
└── term_project/
    ├── 00–34_*.ipynb           # research notebooks (data prep, baselines, distillation, plots)
    ├── *.py                    # standalone eval / sweep / generalization scripts
    ├── utils/                  # requirements, nmslib WJ build helper, figure assets
    └── SigSpatial/             # paper experiment runners (the reproducible core)
        ├── build_50k.py                  # build the 50K dataset + GT-metric oracle
        ├── run_50k_ddp.py                # 50K benchmark: --method {distill,triplet,infonce,rp}
        ├── run_matdistill_fulleval_ddp.py# full 187K corpus, all queries held out
        ├── run_matdistill_10k.py         # 10K subset distillation
        ├── sota_experiment_common.py     # shared loaders, nmslib HNSW, GPU rerank, recall
        └── RESULTS_LOG.csv / NEW_RESULTS.csv  # single source of truth for results
```

## Datasets

We use the **publicly released ShapeToVec dataset** (quadtree encodings + shape-
similarity ground truth) — DOI [10.5281/zenodo.11522173](https://doi.org/10.5281/zenodo.11522173)
— derived from the SpatialHadoop Parks polygons. Subsets:

| Split | Corpus | Queries | Dim (D) |
|-------|:------:|:-------:|:-------:|
| 10K   | 8k     | 2k      | ~18,499 |
| 50K   | 40k    | 10k     | 18,382  |
| 187K  | 187k   | 46,754  | ~18,220 |

> The raw encodings (~10 GB, `encoding/`) and all checkpoints/`.npy`/`.pkl` outputs
> are **git-ignored** — only code, notebooks, paper, and logs are tracked. Point
> the builders at your local copy of the ShapeToVec data before running.

## Setup

```bash
pip install -r term_project/utils/requirements.txt
# numpy · torch · nmslib · tqdm · psutil · jupyter/notebook/ipykernel
```

**NMSLIB Weighted-Jaccard space:** retrieval uses ShapeToVec's *custom* WJ distance
added to NMSLIB. Use the prebuilt module (see
`term_project/utils/copy_nmslib_weighted_from_dgx.sh`) or build NMSLIB with the WJ
space enabled — the stock `nmslib` package does **not** include it.

## Usage

```bash
cd term_project/SigSpatial

# 1) Build the 50K dataset (parses encodings + GT, runs the raw-vs-normalized WJ oracle)
python build_50k.py

# 2) 50K benchmark — 8-GPU DDP, one method at a time
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 run_50k_ddp.py --method distill
#   --method ∈ {distill, triplet, infonce, rp}

# 3) Full 187K corpus, all queries held out
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 run_matdistill_fulleval_ddp.py

# 4) 10K moderate-scale study
python run_matdistill_10k.py
```

Each run appends `date, dataset, method, dim, stage, cand_k, R@10/50/100/500, QPS`
to `RESULTS_LOG.csv` — the single source of truth for the paper tables.

## Platform

Experiments ran on an **NVIDIA DGX A100**: 8 × A100-SXM4 80 GB GPUs, dual 64-core
AMD EPYC 7742 CPUs, 2 TB RAM. QPS is reported on a single GPU; NMSLIB HNSW queries
use multi-threaded CPU candidate generation followed by GPU exact-WJ reranking.

## Citation

```bibtex
@inproceedings{sampath2026wjdistill,
  title     = {Weighted-Jaccard-Native Distillation for Compact Polygon
               Similarity Search under Extreme Area Variation},
  author    = {Sampath, Ruban and Ashan M. K., Buddhi and Prasad, Sushil K.},
  booktitle = {Proceedings of the 34th ACM SIGSPATIAL International Conference
               on Advances in Geographic Information Systems},
  year      = {2026},
  note      = {Short paper, under review}
}
```

Built on **ShapeToVec** (Ashan M. K. et al., IEEE BigData 2025) and **NMSLIB**.
