# Learned Weighted-Jaccard Signatures for Compact Polygon Similarity Search

Code and evaluation scripts for the paper:

> **Learning Compact Polygon Signatures via Weighted-Jaccard-Native Distillation for
> Large-Scale Similarity Search under Extreme Area Variation**
> Ruban Sampath, Buddhi Ashan M. K., Sushil K. Prasad (University of Texas at San Antonio).
> Submitted to the BSD 2026 workshop at IEEE BigData 2026 (under review).

We compress ShapeToVec's high-dimensional polygon vectors into short **learned
signatures** that are searched directly under the Weighted Jaccard (WJ) distance with
an HNSW index. A two-layer encoder is trained by **Listwise WJ distillation**: for each
polygon, its ranking of similar shapes under the exact WJ similarity of the original
float32 vectors is distilled into the compact embedding.

---

## Background

[ShapeToVec](https://doi.org/10.5281/zenodo.11522173) encodes each polygon on an
adaptive quadtree grid as a high-dimensional float32 vector, whose WJ similarity

```
WJ(a, b) = Σ min(aᵢ, bᵢ) / Σ max(aᵢ, bᵢ),   a, b ≥ 0
```

approximates the polygons' Jaccard index, even when areas span many orders of
magnitude. Its vectors are large. ShapeToVec+ shrinks them with bit-quantized
signatures, but loses recall at its most compact (1-bit) setting. This repository
replaces those designed signatures with learned ones.

## Method

- **Preprocessing.** Each input vector is divided by its sum before it enters the
  encoder. The teacher always uses the original, unnormalized vectors.
- **Encoder.** A two-layer MLP (`D → 4096 → 4096`; linear, batch norm, ReLU) whose
  output is made nonnegative and normalized to sum to one, so embeddings lie on the
  probability simplex and are compared directly under the WJ distance. About 29M
  parameters at a 3K input.
- **Listwise WJ distillation.** Within each training batch, a softmax over the
  teacher's exact WJ similarities is matched by a softmax over the embeddings' WJ
  similarities (cross-entropy, τ = 0.1). Batches pair each polygon with one of its
  30 nearest corpus neighbors, found once by exhaustive GPU search.
- **One encoder, many widths.** Matryoshka-style training: the loss is applied to
  nested prefixes (64 to 4,096 dimensions; 64 to 1,024 on Overture), so one trained
  encoder serves every width by truncation and renormalization.
- **8-bit signatures.** After training, each dimension is stored in 8 bits with one
  shared per-dimension scale (99.9th percentile), with no retraining. A 256-dimensional
  signature takes 256 bytes.

## Main results

Recall@50 at 256 dimensions (8-bit signatures, 256 bytes per polygon), all methods
starting from the same 3K-dimensional input vectors:

| Dataset | Polygons | Ours (256 B) | ShapeToVec+ 1-bit (~384 B) | ShapeToVec+ ARE 8-bit (~3 KB) |
|---|---|:---:|:---:|:---:|
| Parks | 234K | **0.90** | 0.76 | 0.96 |
| Water Bodies | 449K | **0.89** | 0.74 | 0.96 |
| Sports | 1.75M | **0.89** | 0.74 | 0.96 |
| Overture 1M | 1M | **0.79** | ~0.64 | ~0.94 |
| Overture 10M | 10M | **0.73** | ~0.55 | ~0.91 |

Index memory and search throughput on Overture 10M (64 threads, ShapeToVec+'s SIMD
distance kernel, median of 20 runs). Throughput covers search only; queries are
encoded beforehand, and memory excludes the encoder (117 MB).

| Method | Bytes / polygon | Index | QPS | Recall@50 |
|---|:---:|:---:|:---:|:---:|
| Ours, 64-d | 64 | 2.2 GB | 20.1K | 0.64 |
| Ours, 256-d | 256 | 4.1 GB | 15.6K | 0.73 |
| ShapeToVec+ 1-bit | 384 | ~5.8 GB | ~7.0K | ~0.55 |
| ShapeToVec+ 2-level | 768 | ~10.1 GB | ~4.7K | ~0.69 |
| ShapeToVec+ ARE (8-bit) | 3,072 | ~35.6 GB | ~2.3K | ~0.91 |

Baselines on Parks at 256 dimensions (Recall@50): Listwise WJ distillation 0.90,
exhaustive ICWS sketches 0.74, MSE distillation 0.80, random projection 0.63, NMF 0.58,
triplet 0.54, InfoNCE 0.51, PCA 0.08.

Offline cost (Overture): training the encoder on 1M polygons takes 83 minutes on eight
A100 GPUs (22 of them for neighbor mining); encoding all 10M polygons takes 2.3 minutes
on one GPU; building the 256-dimensional HNSW index takes 15 minutes.

## Where each result comes from

All experiment code is in [`term_project/SigSpatial/`](term_project/SigSpatial/).
Configurations (`parksfull`, `wbres_3k`, `sportsres_3k`, `ov1m`, `ov10m`, ...) are
defined in [`fulls_common.py`](term_project/SigSpatial/fulls_common.py).

| Paper item | Script(s) |
|---|---|
| Building the full Water Bodies and Sports benchmarks | `build_wbfull.py`, `build_sportsfull.py` |
| Encoder training (all datasets; `--method listwise\|triplet\|infonce`) | `run_fulls_listwise_train.py` |
| Recall of the float32 embeddings, all widths | `eval_fulls_listwise.py` |
| 8-bit and 4-bit signatures (Tables IV, V; Fig. 8) | `eval_quant.py` |
| Untrained baselines: PCA, NMF, random projection (Fig. 4) | `run_parksfull_unsup_baselines.py` |
| Exhaustive ICWS sketches, 256 samples (Fig. 4) | `run_icws_brute_full_256.py` |
| MSE distillation baseline (Table II) | `run_matdistill_fulleval_ddp.py` |
| Robustness across area quartiles (Table III) | `eval_bucketed_parksfull_m64.py` |
| Input-dimension study, 3K/6K/12K inputs (Fig. 5) | `build_parks_res.py`, `build_res.py` |
| Hidden vs. output layer of each loss (Section IV-F) | `analysis_layer_tap_v2.py` |
| Recall vs. index size, 1M to 10M (Table VI) | `eval_ov10m_slices.py` |
| Throughput, latency, and index memory (Table VII, Fig. 6) | `export_codes_are.py`, `bench_are_v2.py`, `run_bench_v2_20261026.sh` |
| Figures 4 to 8 | `make_fig_baselines_20261025.py`, `make_fig_resolution.py`, `make_fig_pareto_10m_v2.py`, `make_fig_recall_bytes_opt1.py`, `make_fig_precision.py` |

Every evaluation appends a row (date, dataset, method, width, R@10/50/100/500, QPS) to
[`RESULTS_LOG.csv`](term_project/SigSpatial/RESULTS_LOG.csv), the source of all
numbers in the paper.

## Data

- **Parks, Water Bodies, Sports:** the public ShapeToVec dataset (quadtree encodings
  and geometric Jaccard ground truth), DOI
  [10.5281/zenodo.11522173](https://doi.org/10.5281/zenodo.11522173). Each collection
  is split 80/20 into an indexed corpus and held-out queries; every query is unseen
  during training.
- **Overture 1M and 10M:** encodings and ground truth from ShapeToVec+, with the same
  1,000 held-out queries at both sizes. The 10M index uses the encoder trained on only
  the first 1M corpus polygons.

Encodings, checkpoints, embeddings, and other large outputs are not tracked in git.
Data paths are set at the top of `fulls_common.py`; point them at your local copy
before running.

## Setup

```bash
pip install -r term_project/requirements.txt   # numpy, torch, nmslib, scikit-learn, matplotlib, tqdm
```

- **WJ distance in HNSW.** Recall evaluations use ShapeToVec's custom Weighted Jaccard
  space in NMSLIB; the stock `nmslib` package does not include it, so build NMSLIB
  with that space enabled.
- **Throughput benchmark.** `bench_are_v2.py` uses ShapeToVec+'s patched NMSLIB space
  (`WeightedJaccardRatio`, its SIMD integer kernel), so the comparison with ShapeToVec+
  uses the same code, machine, and HNSW parameters.

## Usage

```bash
cd term_project/SigSpatial

# Train one encoder (8 GPUs); cfg is one of the CONFIGS in fulls_common.py
torchrun --standalone --nproc_per_node=8 run_fulls_listwise_train.py ov10m

# Recall of the float32 embeddings at every width
python eval_fulls_listwise.py ov10m --threads 64

# 8-bit signatures
python eval_quant.py ov10m --prefixes 64,256 --bits 8

# Throughput and memory with ShapeToVec+'s kernel
python export_codes_are.py ov10m --prefixes 64,256
python bench_are_v2.py ov10m 256
```

HNSW parameters in all evaluations: M = 20, efConstruction = 200, efSearch = 200.

## Hardware

NVIDIA DGX A100: 8 × A100 80 GB GPUs, dual 64-core AMD EPYC 7742 CPUs, 2 TB RAM.
Throughput uses 64 CPU query threads, matching ShapeToVec+.

## Citation

```bibtex
@inproceedings{sampath2026learned,
  title     = {Learning Compact Polygon Signatures via Weighted-Jaccard-Native
               Distillation for Large-Scale Similarity Search under Extreme Area Variation},
  author    = {Sampath, Ruban and Mallika Kankanamalage, Buddhi Ashan and Prasad, Sushil K.},
  booktitle = {IEEE International Conference on Big Data (BigData), BSD Workshop},
  year      = {2026},
  note      = {Under review}
}
```

Built on [ShapeToVec](https://doi.org/10.5281/zenodo.11522173), ShapeToVec+, and
[NMSLIB](https://github.com/nmslib/nmslib).
