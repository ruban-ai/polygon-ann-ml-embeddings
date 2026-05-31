HPML Term Project: Learned Quadtree Retrieval
==============================================

This folder contains the notebooks used for the HPML project on learned
retrieval for polygon/quadtree data. The baseline uses nmslib HNSW directly
on original quadtree vectors with WeightedJaccard. The proposed methods learn
compact neural embeddings for candidate generation and then use exact
Weighted-Jaccard reranking on a small candidate set.


Platform Used
-------------

The reported experiments were run on:

- Host: dgxa100.cs.utsarr.edu
- OS: Ubuntu Linux, x86_64
- CPU: 2 x AMD EPYC 7742 64-Core Processor, 256 hardware threads total
- Python: 3.13.11 in the current environment
- GPU: CUDA-capable NVIDIA GPU environment. The notebooks use cuda:0 for most
  learned MLP/two-stage experiments. The Neural MinHash notebook also contains
  cells that use cuda:2 and DataParallel over multiple CUDA devices.
- CPU baseline: nmslib HNSW WeightedJaccard, 32 query threads.
- Learned methods: CPU nmslib HNSW candidate generation over learned
  embeddings plus GPU batched exact original Weighted-Jaccard reranking.

If a CUDA GPU is not available, the pure baseline notebooks can still run, but
the GPU reranking cells should be changed to CPU/dense reranking or skipped.
Full-set GPU reranking expects enough GPU memory to hold the original corpus
quadtree matrix or a smaller rerank batch size.


Python Requirements
-------------------

Install the packages in requirements.txt:

    pip install -r requirements.txt

The project also includes requiremnt.txt with the same contents because the
file was requested under that spelling. The conventional file to use is
requirements.txt.

Important package notes:

- torch should be installed with the CUDA build that matches the machine.
- nmslib must support the WeightedJaccard space used by the baseline.

  DGX (original build):
    /raid/ruban/nmslib_weighted

  Orion (copy from DGX):
    /mnt/data1/ruban/nmslib_weighted
    Script: term_project/scripts/copy_nmslib_weighted_from_dgx.sh

  After copy: cd /mnt/data1/ruban/nmslib_weighted && pip install -e .

  If running on a new machine, install/build that nmslib variant before running
  the notebooks.


Dataset Locations
-----------------

The notebooks assume the following local dataset layout:

- Raw polygon TSV:

    /raid/ruban/data/parks.tsv

- 10k quadtree encoding files (Orion):

    /mnt/data1/shapeSimilarity/encodings/pk-real10k0.002

  Legacy DGX path: /raid/ruban/encodings/pk-real10k0.002

- 10k ground truth (Orion):

    /mnt/data1/shapeSimilarity/warehouse/pk-query-10k

  Legacy DGX path: /raid/ruban/groundtruth/pk-query-10k

- Full-set ground truth (DGX):

    /raid/ruban/groundtruth/pk-query-187019

- Full quadtree cache expected by the notebooks:

    /tmp/qtree_vectors_full.npy

The 00_cache_data.ipynb notebook creates or verifies the main /tmp caches used
by the rest of the project.


Generated Cache and Result Files
--------------------------------

The notebooks read and write intermediate artifacts under /tmp. The most
important cache files are:

- /tmp/qtree_vectors_full.npy
- /tmp/qt_10k.npy
- /tmp/gt_lookup_full.pkl
- /tmp/gt_lookup_10k.pkl

Training notebooks also save model checkpoints such as:

- /tmp/best_compressor_v1_clean.pt
- /tmp/best_compressor_full_fixed.pt
- /tmp/best_compressor_hardneg_wjdistill_10k.pt
- /tmp/best_compressor_hardneg_wjdistill_full.pt
- /tmp/best_minhash_10k.pt
- /tmp/best_minhash_full.pt

Result summary files are saved as pickle files under /tmp, including:

- /tmp/results_baseline.pkl
- /tmp/results_mlp_cosine.pkl
- /tmp/results_minhash.pkl
- /tmp/results_minhash_rerank.pkl
- /tmp/results_two_stage_mlp_wj.pkl
- /tmp/results_hardneg_wjdistill.pkl
- /tmp/results_adaptive_k.pkl
- /tmp/results_listwise_wjdistill.pkl
- /tmp/results_raw_pointnet_wjdistill.pkl


Notebook Run Order
------------------

Run the notebooks in numerical order:

0. 00_cache_data.ipynb
   Verifies/builds the cached quadtree and ground-truth files under /tmp.

1. 01_baseline.ipynb
   Runs the traditional HNSW WeightedJaccard baseline on the original quadtree
   vectors.

2. 02_mlp_cosine.ipynb
   Evaluates the first learned MLP embedding model with cosine HNSW search.
   This establishes why no-rerank cosine embeddings alone are not enough.

3. 03_neural_minhash.ipynb
   Trains/evaluates Neural MinHash style embeddings and includes exact
   original-WJ reranking experiments.

4. 04_two_stage_mlp_wj_rerank.ipynb
   Main two-stage system: learned MLP candidate generation plus exact
   Weighted-Jaccard reranking.

5. 05_mlp_wj_distillation.ipynb
   Random-negative Weighted-Jaccard distillation ablation.

6. 06_mlp_hard_negative_distillation.ipynb
   Hard-negative Weighted-Jaccard distillation. This is the strongest learned
   candidate generator in the current project.

7. 07_adaptive_k_rerank.ipynb
   Adaptive candidate-depth experiment from cached max-K candidate lists.

8. 08_listwise_wj_distillation.ipynb
   Listwise Weighted-Jaccard distillation ablation.

9. 09_raw_polygon_pointnet_wj_distill.ipynb
   Raw-polygon PointNet ablation using /raid/ruban/data/parks.tsv. This tests
   whether raw coordinates can replace quadtree features directly.

10. 10_results.ipynb
    Final results notebook. Run this after the experiment notebooks have
    produced their /tmp/results_*.pkl files.

15. 15_mlp_wj_native.ipynb
    Advisor experiment: WJ-simplex MLP outputs, WJ triplet training, HNSW
    WeightedJaccard index, optional GPU raw-WJ rerank. Saves
    /tmp/best_compressor_wj_native_{10k,full}.pt and
    /tmp/results_mlp_wj_native.pkl.

16. 16_mlp_intersection_min.ipynb
    Clean build per Dr. Prasad: train with intersection only sum(min), no ratio
    in loss; from scratch (no best_model). Stage-1 search by intersection on
    512-D; optional raw WJ ratio rerank for GT. Saves
    /tmp/best_compressor_intersection_min_10k.pt and
    /tmp/results_mlp_intersection_min.pkl.


Recommended Reproduction Flow
-----------------------------

For a quick sanity check, run:

1. 00_cache_data.ipynb
2. 01_baseline.ipynb on the 10k split
3. 04_two_stage_mlp_wj_rerank.ipynb on the 10k split
4. 06_mlp_hard_negative_distillation.ipynb on the 10k split
5. 10_results.ipynb

After the 10k results look correct, switch the dataset setting in the relevant
notebooks to full and rerun the full-set experiments. Full-set baseline and
training cells can take much longer and need substantially more memory.


Important Notes
---------------

- The ground truth is aligned to original WeightedJaccard on quadtree vectors.
  This is why the best learned systems still rerank candidates using exact
  original-WJ scores.
- Candidate K controls the recall ceiling. For example, reranking K=500
  candidates cannot recover true neighbors that were not present in those 500
  candidates.
- The baseline and proposed systems are not identical hardware configurations:
  the baseline is CPU-only nmslib WeightedJaccard, while the proposed two-stage
  systems are CPU candidate search plus GPU batched reranking. Present the
  comparison as a hybrid learned/hardware-aware retrieval system.
- Jupyter/VS Code may keep an old notebook buffer open. If VS Code says the
  notebook file is newer on disk, reload from disk instead of overwriting, or
  you may erase recently added cells.
