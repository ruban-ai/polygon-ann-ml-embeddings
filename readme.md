# ML-Based Embeddings for Polygon Similarity Search

## Overview
This project explores machine learning approaches for **polygon similarity search** on large-scale geospatial data.

We implement and compare two fundamentally different methods:

1. **MLP (Method 1)** — Representation-aligned embeddings using spatial cell encodings  
2. **PolyMP (Method 2)** — Graph Neural Network (GNN) based embeddings using polygon structure  

The goal is to evaluate whether **representation alignment or model complexity** plays a more important role in retrieval performance.

---

## Dataset
- **Source:** OpenStreetMap park polygons (UK region)
- **Size:** 50,000 polygons  
  - 40,000 → Database  
  - 10,000 → Query set  

- **Representation (Method 1):**
  - Quad-tree spatial cell encoding (~18K dimensions)

- **Representation (Method 2):**
  - Polygon graphs with rotation-invariant geometric features

- **Ground Truth:**
  - Method 1: Jaccard similarity over spatial cell encodings  
  - Method 2: Shape similarity via rotation-invariant boundary descriptors  

> Note: Dataset and ground truth files are not included due to size.

---

## Methods

### Method 1: MLP (Representation-Aligned)
- Input: Binary spatial cell vectors  
- Model: Multi-Layer Perceptron (MLP)  
- Loss: NT-Xent (contrastive learning)  
- Retrieval: HNSW (cosine similarity)  

---

### Method 2: PolyMP (GNN-Based)
- Input: Polygon as graph (nodes = vertices)  
- Features: Rotation-invariant geometric features  
- Model: 4-layer GNN with residual connections  
- Training:
  - Self-supervised pretraining  
  - Triplet loss fine-tuning  
- Retrieval: Cosine similarity  

---

## Results

| Method | Recall@10 | Recall@50 | Recall@100 | Recall@500 |
|--------|----------|----------|-----------|-----------|
| MLP (Method 1) | **7.94%** | 20.27% | 28.90% | 59.51% |
| PolyMP (Method 2) | 2.8% | 4.0% | 5.0% | 8.3% |

---

## Key Insight

> **Representation alignment with the ground truth metric is more important than model complexity.**

- The MLP significantly outperforms the GNN because it directly models the **cell-based Jaccard similarity**.
- The GNN captures geometric structure, but **shape similarity does not align with spatial overlap**, leading to poor retrieval performance.
