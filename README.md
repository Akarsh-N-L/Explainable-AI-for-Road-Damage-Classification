# Road Crack Classification and Explainability

MCM Practicum — Dublin City University, School of Computing (2025–2026)

**Student:** Balaji Segu Krishnaiah, Akarsh Nichunnaduttlu Lakshmisha

---

## Project Overview

This project investigates deep learning approaches to road crack classification using the RDD2022 benchmark dataset.  
The focus is on both classification accuracy and model explainability (XAI), providing transparency into how models make decisions across different countries and crack types.

---

## Phase 1 — CNN Classification and XAI Analysis (Complete)

Phase 1 established the full CNN-based classification pipeline using transfer learning backbones (ResNet50, VGG16, Swin-Tiny) with frozen and fine-tuned heads.

**What was done:**
- Patch extraction and stratified train/val/test splits across 5 countries
- Feature extraction and linear head training 
- Full fine-tuning of best backbone with country-specific specialisation (Czech Republic, India)
- XAI evaluation using GradCAM, LIME, and DeepLIFT across 2,500 samples
- ML verification: calibration, leakage checks, stability, test hardness analysis 
- Thesis result compilation with literature comparison 

**Key results:**
- Best backbone: ResNet50 with fine-tuned head
- XAI method agreement analysis across 3 methods per country and crack class
- Country-specialised models showed improved per-country performance

**Code structure (`src/cnn/`):**
```
src/cnn/
├── dataset.py / models.py / metrics.py / seed_utils.py   ← core modules
├── requirements.txt / Dockerfile
├── scripts/    ← pipeline scripts
├── scratch/    ← analysis utilities
└── results/    ← all CSVs, logs, figures, verification outputs
```

---

## Phase 2 — GNN-Based Approach (In Progress)

Phase 2 extends the work by modelling road crack patches as graphs to capture spatial relationships between crack regions. Each 224x224 patch is converted to a graph three different ways (fixed grid cells, SLIC superpixels, crack skeleton), and each representation is fed into three GNN architectures (GIN, GAT, APPNP) under identical shared hyperparameters, for a 9-combination comparison against the CNN baseline.

**What was done:**
- Image-to-graph conversion via three extractors: `patch` (grid cells), `slic` (superpixels + region-adjacency), `skeleton` (crack skeletonisation)
- GIN / GAT / APPNP classifiers sharing identical depth, width, dropout, LR, epochs, and seeds across all 9 (method × model) combinations
- Multi-seed train/eval protocol with paired significance testing across combinations

**Code structure (`src/gnn/`):**
```
src/gnn/
├── config.py / dataset.py / model.py        ← core modules
├── train.py / run_comparison.py             ← single-combo training / full matrix comparison
├── src/graphs/                              ← image-to-graph extractors (patch / slic / skeleton)
├── src/metrics.py / log_utils.py / seed_utils.py
└── scripts/                                 ← patch extraction, splits, graph visualisation
```

**Planned work:**
- Apply graph-level XAI (GNNExplainer, attention weights)
- Evaluate whether GNNs generalise better across countries

---

## Repository Structure

```
src/
├── cnn/    ← Phase 1: CNN pipeline, scripts, results
└── gnn/    ← Phase 2: GNN experiments (in progress)
docs/
├── proposal/           ← project proposal
├── documentation/      ← ongoing documentation
└── Literature_Review_Road_Crack_Classification.pdf
```

---

