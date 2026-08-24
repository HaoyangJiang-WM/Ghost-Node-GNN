<div align="center">

# Boundary-Consistent Graph Neural Networks for Topological Flux Prediction

[![TMLR](https://img.shields.io/badge/TMLR-2026-8A2BE2)](https://openreview.net/forum?id=31gTIfhoH0)
[![Python](https://img.shields.io/badge/Python-PyTorch-3776AB)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Data-LamaH--CE-2E8B57)](https://zenodo.org/records/5153305)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Paper](https://openreview.net/forum?id=31gTIfhoH0) · [LamaH-CE Dataset](https://zenodo.org/records/5153305) · [Code](#repository-structure) · [Citation](#citation)**

</div>

---

## Overview

Graph Neural Networks can suffer from large errors at **upstream boundary nodes** in directed fluid networks because the observed graph does not explicitly contain the external boundary context that drives transport.

We introduce **gTFP**, which augments each boundary node with a learned **ghost-node proxy**. The ghost representation is inferred from the local boundary state and downstream context, providing a data-driven boundary closure before information is propagated through the graph.

This repository contains the **river-data pipeline, ghost-node construction, GNN backbones, and training/evaluation code** used for the LamaH-CE experiments.

## Method

<p align="center">
  <a href="assets/model1.pdf">
    <img src="assets/model1.jpg" width="1000" alt="gTFP framework">
  </a>
</p>

<p align="center"><em>Overview of the gTFP framework. Click the figure to open the original PDF.</em></p>

The implementation follows three main steps:

1. **Build the river graph.** LamaH-CE gauges are connected using the directed river topology and physical edge attributes such as stream distance, elevation difference, and average slope.
2. **Add ghost nodes at upstream boundaries.** Each zero-in-degree boundary node receives a virtual upstream ghost node. Local boundary and downstream histories are concatenated to construct the ghost-node input.
3. **Forecast with a GNN backbone.** The learned ghost representation is inserted into the augmented graph and propagated through backbones such as **ResGCN, ResGAT, and GCNII**. Prediction loss is evaluated only on the original observed river nodes.

The dense variant additionally connects each river node to its downstream descendants using cumulative physical attributes, which is used by the current training script.

## River Example

The main river experiment uses a connected Danube subnetwork extracted from **LamaH-CE**:

- **358** observed river nodes
- **357** directed edges
- **209 boundary nodes** (58.4%)
- **149 interior nodes** (41.6%)
- **24-hour** input window
- **6-hour-ahead** discharge prediction

<p align="center">
  <a href="assets/river_topo1.pdf">
    <img src="assets/river_topo1.jpg" width="760" alt="LamaH-CE river topology">
  </a>
</p>

<p align="center"><em>LamaH-CE river topology used in the paper. Click the figure to open the original PDF.</em></p>

## Repository Structure

```text
Ghost-Node-GNN/
├── README.md
├── LICENSE
├── train_full.py              # Main river experiment entry point
│
├── dataset_ext_dense.py       # Main LamaH-CE loader + ghost nodes + dense downstream graph
├── dataset_ext1.py            # Ghost-augmented LamaH-CE loader (original directed graph)
│
├── models_ext1.py             # Ghost-aware GNN models (ResGCN / ResGAT / GCNII / others)
├── models_o.py                # Standard/baseline GNN implementations
│
├── functions_ext1.py          # Main model construction, training, validation, checkpoint utilities
├── functions_split_2.py       # Baseline / earlier experimental utilities
│
├── examples/
│   └── river_sample.py        # Minimal LamaH-CE loading example
│
└── assets/
    ├── model1.pdf             # Original method figure from the paper
    ├── model1.jpg             # README render of model1.pdf
    ├── river_topo1.pdf        # Original river-topology figure from the paper
    └── river_topo1.jpg        # README render of river_topo1.pdf
```

### Main execution path

```text
train_full.py
    └── functions_ext1.py
          ├── dataset_ext_dense.py
          └── models_ext1.py
```

### Main files

| File | Purpose |
|---|---|
| `train_full.py` | Defines the River experiment configuration, model backbone, graph direction, training years, and optimization settings. |
| `dataset_ext_dense.py` | Downloads/preprocesses LamaH-CE, constructs the Danube graph, identifies boundary nodes, creates ghost nodes, builds ghost features, and optionally densifies downstream connectivity. |
| `dataset_ext1.py` | Ghost-node dataset implementation on the original directed river graph. |
| `models_ext1.py` | Implements the ghost feature transform and GNN backbones including ResGCN, ResGAT, GCNII, GraphSAGE, and custom message passing. |
| `functions_ext1.py` | Connects datasets and models; handles edge weighting, training/validation loops, reproducibility, and checkpoint saving. |
| `models_o.py` | Standard GNN implementations used for baseline comparisons and earlier experiments. |
| `functions_split_2.py` | Earlier/baseline training utilities retained for experimental comparisons. |

## River Dataset

We use the hourly **LamaH-CE** hydrology dataset:

- **Dataset:** [LamaH-CE v1.0 (Zenodo)](https://zenodo.org/records/5153305)
- **Direct archive:** [1_LamaH-CE_daily_hourly.tar.gz](https://zenodo.org/record/5153305/files/1_LamaH-CE_daily_hourly.tar.gz)
- **Period used:** 2000–2017
- **Target:** discharge (`qobs`)
- **Meteorological inputs:** precipitation (`prec`), top-soil moisture (`volsw_123`), air temperature (`2m_temp`), and surface pressure (`surf_press`)

The dataset classes automatically download and preprocess the required LamaH-CE files on first use.

## Minimal Data Example

```python
from dataset_ext1 import LamaHDataset

dataset = LamaHDataset(
    root_dir="./LamaH-CE",
    years=[2016, 2017],
    root_gauge_id=399,
    window_size=24,
    lead_time=6,
    normalized=True,
)

sample = dataset[0]
print(sample.x.shape)
print(sample.y.shape)
print(sample.edge_index.shape)
print("real nodes:", int(sample.mask.sum()))
print("ghost nodes:", int(sample.ghost_mask.sum()))
```

## Training

The current experiment entry point is:

```bash
python train_full.py
```

Key settings are defined directly in the `hparams` dictionary in `train_full.py`, including the GNN architecture, edge orientation, number of layers, learning rate, training years, and forecast horizon.

## River Results

Main River results reported in the TMLR paper:

| Method | Overall MSE | Boundary MSE | Interior MSE |
|:--|--:|--:|--:|
| Avg. Base GNNs | 0.1218 | 0.1439 | 0.0905 |
| **gTFP Avg.** | **0.1114** | **0.1280** | **0.0885** |
| Implicit GNN w/ Ghost | **0.1084** | **0.1235** | **0.0867** |

Across the three GNN backbones, ghost-node modeling reduces River **overall MSE by 8.5%** and **boundary-node MSE by 11.0%** relative to the corresponding base GNN average.

## Citation

If you find this work useful, please cite:

```bibtex
@article{jiang2026boundary,
  title   = {Boundary-Consistent Graph Neural Networks for Topological Flux Prediction},
  author  = {Jiang, Haoyang and Qu, Bojian and Zhu, Xingquan and Tan, Jifu and He, Yi},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  month   = {August},
  url     = {https://openreview.net/forum?id=31gTIfhoH0}
}
```

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Transactions on Machine Learning Research · 2026</sub>
</div>
