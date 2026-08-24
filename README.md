# Boundary-Consistent Graph Neural Networks for Topological Flux Prediction

Official code for the TMLR paper **Boundary-Consistent Graph Neural Networks for Topological Flux Prediction**.

This repository studies the boundary-context closure deficit in GNN-based flux prediction. It introduces learned ghost-node proxies for missing upstream boundary context, together with implicit fixed-point and explicit inverse-operator solvers for boundary-consistent graph learning.

**Paper:** [Transactions on Machine Learning Research (TMLR), 2026](https://openreview.net/forum?id=31gTIfhoH0)

## River Example

The main river experiment uses a connected **358-node / 357-edge** Danube subnetwork extracted from LamaH-CE. The graph contains **209 boundary nodes** and **149 interior nodes**.

<p align="center">
  <img src="https://raw.githubusercontent.com/HaoyangJiang-WM/Ghost-Node-GNN/main/assets/river_topology.jpg" width="900" alt="LamaH-CE river topology with boundary and interior nodes">
</p>

## River Dataset

We use the hourly **LamaH-CE** hydrology dataset:

- **Dataset:** [LamaH-CE v1.0 on Zenodo](https://zenodo.org/records/5153305)
- **Direct hourly download:** [1_LamaH-CE_daily_hourly.tar.gz](https://zenodo.org/record/5153305/files/1_LamaH-CE_daily_hourly.tar.gz)
- **Time period used by the code:** 2000--2017
- **Input window:** 24 hours
- **Forecast horizon:** 6 hours
- **Node features:** discharge (`qobs`), precipitation (`prec`), top-soil moisture (`volsw_123`), air temperature (`2m_temp`), and surface pressure (`surf_press`)

`dataset_ext1.py` automatically downloads and preprocesses LamaH-CE. A minimal loading example is provided in [`examples/river_sample.py`](examples/river_sample.py).

```python
from dataset_ext1 import LamaHDataset

dataset = LamaHDataset(
    root_dir="./data",
    years=range(2000, 2018),
    root_gauge_id=399,
    window_size=24,
    lead_time=6,
    normalized=True,
)

sample = dataset[0]
print(sample.x.shape)
print(sample.y.shape)
print(sample.edge_index.shape)
```

## River Results

Mean test MSE across the three GNN backbones reported in the paper:

| Method | Overall MSE | Boundary MSE | Interior MSE |
|---|---:|---:|---:|
| Avg. Base GNNs | 0.1218 | 0.1439 | 0.0905 |
| gTFP Avg. | **0.1114** | **0.1280** | **0.0885** |
| Implicit GNN w/ Ghost | **0.1084** | **0.1235** | **0.0867** |

Averaged across the three GNN backbones, learned ghost nodes reduce the overall MSE by **8.5%** and the boundary-node MSE by **11.0%** on the river benchmark.

## Code

Core files:

- `dataset_ext1.py`: LamaH-CE download, preprocessing, graph construction, and ghost-node augmentation.
- `models_ext1.py`: model components.
- `functions_ext1.py`: training/evaluation utilities.
- `train_full.py`: training entry point.

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
