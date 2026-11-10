# SA-LCDGNet

**State-Aware Lag-Conditioned Dynamic Graph Network for Traffic Flow Forecasting**

This repository provides the implementation of **SA-LCDGNet** for short-term traffic flow forecasting. The implementation is built on **[BasicTS v0.5.8](https://github.com/GestaltCogTeam/BasicTS/tree/v0.5.8)** and includes the **full model**.

**Current release:** training and evaluation configurations are provided for **PEMS04** and **PEMS08**. Although the paper also evaluates PEMS03 and PEMS07, their configurations are not included in this release.

## Overview

SA-LCDGNet is designed to model traffic-dependent spatial relationships by incorporating traffic-state information and node-specific historical lag dependencies into dynamic graph construction. Its main components are:

- **Traffic State Transition Module (TSTM):** uses first- and second-order temporal differences and a learnable state-transition prior to derive node-level traffic-state representations.
- **Lag-Conditioned Dynamic Graph (LCDG):** learns node-specific weights over candidate historical lag steps and uses the resulting lag summaries to construct context-dependent dynamic node relations.
- **Memory Enhancement:** incorporates learned, relatively stable relation patterns to complement the input-dependent dynamic graph.
- **Multi-Graph Spatial Propagation:** combines predefined road-network graphs, an adaptive graph, and the lag-conditioned dynamic graph.

The forecasting backbone uses gated dilated causal convolutions. The model receives 12 historical time steps and predicts the next 12 time steps (one hour each at a five-minute sampling interval).

## Repository Structure

```text
SA-LCDGNet/
├── model/
│   ├── __init__.py
│   └── sa_lcdgnet.py
├── configs/
│   ├── PEMS04.py
│   └── PEMS08.py
├── basicts/                  # BasicTS v0.5.8 framework code
├── experiments/              # BasicTS training and evaluation entry points
├── scripts/
│   └── data_preparation/     # PEMS preprocessing scripts
├── requirements.txt
├── LICENSE
└── README.md
```

The `datasets/` directory and trained checkpoints are **not distributed** with this repository. Prepare the data locally as described below.

## Requirements

Use a Python and PyTorch environment compatible with the included BasicTS v0.5.8 framework and your CUDA / GPU installation. From the repository root, install the dependencies listed in the provided file:

```bash
pip install -r requirements.txt
```

If PyTorch is not covered by that file, install the appropriate PyTorch build for your machine separately. Run the commands below from the **repository root**, not from `configs/` or `experiments/`.

## Data Preparation

The released configurations use two public traffic-flow datasets:

| Dataset | Sensors | Sampling interval | History | Forecast horizon |
| --- | ---: | --- | ---: | ---: |
| PEMS04 | 307 | 5 minutes | 12 steps | 12 steps |
| PEMS08 | 170 | 5 minutes | 12 steps | 12 steps |

The data are split chronologically into training, validation, and test sets using a **6:2:2** ratio.

Obtain the corresponding datasets from the [BasicTS dataset resources](https://github.com/GestaltCogTeam/BasicTS/tree/v0.5.8/datasets). After preprocessing, the local directory should contain:

```text
datasets/
├── PEMS04/
│   ├── data.dat
│   ├── adj_mx.pkl
│   └── desc.json
└── PEMS08/
    ├── data.dat
    ├── adj_mx.pkl
    └── desc.json
```

Both configurations load `adj_mx.pkl` using the `doubletransition` adjacency setting. **Do not omit this file**, even if the preprocessed time-series data are already available.

### Preparing data from raw files (optional)

If using the original BasicTS v0.5.8 preprocessing scripts, place the raw PEMS files under `datasets/raw_data/PEMS04/` and `datasets/raw_data/PEMS08/` as expected by the respective scripts, and run:

```bash
python scripts/data_preparation/PEMS04/generate_training_data.py
python scripts/data_preparation/PEMS08/generate_training_data.py
```

The original preprocessing scripts expect `PEMS04.npz` / `PEMS08.npz`, together with the required graph source files (or source information needed to generate the adjacency matrices). They create the processed files shown above. The local data directories should be created before training.

## Training

Train the full SA-LCDGNet model on PEMS04:

```bash
python experiments/train.py -c configs/PEMS04.py -g 0
```

Train on PEMS08:

```bash
python experiments/train.py -c configs/PEMS08.py -g 0
```

Here, `-g 0` selects GPU index 0. Both configurations use **100 training epochs** and the Adam optimizer. Dataset-specific settings are kept in `configs/PEMS04.py` and `configs/PEMS08.py`.

Checkpoints are saved under `checkpoints/SALCDGNet/` in dataset-specific subdirectories, subject to the naming conventions used by BasicTS.


## Citation

If you use this implementation in your research, please cite the paper:

*State-Aware Lag-Conditioned Dynamic Graph Network for Traffic Flow Forecasting.*

The full publication details and BibTeX entry will be added when available.

