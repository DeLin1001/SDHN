# SDHN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation code for the IROS 2025 submission "SDHN: Skewness-Driven Hypergraph Networks for Enhanced Localized Multi-Robot Coordination".

## Abstract
Multi-Agent Reinforcement Learning is widely
used for multi-robot coordination, where simple graphs typically model pairwise interactions. However, such representations fail to capture higher-order collaborations, limiting
effectiveness in complex tasks. While hypergraph-based approaches enhance cooperation, existing methods often generate arbitrary hypergraph structures and lack adaptability to
environmental uncertainties. To address these challenges, we
propose the Skewness-Driven Hypergraph Network (SDHN),
which employs stochastic Bernoulli hyperedges to explicitly
model higher-order multi-robot interactions. By introducing
a skewness loss, SDHN promotes an efficient structure with
Small-Hyperedge Dominant Hypergraph, allowing robots to
prioritize localized synchronization while still adhering to the
overall information, similar to human coordination. Extensive
experiments on Moving Agents in Formation and Robotic
Warehouse tasks validate SDHN’s effectiveness, demonstrating superior performance over state-of-the-art baselines.



## Quick Start
### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Running SDHN
### Run on grid_maif environment
python src/main.py --config=mappo_hgcn --env-config=grid_maif
### Run on rware environment 
python src/main.py --config=mappo_hgcn --env-config=rware