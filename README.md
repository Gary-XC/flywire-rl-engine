# FlyWire-RL: The Connectome Controller

> **Can a biological brain beat Super Mario Bros?**  
> `FlyWire-RL` replaces conventional black-box neural networks (MLPs, CNNs) in Reinforcement Learning with a Graph Neural Network (GNN) constrained directly to the 1:1 synaptic wiring diagram of the *Drosophila melanogaster* (fruit fly) connectome.

---

## Overview

Modern Reinforcement Learning agents rely on dense, uniform artificial networks where internal decision-making is functionally opaque. In contrast, biological brains operate on highly specialized, sparse, and evolutionarily tuned circuit topologies.

`FlyWire-RL` constructs an embodied controller whose computation graph strictly follows actual biological synapses:
- Visual game states project onto simulated **optic lobe** photoreceptors.
- Signals propagate across biological pathways through the **central brain** and **mushroom body**.
- Motor commands are decoded directly from **descending motor neurons** to trigger NES controller buttons.

Every activation pathway in the network corresponds to real neurons, enabling full, mechanistic interpretability of every decision the agent makes.

---

## Architecture & Biological Mapping

```
[ Game Frame / Depth ]
         │
         ▼
 1. Optic Lobes (Sensory Inputs)
         │
         ▼
 2. Central Complex & Mushroom Body (Associative Processing & Plasticity)
         │
         ▼
 3. Descending Neurons (Ventral Nerve Cord / Motor Primitives)
         │
         ▼
[ NES Controller Actions (Run, Jump, Steer) ]
```

### 1. Sensory Interface (Vision)
Visual frames from `gym-super-mario-bros` are downsampled into spatial grid signals and mapped directly to photoreceptor input clusters in the fly's optic lobes (lobula and medulla circuits).

### 2. Connectome Policy Graph
Using open synaptic datasets, the policy is structured as an adjacency-constrained sparse graph network:
- **Nodes**: Biological neurons.
- **Edges**: Verified synaptic connections (restricted to biological coordinates).
- **Signal Dynamics**: Recurrent propagation mimicking forward and feedback neuronal circuits.

### 3. Motor Decoding (Actions)
Specific clusters of descending motor neurons responsible for directional locomotion and flight-initiation are monitored. Their summed firing rates determine discrete NES controller inputs (e.g., `Right`, `A` [Jump], `B` [Sprint]).

### 4. Biologically Grounded Plasticity
Rather than treating all network weights uniformly:
- **Structural Constancy**: Structural connectivity is fixed to biological ground truth.
- **Selective Plasticity**: Learning (via PPO/RL updates) is localized primarily to known plastic sites (e.g., Kenyon cell–mushroom body output neuron synapses driven by simulated dopamine pathways).

---

## Tech Stack

| Component | Library | Purpose |
| :--- | :--- | :--- |
| **RL Environment** | `gym-super-mario-bros`, `nes-py` | NES emulation, RAM/frame state extraction, and discrete control execution |
| **Connectome Extraction** | `neuprint-python`, `navis` | Querying synaptic adjacency matrices, cell morphology, and connectomic subgraphs |
| **3D Mesh & Activation Viz** | `pyvista`, `vedo` | Real-time 3D volumetric rendering of brain morphology and active neural signal tracing |

---

## Quickstart

### Prerequisites
- Python 3.13+ (for nes-py and gym-super-mario-bros to work, or Python 3.11.X if stable-baselines3 is used)

### Installation

```bash
# Clone repository
git clone https://github.com/Gary-XC/flywire-rl-engine.git
cd flywire-rl-engine

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure (Target)

```text
FlyWire-RL/
├── data/               # Cached adjacency matrices and neuron subgraphs
├── env/                # Mario environment wrappers and sensory downsamplers
├── model/              # Biological GNN policy and synaptic weight layers
├── visualizer/         # 3D brain mesh rendering and dynamic signal tracer
├── train.py            # Training loops and plasticity updates
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Fetch optic lobe to descending neuron subgraphs via NeuPrint.
- [ ] Implement sensory projection wrapper for 8-bit NES video streams.
- [ ] Build baseline sparse forward-pass GNN in PyTorch / PyTorch Geometric.
- [ ] Connect descending neuron outputs to discrete NES input actions.
- [ ] Integrate 3D visualization using `pyvista`/`vedo` to trace real-time signal propagation during gameplay.
- [ ] Implement mushroom-body-restricted plasticity reward mechanisms.

---

## License

Distributed under the [MIT License](LICENSE).
