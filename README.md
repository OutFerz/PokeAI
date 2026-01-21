# 🧠🎮 indigoRL — Autonomous Pokémon Yellow Reinforcement Learning Agent

<!-- ===================================================== -->
<!-- BANNER IMAGE -->
<!-- Recommended size: 1200x400 -->
<!-- Place at: assets/banner.png -->
<!-- ===================================================== -->

<p align="center">
  <img src="assets/banner.png" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-success" />
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
  <img src="https://img.shields.io/badge/RL-PPO-orange" />
  <img src="https://img.shields.io/badge/Emulator-PyBoy-purple" />
  <img src="https://img.shields.io/github/stars/OutFerz/PokeAI?style=flat" />
</p>

<p align="center">
  <strong>Hybrid Vision + RAM Reinforcement Learning Agent</strong><br>
  Trained using PPO to solve a sparse, long-horizon RPG environment.
</p>

<!-- ===================================================== -->
<!-- DEMO GIF -->
<!-- Recommended: 10–15 seconds, <5MB -->
<!-- Record: exploration, map transitions, battles -->
<!-- Place at: assets/demo.gif -->
<!-- ===================================================== -->

<p align="center">
  <img src="assets/demo.gif" width="600" />
</p>

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Technical Description](#technical-description)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation-setup)
- [Workflow & Execution](#workflow-execution)
- [Agent Architecture](#agent-architecture)
- [Project Structure](#project-structure)
- [Hardware & Scalability](#hardware-scalability)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)

---

## 🎯 Project Overview <a id="project-overview"></a>

**IndigoRL** is a Deep Reinforcement Learning research project focused on solving **long-horizon RPG environments** using **Pokémon Yellow** as a benchmark.

The game presents:
- Extremely sparse rewards
- Large state space
- Long-term dependencies
- Partial observability from pixels alone

To overcome these challenges, IndigoRL combines **visual perception** with **explicit symbolic state extraction from emulator RAM**, allowing the agent to both *see* and *understand* the game world.

---

## 🧩 Technical Description <a id="technical-description"></a>

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Emulator:** PyBoy (headless during training)
- **Observation Space:**
  - CNN-processed screen frames
  - Structured RAM-based state vectors
- **Reward Design:**
  - Dense exploration rewards
  - Event-based progress signals
  - Implicit stagnation penalties

This neuro-symbolic approach significantly improves sample efficiency and training stability.

---

## ✨ Key Features <a id="key-features"></a>

- ⚡ **Accelerated Emulation** — 1000+ FPS headless training
- 👁️ **Hybrid Observations** — Vision + RAM decoding
- 🗺️ **Dense Exploration Rewards** — Unique `(x, y)` tracking
- 🎥 **Streamer-Ready** — Train in background, watch at 60 FPS
- ⚙️ **Hardware-Aware & Scalable** — CPU usage configurable

---

## 🛠️ Technology Stack <a id="technology-stack"></a>

| Component | Technology |
|---------|-----------|
| Language | Python 3.11 |
| RL | Stable-Baselines3 (PPO) |
| Emulator | PyBoy |
| Vision | OpenCV, NumPy |
| Logging | TensorBoard |

---

## 🚀 Installation & Setup <a id="installation-setup"></a>

### Prerequisites
- Python 3.11 (Conda recommended)
- Pokémon Yellow ROM  
  Must be named `PokemonYellow.gb` and placed in `roms/`

### Setup Steps

```bash
git clone https://github.com/OutFerz/indigoRL.git
cd indigoRL
conda create -n indigoRL python=3.11
conda activate indigoRL
pip install gymnasium pyboy shimmy stable-baselines3[extra] opencv-python torch-directml
```

### Initial Save State (Skip Intro)

```bash
python src/utils/create_initial_state.py
```

> Play manually until you gain control in Ash’s room, then close the window.

---

## 🏃 Workflow & Execution <a id="workflow-execution"></a>

### 🧠 Training
```bash
python train.py
```

- Headless, high-speed PPO training
- Automatic checkpoints
- Safe interrupt via **Ctrl + C**

### 👀 Visualization
```bash
python watch_continuous.py
```

- 60 FPS real-time playback
- Hot-reloads improved models

### 📊 Monitoring
```bash
tensorboard --logdir experiments/poke_ppo_v1/logs
```

---

## 🧠 Agent Architecture <a id="agent-architecture"></a>

**Action Space:**  
`[DOWN, LEFT, RIGHT, UP, A, B]`  
`START` and `SELECT` disabled to reduce noise.

**Reward Function:**
```
R_t = R_exploration + R_events
```

---

## 📂 Project Structure <a id="project-structure"></a>

```
indigoRL/
├── config/
├── experiments/
├── roms/
├── states/
├── src/
│   ├── environment/
│   ├── utils/
├── train.py
├── watch_continuous.py
└── README.md
```

---

## 💻 Hardware & Scalability <a id="hardware-scalability"></a>

Default settings prioritize compatibility with consumer hardware.  
Training parallelism can be scaled by editing `train.py` or setting:

```bash
export OMP_NUM_THREADS=8
```

---

## 🔮 Roadmap <a id="roadmap"></a>

- [ ] Integrate **HippoTorch / S4** for long-term memory
- [ ] Add **Vision-Language Model (VLM)** for on-screen dialogue understanding

---

## 📜 Disclaimer <a id="disclaimer"></a>

This project is for **research and educational purposes only**.  
You must legally own a copy of Pokémon Yellow to use the ROM.

---

⭐ If you find this project interesting, consider giving it a star!