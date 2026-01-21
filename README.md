
# IndigoRL - Pokémon Yellow Deep Reinforcement Learning 🧠🎮

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
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/PyBoy-2.0-green" />
  <img src="https://img.shields.io/badge/RL-Recurrent%20PPO-orange" />
  <img src="https://img.shields.io/github/stars/OutFerz/indigoRL?style=flat" />
</p>

<p align="center">
  <strong>Neuro-Symbolic Vision + RAM Reinforcement Learning Agent</strong><br>
  Autonomous completion of Pokémon Yellow using long-term memory.
</p>

<!-- ===================================================== -->
<!-- DEMO GIF -->
<!-- ===================================================== -->

<p align="center">
  <img src="assets/demo.gif" width="600" />
</p>

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Agent Architecture](#agent-architecture)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [Disclaimer](#disclaimer)

---

## 🎯 Project Overview <a id="project-overview"></a>

**IndigoRL** is an autonomous Artificial Intelligence agent designed to complete
*Pokémon Yellow* using **Deep Reinforcement Learning**.

Unlike generic agents that randomly press buttons, IndigoRL implements a
**Neuro-Symbolic Architecture** combining:

- 🖼️ Computer Vision (CNN over game frames)
- 🧠 Direct RAM memory inspection (symbolic state)
- 🔁 Long-term memory via **LSTM (Recurrent PPO)**

This allows the agent to reason about **story progression, battles, and exploration**
in an extremely sparse, long-horizon RPG environment.

---

## ✨ Key Features <a id="key-features"></a>

### 🧠 LSTM Brain (Long-Term Memory)
- Uses `RecurrentPPO` (PPO + LSTM) to retain past information.
- Enables maze navigation, backtracking, and objective persistence.

### 🧩 Neuro-Symbolic Reward System
- **Story Progress**
  - Reads *event flags* directly from game RAM.
  - Rewards medals, key items, and narrative milestones.
- **Battle Awareness**
  - Reads enemy HP and battle results from memory.
  - Learns combat strategies instead of brute force.
- **Exploration**
  - Rewards new Map IDs.
  - Penalizes stagnation and looping behavior.

### ⚡ Extreme Efficiency
- **State Loading**
  - Automatically skips Oak’s intro using a clean save-state.
  - ~20% reduction in compute per episode.
- **Parallel Training**
  - Multiple emulator instances running simultaneously.

### 🎮 Emulator Compatibility
- Fully compatible with **PyBoy 2.0+**.

---

## 🛠️ Technology Stack <a id="technology-stack"></a>

| Component | Technology |
|---------|-----------|
| Language | Python 3.10+ |
| RL | Stable-Baselines3 Contrib (Recurrent PPO) |
| Emulator | PyBoy 2.0+ |
| Vision | OpenCV, NumPy |
| Logging | TensorBoard |

---

## 🚀 Installation & Setup <a id="installation--setup"></a>

### Prerequisites
- Python 3.10+ (Conda recommended)
- Pokémon Yellow ROM (legally owned)

### Setup

```bash
git clone https://github.com/OutFerz/indigoRL.git
cd indigoRL
conda create -n poke-rl python=3.10
conda activate poke-rl
pip install -r requirements.txt
```

### ROM
Place your ROM at:

```
roms/PokemonYellow.gb
```

---

## 🕹️ Usage <a id="usage"></a>

### 1️⃣ Generate Initial Save State (Optional)

```bash
python record_state.py
```

Play the intro manually and close the window once you have control of Ash.

---

### 2️⃣ Train the Agent (Recurrent PPO + LSTM)

```bash
python train_lstm.py
```

Models are saved in:

```
experiments/poke_lstm_v1/
```

---

### 3️⃣ Watch the Agent Play

```bash
python watch_continuous.py
```

- Real-time 60 FPS playback
- Live action and memory overlay

---

## 🧠 Agent Architecture <a id="agent-architecture"></a>

**Policy:** Multi-Input Recurrent Policy

- **Visual Input**
  - CNN over resized game frames
- **RAM Input**
  - Player X/Y
  - HP, Level
  - Map ID
- **Memory Core**
  - LSTM (256 units)
- **Output**
  - Discrete GameBoy actions

---

## 📂 Project Structure <a id="project-structure"></a>

```
indigoRL/
├── src/
│   └── environment/
│       └── pokemon_env.py
├── experiments/
├── roms/
├── states/
├── train_lstm.py
├── watch_continuous.py
├── record_state.py
└── README.md
```

---

## 🤝 Credits <a id="credits"></a>

- PyBoy Emulator
- Stable-Baselines3 Contrib
- pret/pokeyellow disassembly project

---

## 📜 Disclaimer <a id="disclaimer"></a>

This project is for **research and educational purposes only**.  
You must legally own a copy of *Pokémon Yellow* to use the ROM.

---

⭐ If you find this project interesting, consider giving it a star!
