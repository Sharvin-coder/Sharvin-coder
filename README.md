<h2 align="center">Hey, I'm Sharvin 👋</h2>

<p align="center">
<b>LLMs</b> · <b>evaluations & benchmarks</b> · <b>AI safety</b>
</p>

---

- 🧪 I like poking at language models until they break — then building evals and benchmarks that catch it properly.
- 🧮 Also into the math side of things: geometry, representations, and why models learn what they learn.
- 🎮 Occasionally I teach machines to play games better than I can.
- ☕ Fueled by way too much caffeine and "one more experiment" energy.

## 🛠️ Languages & Tools

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white" alt="Java">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white" alt="LaTeX">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/JAX-8A2BE2?style=for-the-badge&logoColor=white" alt="JAX">
  <img src="https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white" alt="PyTorch Lightning">
  <img src="https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge&logoColor=black" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/XGBoost-1A6FA8?style=for-the-badge&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/W%26B-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" alt="Weights & Biases">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sharvin-coder/Sharvin-coder/output/pacman-contribution-graph-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Sharvin-coder/Sharvin-coder/output/pacman-contribution-graph.svg">
  <img alt="pacman eating my contributions" src="https://raw.githubusercontent.com/Sharvin-coder/Sharvin-coder/output/pacman-contribution-graph.svg" width="100%">
</picture>

## 🚀 Things I've built

| Project | What it is |
|---|---|
| [terminal-velocity](https://github.com/anish-agr/terminal-velocity) | Bit-exact Rust reimplementation of a tower-defense game engine (99.87% frame-level parity with the reference implementation), league-based self-play training on H100, and a layered inference stack with hard real-time fallbacks guaranteeing a legal action under a 5-second wall-clock budget |
| [PersistBench](https://github.com/andrewzhao06/mash) | Benchmark suite for memory-augmented LLMs — quantifies cross-domain memory leakage and sycophantic retention to answer when long-term memories should be forgotten |

## 📖 Papers I keep coming back to

**Agents**
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the interleaved thought/action loop underneath basically every modern agent
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — self-improvement through episodic memory of your own failures
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) — skill libraries + automatic curriculum, no gradient updates required

**Evals & benchmarking**
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — how your choice of metric manufactures "emergence"
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) — what an eval looks like when it can't be memorized
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) — calibration as a first-class eval target

**Representations & theory**
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — why features share neurons, from first principles
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) — the weirdest generalization curve in deep learning
- [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) — models trained on different modalities converge to the same geometry

**Safety**
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566) — backdoors that survive RLHF
- [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390) — can a weak supervisor elicit a strong model's full capability?

---

<p align="center"><i>Always down to chat — sharvin.goyal3@gmail.com</i></p>
