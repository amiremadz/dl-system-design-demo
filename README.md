# 🧠 Deep Learning System Design — 20-Min Demo Session

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amiremadz/dl-system-design-demo/blob/main/DL_System_Design_Demo.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Instructor:** Dr. Amir Masoud Zonoozi  
> **Format:** Live coding session — theory in markdown, implementation in Python  
> **Level:** Intermediate to Advanced  
> **Runtime:** ~20 minutes (demo mode) | GPU recommended

---

## 🎯 What This Session Covers

Most ML courses stop at "here's how to train a model." This session goes one level deeper: how do you design a system that is not just accurate, but **fast, memory-efficient, and production-ready**?

| # | Topic | Time |
|---|-------|------|
| 1 | The 4 Pillars of DL System Design | 0:00 – 4:00 |
| 2 | Data Pipeline Engineering | 4:00 – 8:00 |
| 3 | Model Architecture & Training Loop | 8:00 – 13:00 |
| 4 | Inference Optimization & Serving | 13:00 – 17:00 |
| 5 | Scalability Trade-offs & Wrap-up | 17:00 – 20:00 |

---

## 🚀 Quick Start

Click the badge above or use this direct link:

```
https://colab.research.google.com/github/amiremadz/dl-system-design-demo/blob/main/DL_System_Design_Demo.ipynb
```

**Recommended runtime:** Runtime → Change runtime type → **GPU (T4)**

---

## 📦 What's in This Repo

```
dl-system-design-demo/
├── DL_System_Design_Demo.ipynb   # Main notebook — theory + live code
├── Demo_Speaker_Script.md        # Timestamped speaker guide with talking points
└── README.md                     # This file
```

---

## 🏗️ Technical Content

### Data Pipeline
- Stratified dataset subsetting for fast demo iteration (`DEMO_MODE` flag)
- Optimized `DataLoader` configuration: `num_workers`, `pin_memory`, `persistent_workers`
- Augmentation strategy: training vs. validation transforms
- Pipeline throughput benchmarking

### Model Architecture
- `ResidualBlock` with projection shortcuts — built from scratch
- `TinyResNet` tuned for 32×32 CIFAR-scale images
- Kaiming weight initialization
- `torch.compile()` integration for kernel fusion on GPU

### Training Infrastructure
- Mixed precision (AMP) with `GradScaler`
- AdamW optimizer with decoupled weight decay
- Cosine annealing learning rate schedule
- Gradient clipping for training stability
- Label smoothing regularization
- Learning curve diagnostics (overfitting/underfitting detection)

### Inference Optimization
- Latency vs. throughput benchmarking across batch sizes
- Dynamic INT8 quantization (post-training, no retraining needed)
- Accuracy validation pre/post quantization
- Production inference server pattern with latency tracking

### System Design Analysis
- Accuracy–Latency–Parameters Pareto frontier visualization
- Trade-off framework across model families (MobileNet, EfficientNet, ResNet, ViT)

---

## ⚙️ Configuration

The notebook has a `DEMO_MODE` flag at the top of the data pipeline cell:

```python
DEMO_MODE    = True    # 5k train / 1k val — full run in ~45 sec on GPU
DEMO_MODE    = False   # Full CIFAR-10 (50k train / 10k val)
```

Flip to `False` for a complete training run with full accuracy.

---

## 🔧 Requirements

All dependencies are pre-installed in Google Colab. For local execution:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 📖 Further Reading

- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2016  
- **Paper:** [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019  
- **Book:** *Designing Machine Learning Systems* — Chip Huyen  
- **Frameworks:** [TorchServe](https://pytorch.org/serve/), [NVIDIA Triton](https://developer.nvidia.com/triton-inference-server), [BentoML](https://www.bentoml.com/)

---

## 📄 License

MIT License — free to use and adapt with attribution.
