# Bicameral Manifold GMoE (BM-GMoE)

## The Lifelong Dual-Manifold Architecture

Official implementation of the **Fractal Bottleneck Hypothesis** and the **Dual-Manifold Lifelong Learning Framework**.

### 🌌 Overview

BM-GMoE is a next-generation architecture designed to solve the "static parametric intelligence" crisis. It moves beyond the traditional train-then-freeze paradigm by implementing a dynamic switching process between two topological regimes:

1.  **The Logic Manifold ($M_L$):** A sparse, fractal attractor optimized for rigorous causal reasoning, formal logic, and epistemic verification.
2.  **The Creative Manifold ($M_C$):** A high-entropy, isotropic space optimized for stylistic generation, fluid association, and creative synthesis.

### 🏗️ Core Architecture

- **Geometric Router:** Real-time Intrinsic Dimension (ID) estimation to route tokens to the appropriate manifold.
- **Neural Functional Encoders (NFE):** Dynamic weight modulation based on the current manifold state.
- **Lifelong Consolidation:** A mechanism for meta-optimized weight updates that prevent catastrophic forgetting.

### 🚀 Getting Started

#### 1. Installation

```bash
git clone https://github.com/your-repo/Bicameral_Manifold_GMoE.git
cd Bicameral_Manifold_GMoE
uv pip install -r requirements.txt
```

#### 2. Model Acquisition

Download the base expert weights and configuration:

```bash
python scripts/setup.py --model llama-3-8b-bicameral
```

#### 3. Interactive Mode (The Bicameral Chat)

Engage with the dual-manifold system directly. See the manifold switching in your terminal:

```bash
python cli/interact.py
```

### 📂 Repository Structure

- `/core`: Intrinsic Dimension estimation, Geometric Routing, and NFE logic.
- `/experts`: Implementation of the Bicameral GMoE layers.
- `/scripts`: Setup, data ingestion, and model acquisition.
- `/cli`: Interactive interfaces and visualization tools.
- `/docs`: Detailed research papers and architectural specifications.
- `/experiments`: Orthogonal task arithmetic and fractal fine-tuning logs.

### 🧪 Testing

Run the verification suite to ensure manifold orthogonality:

```bash
pytest tests/
```

### 📜 Citation

If you use this architecture in your research, please cite the papers in `/docs`.
