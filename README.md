# GOLLuM+ — GP–LLM Integration for Molecular Design

Research code accompanying the MSc thesis project on **Gaussian-process–guided optimization over LLM embeddings for molecular design** (property focus: **logP**). The repository implements a **GOLLuM-style deep-kernel Gaussian Process (GP)** coupled with **LLM-based molecular embeddings** and benchmarks multiple **sequence decoders** (MLP, GRU and Vec2Text-style) within **Bayesian optimization** workflows.

> **At a glance**
> - Deep-kernel GP operating on LLM embeddings (T5-family; LoRA-friendly design).
> - Decoders for SELFIES/SMILES reconstruction: **MLP**, **GRU** and **Vec2Text-style**.
> - Two BO modes:
>   1) **Iterative BO** using the Deep GP as the objective in latent space.
>   2) **Decoder-driven BO** treating the decoder as a black-box objective.

---

## Installation

> A clean Python environment (≥ 3.10) and a CUDA-enabled GPU are recommended.

1. **Create and activate an environment**
   ```bash
   # Using conda (recommended)
   conda create -n gollum_env python=3.10 -y
   conda activate gollum_env

2. **Install PyTorch**
- Install a CUDA-compatible build via the official selector for your GPU.
- Example (adjust the CUDA tag to your system):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
3. **Install core dependencies**
  ```bash
  pip install transformers peft accelerate
  pip install botorch gpytorch
  pip install rdkit-pypi selfies
  pip install numpy pandas scikit-learn matplotlib optuna
  # optional logging
  pip install wandb
  ```

---

Data & Assumptions
- Input molecules as SMILES (converted to SELFIES internally when decoding).
- Target property: logP (Crippen, RDKit implementation).
- Tokenization and vocabulary are expected to be consistent across training/validation.

---

Typical Workflow
1. **Embed molecules with the LLM featurizer**
    - Use functions/utilities in `model.gollum_LLM.py` and `util.gollum_util.py` to tokenize strings, run the LLM, and pool hidden states (e.g., CLS, mean, weighted).
    - Optionally apply LoRA/parameter-efficient adaptation if configured in your workflow.
2. Fit the Deep-Kernel GP
    - Train the GP in latent space using embeddings as inputs and logP as targets.
    - Implementation scaffold in `model.gollum_DeepGP.py`.
3. Train decoders
    - SimpleMLP decoder `model.MLP_decoder.py`
    - GRU decoder: `modle.GRU_decoder.py`
    - Vec2Text-style decoder: `Vec2Text_decoder10.py` (Base + Corrector; iterative refinement)
    - Utilities for SELFIES/SMILES conversions and token metrics are in `util.decoder_util.py` and `util.util.py`.
5. Run Bayesian Optimization
    - Approach 1 (Iterative BO; recommended): `optimization.Approach1.py`
      Uses the Deep GP as the objective in latent space (e.g., qEI-style acquisition).
    - Approach 2 (Decoder-driven BO; experimental): `optimization.Approach2_2.py`
      Treats the decoder as the black-box objective.

---

Reproducibility Notes
- Fix random seeds (Python, NumPy, Torch) for controlled comparisons.
- The Vec2Text-style decoder involves iterative refinement; set a consistent max iteration budget and early stopping policy for fair benchmarking.
- GPU memory requirements may increase when using LLM + GP + decoder jointly.

---

Citing
If you use this repository, please cite the thesis and the primary methodological references:

- Thesis
  - M. Soh, “GOLLuM+: Integrated GP with LLM application in molecular design,” MSc Thesis, 2025.

- Foundations
  - Gómez-Bombarelli, R., et al. “Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules,” ACS Central Science, 2018.
  - Morris, J. X., et al. “Text Embeddings Reveal (Almost) As Much As Text,” 2023 (Vec2Text).
  - Ranković, B., et al. “GOLLuM: Gaussian Process Optimized LLMs,” 2025.
