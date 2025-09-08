# GOLLuM+ — GP–LLM Integration for Molecular Design

Research code accompanying the MSc thesis **“GOLLuM+: Integrated GP with LLM application in molecular design”** (Imperial College London). This repository implements a **GOLLuM-style deep-kernel Gaussian Process (GP)** coupled to **LoRA-adapted LLM embeddings** for **logP-guided** molecular design, and benchmarks three decoders (SimpleMLP, GRU, Vec2Text-style) within Bayesian optimization workflows.

> **Highlights**
> - Deep-kernel GP over LLM embeddings (T5-base) with parameter-efficient tuning (LoRA).
> - Decoders for SELFIES/SMILES reconstruction: **SimpleMLP**, **GRU**, **Vec2Text-style** (Base + Corrector).
> - Three optimization modes:
>   1) **Batch one-shot** BO in latent space.  
>   2) **Iterative** BO using the deep-kernel GP as objective.  
>   3) **Decoder-driven** BO treating the decoder as the black-box objective.

---

## Installation

### 1) Python and PyTorch
- Python **3.10+** recommended.
- Install a **CUDA-compatible** PyTorch build for your GPU (see the official PyTorch selector).

### 2) Core dependencies
Install the remaining libraries:
pip install transformers peft accelerate
pip install botorch gpytorch
pip install rdkit-pypi selfies
pip install optuna numpy pandas scikit-learn matplotlib
# optional (if you log runs)
pip install wandb

---

Data expectations
- Molecular strings as SMILES (converted to SELFIES internally for decoding tasks).
- Target property: logP (Crippen) computed via RDKit.
- The code expects a token vocabulary consistent with the SELFIES used for training/validation.

---

Results summary (reproducibility and scope)
- The iterative BO with the deep-kernel GP objective outperformed batch one-shot suggestions in the latent space for logP (e.g., reaching ≈8.8 by iteration 15 vs. ≈5.2 best from batch one-shot in one run/seed).
- Decoder fidelity (exact-match, token-match, property-match) was insufficient for reliable end-to-end optimization in this small-data setting; Vec2Text-style benefited from iterative refinement but remained below practical thresholds at the 10-iteration inference budget.
- Experiments were conducted once per seed; random, NumPy, and Torch seeds were fixed. Hyperparameter tuning is not fully reproducible due to compute limits; all result artifacts/plots are saved for inspection.

---

Citation

Thesis
Soh, M. C. “GOLLuM+: Integrated GP with LLM application in molecular design,” MSc Thesis, Imperial College London, 2025.

Methodological foundations
Ranković, B., & Schwaller, P. “GOLLuM: Gaussian process optimized LLMs—reframing LLM finetuning through Bayesian optimization,” arXiv, 2025.
Morris, J. X., et al. “Text Embeddings Reveal (Almost) As Much As Text,” 2023. (Vec2Text)
Gómez-Bombarelli, R., et al. “Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules,” ACS Central Science, 2018. (ChemVAE)

---

Acknowledgments
Supervision by Dr. Antonio del Rio Chanona and Mathias Neufang. This work builds upon the GOLLuM framework and Vec2Text decoding, adapted to molecular design under data-scarce conditions.
