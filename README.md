# RidgeFT — Exemplar-Free Analytical Continual MGT Attribution

Official repository for our **EMNLP 2026 Main Conference** paper,
*When New Generators Arrive: Lifelong Machine-Generated Text Attribution via
Ridge Feature Transfer*.

**RidgeFT** (**Ridge** **F**eature **T**ransfer) is a fully exemplar-free,
backprop-free framework for class-incremental machine-generated text (MGT)
attribution. A text encoder is fine-tuned **once** on the initial (base)
generators and then **frozen forever**. Every subsequently released generator
is absorbed by a *closed-form* update of per-class sufficient statistics —
no replay buffer, no distillation, no encoder retraining. Old-class knowledge
is never overwritten, so catastrophic forgetting is impossible by
construction, and "adding a new LLM" reduces to one dense matrix solve.

This repository contains the official reference implementation of the method
only (no baselines or full experiment harness).

---

## 1. Method

Let `f_enc : x -> h ∈ R^{d_h}` be the frozen CLS encoder (we use
DeBERTa-v3-base in the paper; any sentence encoder works). RidgeFT stacks
three analytical stages on top of `h`:

```
h  --(Module 1)-->  h~  --(Module 2)-->  φ(x)  --(Module 3)-->  ŷ
```

### Module 1 — Covariance calibration (`ridgeft.spectral.FractionalWhitening`)

Estimate the within-class scatter on the base classes,

```
S_w = (1/(N-C)) Σ_c Σ_{i: y_i=c} (h_i - μ_c)(h_i - μ_c)ᵀ,
```

apply trace-scaled shrinkage,

```
S_w' = (1-α) S_w + α (tr(S_w)/d) I,        α = 0.05,
```

eigen-decompose `S_w' = U diag(σ_j) Uᵀ` and rescale

```
h~ = U diag((σ_j + ε)^(-δ)) Uᵀ (h - μ).
```

The exponent `δ` acts **directly on the covariance eigenvalues**, so the
family interpolates between no scaling (`δ = 0`), **conventional (full)
whitening at `δ = 0.5`** (i.e. `Λ^(-1/2)`), and inverse-covariance scaling
at `δ = 1` (`Λ^(-1)`, stronger than whitening). The paper default is
`δ = 0.5` — standard whitening computed on the *shrunk within-class*
scatter. Directions with small within-class variance (stable
generator-"fingerprint" directions) are amplified relative to topic/length
nuisance directions, which dominate `S_w` and get compressed.

### Module 2 — Random feature lift (`ridgeft.random_features.RandomFeatureLift`)

```
φ(x) = LayerNorm( ReLU( R h~ ) ) ∈ R^{d_φ},   R_ij ~ N(0, 1/d_h),   d_φ = 4096.
```

`R` is isotropic Gaussian, sampled once at base time and never re-sampled.
The LayerNorm is **parameter-free** (per-row `(x - mean)/std`, no learnable
affine). The whole lift has zero trainable weights and is byte-identical
across all incremental steps.

### Module 3 — Class-balanced analytic ridge (`ridgeft.classifier.ClassBalancedRidgeClassifier`)

For every class `c` keep three sufficient statistics:

```
A_c = Φ_cᵀ Φ_c,    b_c = Σ_{i: y_i=c} φ_i,    N_c = #samples of class c.
```

At solve time re-weight per class,

```
ω_c = (N_c + τ)^(-β),   normalised so mean_c ω_c = 1     (τ = 0, β = 1),
```

assemble `Ā = Σ_c ω_c A_c`, `B̄[:,c] = ω_c b_c`, and solve one dense system:

```
W = (Ā + λI)^(-1) B̄,        ŷ(x) = argmax_c (φ(x)ᵀ W)_c.
```

With balanced classes the normalisation forces `ω_c ≡ 1`, so the head
collapses exactly to vanilla ridge regression — the class balancing is
free on balanced streams and only activates when a new generator arrives
with few samples.

### Continual update

```python
model.update_manyshot(H_new, y_new)   # new class(es), any sample size
```

internally computes `(A_c, b_c, N_c)` for the new class(es), merges them,
and re-solves the `(d_φ × d_φ)` ridge system. Old per-class statistics are
kept byte-identical, so old classes cannot be forgotten.

---

## 2. Default hyper-parameters

| Hyper-parameter        | Symbol | Default | Notes                                        |
|---                     |---     |---:     |---                                           |
| Whitening exponent     | `delta` | `0.5`  | = conventional whitening on the shrunk `S_w` |
| Trace shrinkage        | `shrinkage` | `0.05` | stabilises tail eigenvalues              |
| Random-feature dim     | `total_dim` | `4096` | larger helps at high class counts        |
| Ridge L2               | `ridge_lam` | `1.0` |                                           |
| Class-balance exponent | `beta`  | `1.0`  | inverse-frequency weighting                  |
| Class-balance smoothing| `tau_smoothing` | `0.0` | set `>0` only if some `N_c → 0`       |
| Random seed            | `seed`  | `42`   | fixes the projection `R`                     |

One fixed configuration is used for every experiment in the paper — no
per-dataset tuning.

---

## 3. Installation

```bash
git clone https://github.com/Vincent-HKUSTGZ/RidgeFT-Lifelong-MGT.git
cd RidgeFT-Lifelong-MGT
pip install -e .
```

The core package depends only on `numpy`. The real-data example additionally
needs `torch`, `transformers`, and `scikit-learn`:

```bash
pip install -r requirements.txt
```

---

## 4. Quick start (synthetic, runs in seconds on CPU)

```bash
python examples/minimal_demo.py
```

```python
import numpy as np
from ridgeft import RidgeFTModel

# H_base : (N, d_h) frozen-encoder embeddings of the base generators
# y_base : (N,)     integer class ids
model = RidgeFTModel.fit_base(H_base, y_base)   # all defaults from §2

model.update_manyshot(H_new, y_new)             # a brand-new generator arrives
y_hat = model.predict(H_test)
```

---

## 5. Real-data example (MGT-Academic layout)

`examples/run_mgt_academic.py` is a self-contained end-to-end sample:
it loads one topic of an MGT-Academic-style corpus, fine-tunes a DeBERTa
encoder on the 5 initial classes, freezes it, and runs the paper's P5
protocol (5 base classes + 1 incrementally added generator) with RidgeFT.

### Expected data layout

```
AI_Polish_clean/
├── Human/
│   ├── Physics/*.json          # each file: [ {"text": ...}, ... ]
│   ├── Math/*.json
│   └── ...
├── gpt35_new/
│   ├── Physics_task3.json      # each file: [ {"text": ...}, ... ]
│   └── ...
├── Mixtral_new/ ...
├── Moonshot_new/ ...
├── Llama3_new/ ...
└── gpt-4omini_new/ ...
```

Classes (canonical order): `Human, gpt35, Mixtral, Moonshot, Llama3,
gpt-4omini`. Subjects are grouped into topics (STEM / Humanities /
Social_sciences); per topic all 6 classes are balanced to the smallest
class count and split 80/10/10 with a fixed seed.

### Run

```bash
python examples/run_mgt_academic.py \
    --data-root /path/to/AI_Polish_clean \
    --encoder microsoft/deberta-v3-base \
    --topic STEM \
    --per-class 1000 \
    --device cuda:0
```

`--per-class 1000` subsamples each class so the demo finishes quickly;
drop the flag for the paper's full split. The script prints macro-F1 on
the base classes (S0), then adds the held-out generator with
`update_manyshot` and prints final full / old / new macro-F1.

Steps performed by the script (mirroring the paper recipe):

1. Load & balance the 6 classes, split 80/10/10 (seed 42).
2. Fine-tune the encoder on the **5 initial classes only**
   (2 epochs, batch 32, max_len 384, AdamW lr 2e-5), then **freeze** it.
3. Encode all splits to CLS features once.
4. `RidgeFTModel.fit_base` on the initial classes.
5. `update_manyshot` with the new generator's training features.
6. Evaluate 6-class macro-F1 on the untouched test split.

---

## 6. Repository layout

```
RidgeFT-Lifelong-MGT/
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
├── ridgeft/
│   ├── __init__.py
│   ├── spectral.py              # Module 1: FractionalWhitening
│   ├── random_features.py       # Module 2: RandomFeatureLift
│   ├── classifier.py            # Module 3: ClassBalancedRidgeClassifier
│   ├── model.py                 # end-to-end RidgeFTModel
│   └── utils.py                 # numerical helpers (incl. parameter-free LayerNorm)
├── examples/
│   ├── minimal_demo.py          # synthetic smoke demo (numpy only)
│   └── run_mgt_academic.py      # real-data end-to-end sample
└── tests/
    └── test_smoke.py
```

---

## 7. License

RidgeFT is released under the [MIT License](LICENSE).

<sub>Repository note: this repository was organized with Codex; if you encounter any issues, please open a GitHub issue.</sub>

## 8. Citation

If you use RidgeFT in your research, please cite our paper:

```bibtex
@inproceedings{SLHWHYH26,
author = {Zhen Sun and Yifan Liao and Zhicong Huang and Jiaheng Wei and Cheng Hong and Yutao Yue and \textbf{Xinlei He}},
title = {{When New Generators Arrive: Lifelong Machine-Generated Text Attribution via Ridge Feature Transfer}},
booktitle = {{Conference on Empirical Methods in Natural Language Processing (EMNLP)}},
publisher = {ACL},
year = {2026}
}
```
