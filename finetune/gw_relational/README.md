# GW-Relational: Projector-Free Representation Alignment via Gromov-Wasserstein

Canonical, dependency-light implementation of the **GW alignment** used in VideoREPA.

> Align a diffusion transformer's internal features to a frozen video encoder
> **without any learnable projector** — the dimension mismatch is bridged by an
> optimal transport plan instead of an MLP, so 100% of the alignment gradient
> reaches the backbone.

---

## 1. Idea in one paragraph

REPA aligns the student's hidden states $X \in \mathbb{R}^{N \times D_1}$ to a teacher's
features $Y \in \mathbb{R}^{N \times D_2}$ by learning an MLP $f_\theta: \mathbb{R}^{D_1}
\to \mathbb{R}^{D_2}$. That projector has two downsides: it adds parameters, and — more
importantly — it *absorbs* the alignment gradient, so the backbone sees a weaker signal.

We drop the projector entirely. Instead we treat each **feature dimension** as a point to
be transported and solve a **Gromov-Wasserstein (GW)** problem between the two spaces. GW
never compares $X$ and $Y$ directly (they are not metrically comparable — different
dimensionalities); it compares their **intra-space relational structures**. The resulting
plan $T \in \mathbb{R}^{D_1 \times D_2}$ is then used as a soft projection to bring the
teacher into the student's space, where an ordinary cosine loss applies.

---

## 2. Method

### 2.1 Intra-space structure

Each dimension is a point, each token is an observation of that point:

$$C_X[i,k] = 1 - \cos\big(X_{:,i},\, X_{:,k}\big), \qquad
  C_Y[j,l] = 1 - \cos\big(Y_{:,j},\, Y_{:,l}\big)$$

with $C_X \in \mathbb{R}^{D_1 \times D_1}$, $C_Y \in \mathbb{R}^{D_2 \times D_2}$.

### 2.2 Entropic Gromov-Wasserstein

$$T^\* = \arg\min_{T \in \Pi(a,b)} \sum_{i,j,k,l} \big|C_X[i,k] - C_Y[j,l]\big|^2\,
T[i,j]\,T[k,l] \;-\; \varepsilon H(T)$$

where $\Pi(a,b) = \{T \ge 0 \mid T\mathbf{1} = a,\; T^\top\mathbf{1} = b\}$ with uniform
marginals $a = \tfrac{1}{D_1}\mathbf{1}$, $b = \tfrac{1}{D_2}\mathbf{1}$, and
$H(T) = -\sum_{ij} T[i,j]\log T[i,j]$.

### 2.3 Solver: linearize, then Sinkhorn

Linearize the quadratic term around the current plan, and solve the resulting entropic OT
in the log domain:

$$M^{(t)} = \big(C_X^2 a\big)\mathbf{1}^\top + \mathbf{1}\big(C_Y^2 b\big)^\top
            - 2\,C_X T^{(t)} C_Y^\top$$

$$u \leftarrow \log a - \mathrm{LSE}_j\!\left(-\tfrac{M^{(t)}}{\varepsilon} + v\right),
\qquad
v \leftarrow \log b - \mathrm{LSE}_i\!\left(-\tfrac{M^{(t)}}{\varepsilon} + u\right),
\qquad
T^{(t+1)}[i,j] = \exp\!\left(u_i - \tfrac{M^{(t)}[i,j]}{\varepsilon} + v_j\right)$$

`outer_tol` / `sinkhorn_tol` trigger early stopping; set to `0` to always run the full
budget.

### 2.4 Alignment loss

Row-normalize the plan into a soft projection and map the teacher into the student's space:

$$\hat T[i,j] = \frac{T^\*[i,j]}{\sum_{j'} T^\*[i,j']}, \qquad
  \hat Y = Y \hat T^\top \in \mathbb{R}^{N \times D_1}$$

**`gw_relational`** (default) — per-token cosine:

$$\mathcal{L}_{\text{GW}} = \frac{1}{N}\sum_{n=1}^{N}\left(1 -
\frac{X_n \cdot \hat Y_n}{\|X_n\|\,\|\hat Y_n\|}\right)$$

**`gw_gram`** — Gram-matrix MSE, $\;G_X = \tilde X \tilde X^\top$ on L2-normalized
features, $\;\mathcal{L} = \mathrm{MSE}(G_X, G_{\hat Y})$.

Total objective: $\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda\,\mathcal{L}_{\text{GW}}$
(`--proj_coeff`).

### 2.5 Zero-parameter spatial/temporal adaptation

CogVideoX emits tokens on a $13 \times 30 \times 45$ grid; VideoMAEv2 expects
$24 \times 10 \times 15$. We bridge this with **only non-parametric ops**: drop the first
latent frame, trilinear temporal upsample $\times 2$, then $3\times3$ average pooling.
No convolution, no linear layer — see `adapter.py`.

---

## 3. What's in this package

```
finetune/gw_relational/
├── gromov.py     # solver: structure matrices, entropic GW, log-Sinkhorn, GW cost
├── losses.py     # GWRelationalLoss / GWGramLoss (nn.Module, batched (B, N, D))
├── adapter.py    # ZeroParamFeatureAdapter: student grid -> teacher grid
├── example.py    # runnable 2-part demo (CPU, seconds)
└── tests/        # unit tests
```

The training entry point lives one level up in
`finetune/models/cogvideox_t2v_gw_align/lora_trainer.py`
(registered as `cogvideox-t2v-gw-align` for both `lora` and `sft`).

---

## 4. Usage

### 4.1 Standalone (no training loop needed)

```python
import torch
from finetune.gw_relational import GWRelationalLoss, GWSolverConfig, dimension_transport_plan

student = torch.randn(3600, 1920, requires_grad=True)   # (N, D1) from the transformer
teacher = torch.randn(3600, 768)                        # (N, D2) from the frozen encoder

cfg  = GWSolverConfig(reg=0.1, num_outer_iters=50, num_sinkhorn_iters=100)
loss = GWRelationalLoss(cfg)
value = loss.forward_flat(student, teacher)   # or loss(student[None], teacher[None])
value.backward()                              # gradient flows into `student` only

T = dimension_transport_plan(student.detach(), teacher, cfg)   # (1920, 768), no grad
```

### 4.2 Training

```bash
bash configs/gw_relational/train_5b_lora_gw_relational.sh     # CogVideoX-5B LoRA
bash configs/gw_relational/train_5b_lora_gw_gram.sh           # Gram-matrix variant
bash configs/gw_relational/train_2b_sft_gw_relational.sh      # CogVideoX-2B SFT
```

### 4.3 Tests and demo

```bash
pytest finetune/gw_relational/tests -q          # 29 unit tests, ~1 s on CPU
python finetune/gw_relational/example.py        # recovers a hidden dimension permutation
```

---

## 5. Hyper-parameters

| Group | Flag | Symbol | Default | Notes |
|---|---|---|---|---|
| Solver | `--gw_reg` | $\varepsilon$ | `0.1` | Larger $\Rightarrow$ smoother plan |
| Solver | `--gw_outer_iters` | $K_{\text{outer}}$ | `50` | Linearization steps |
| Solver | `--gw_sinkhorn_iters` | $K_{\text{sink}}$ | `100` | Inner Sinkhorn steps |
| Solver | `--gw_outer_tol` | $\tau_{\text{outer}}$ | `1e-4` | `0` disables early stopping |
| Solver | `--gw_sinkhorn_tol` | $\tau_{\text{sink}}$ | `1e-4` | `0` disables early stopping |
| Solver | `--gw_sample_size` | — | `0` | `0` = use all tokens |
| Loss | `--loss` | — | `gw_relational` | or `gw_gram` |
| Loss | `--gw_margin` | — | `0.0` | Hinge on cosine distance; `0` disables |
| Alignment | `--align_layer` | $l$ | `18` | Transformer layer to tap |
| Alignment | `--align_models` | — | `VideoMAEv2` | Frozen teacher |
| Alignment | `--proj_coeff` | $\lambda$ | `0.5` | Weight of $\mathcal{L}_{\text{GW}}$ |

> **`--margin` is *not* a GW flag.** It belongs to the TRD (token relation distillation)
> loss and is silently ignored by `gw_relational` / `gw_gram`. Use `--gw_margin`.
> Likewise `--gw_update_interval` and `--gw_distance_type` are deprecated no-ops kept for
> script compatibility.

### Choosing $\varepsilon$

$\varepsilon$ must be small relative to the **spread of the structure matrix**. For i.i.d.
Gaussian features that spread shrinks as $\sim 1/\sqrt{N}$, so $\varepsilon$ has to be
tightened as the problem grows — in `example.py`, $\varepsilon = 0.02$ already over-smooths
at $N = 128$ while $\varepsilon = 0.002$ recovers the permutation exactly. Real (trained)
features have far stronger structure and tolerate the default $\varepsilon = 0.1$.
**Symptom of too-large $\varepsilon$:** the plan is nearly uniform and the alignment loss
stops moving.

---

## 6. Cost

Per sample, the plan costs $O\!\left(K_{\text{outer}} (D_1^2 D_2 + D_1 D_2^2 +
K_{\text{sink}} D_1 D_2)\right)$. With $D_1 = 1920$, $D_2 = 768$,
$K_{\text{outer}} = 50$, $K_{\text{sink}} = 100$ this is roughly 15–20% wall-clock overhead
on top of the diffusion step. Two facts keep it cheap:

* it runs under `torch.no_grad()`, so it adds **no** backward memory;
* it is recomputed **per sample** (not cached), which keeps the correspondence
  sample-specific without a stale-plan lag.

---

## 7. Relation to other alignment objectives

| Method | Projector | Dim matching | Extra params | Structure-preserving |
|---|---|---|---|---|
| REPA | MLP | Learned | ~1.5M | No |
| TRD | MLP | Learned | ~1.5M | Partial |
| OT dim align | None | Sinkhorn **OT** | 0 | No |
| **GW (ours)** | **None** | **Gromov-Wasserstein** | **0** | **Yes** |

The distinguishing property: GW compares *intra-space relational structures* rather than
absolute feature values, so the correspondence is invariant to isometric transformations of
either space. Standard OT (as in `cogvideox_t2v_ot_dim_align`) needs the two spaces to be
metrically comparable and therefore relies on tokens already being spatially aligned.

---

## 8. Implementation notes / gotchas

* **No gradient through the plan.** `dimension_transport_plan` is `@torch.no_grad()`.
  Gradients reach the student only through the cosine/Gram term.
* **Everything is computed in fp32** even under bf16 mixed precision — the log-domain
  Sinkhorn is sensitive to precision.
* **Uniform marginals + $D_1 \ne D_2$ imply a soft plan.** A hard assignment is infeasible,
  so the transport plan is necessarily many-to-many. This is by design; it is what makes
  the projection differentiable.
* **Cosine distance is *not* invariant to sign flips** of a feature dimension. Do not
  expect GW to recover correspondences that differ by a sign.
* `gromov_cost(C_X, C_Y, T)` evaluates the objective at a given plan — handy for logging
  and for sanity checks (it should drop well below the value at a uniform plan).
