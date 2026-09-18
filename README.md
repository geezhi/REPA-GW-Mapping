# VideoREPA

Representation alignment for video diffusion transformers — aligning the internal features
of a CogVideoX denoiser to a frozen video foundation model so that generated videos are
physically plausible.

Standard REPA bridges the student / teacher feature gap with a *learnable* MLP projector.
We argue that the projector is the weak link: it adds parameters and, worse, **absorbs the
alignment gradient**, leaving the denoiser backbone with a diluted training signal.
Our method, **REPA w/ GW Mapping**, removes it entirely. Instead of *learning* a mapping
$\mathbb{R}^{D_1} \to \mathbb{R}^{D_2}$, we *solve* for one: an entropic
**Gromov-Wasserstein** transport plan $T \in \mathbb{R}^{D_1 \times D_2}$ between feature
**dimensions**, computed per sample with no gradients and no parameters. Row-normalizing
$T$ yields a soft projection that brings the teacher into the student's space, where an
ordinary cosine (or Gram) loss applies — so **100% of the alignment gradient reaches the
backbone**.

This repository also ships the baselines and the projector-free alternatives we compared
against (REPA, TRD, Sinkhorn-OT dimension alignment, local Gram flow, ...).

---

## Contents

* [Alignment objectives](#alignment-objectives)
* [Repository layout](#repository-layout)
* [Installation](#installation)
* [Quickstart](#quickstart)
* [GW Mapping](#gw-mapping)
* [Results](#results)
* [Acknowledgements](#acknowledgements)

---

## Alignment objectives

All objectives tap the CogVideoX transformer at layer $l$ (default 18) and align the
resulting tokens to a frozen encoder (default VideoMAEv2, $D_2 = 768$). The last two rows
are **REPA w/ GW Mapping**.

| `--loss` | Projector | How dimensions are matched | Extra params |
|---|---|---|---|
| `cosine_similarity` | MLP | Learned (REPA baseline) | ~1.5M |
| `token_relation_distillation` | MLP | Learned | ~1.5M |
| `cka_alignment` | MLP | Learned | ~1.5M |
| `gram_matrix` | MLP | Learned | ~1.5M |
| `local_gram_flow` | None | Fixed pooling + local Gram | 0 |
| `ot_dim_align` | None | Sinkhorn **OT** across dims | 0 |
| **`gw_relational`** | **None** | **Gromov-Wasserstein** | **0** |
| **`gw_gram`** | **None** | **Gromov-Wasserstein** + Gram MSE | **0** |

The key hypothesis: *a learnable projector absorbs the alignment gradient*. Removing it
(projector-free objectives) sends the full signal into the denoiser backbone.

---

## Repository layout

```
VideoREPA/
├── configs/                    # launch scripts, grouped by alignment method
│   ├── gw_relational/          # ★ the released GW runs (2B SFT, 5B LoRA, gw_gram)
│   ├── repa_baseline/          #   REPA / TRD / ablations (learned projector)
│   ├── ot_dim_align/           #   projector-free, Sinkhorn-OT across dimensions
│   ├── dim_align/              #   projector-free, dimension-level OT
│   ├── dim_token_align/        #   dimension-guided token alignment
│   ├── direct_align/           #   direct relational alignment (no transport plan)
│   ├── local_gram_flow/        #   local Gram matrix + temporal flow
│   ├── merger_align/           #   token-merging alignment
│   ├── fixed_proj_align/       #   frozen random projector
│   └── data/                   #   dataset precomputation
├── finetune/
│   ├── gw_relational/          # ★ canonical GW solver + losses + adapter + tests
│   │   ├── gromov.py           #   entropic GW, log-Sinkhorn, GW cost
│   │   ├── losses.py           #   GWRelationalLoss / GWGramLoss
│   │   ├── adapter.py          #   zero-parameter student -> teacher grid resampling
│   │   ├── example.py          #   runnable CPU demo
│   │   └── tests/              #   unit tests
│   ├── models/                 # one sub-package per alignment variant (trainers)
│   ├── schemas/                # args / components / state
│   ├── paths.py                # ★ REPO_ROOT / CKPT_DIR (env-overridable)
│   ├── train.py                # entry point (accelerate)
│   └── openvid/                # training data (not tracked by git)
├── inference/                  # video generation from a trained checkpoint
└── evaluation/                 # VideoPhy / VBench evaluation
```

---

## Installation

```bash
conda create -n videorepa python=3.10 -y && conda activate videorepa
pip install -r requirements.txt
```

Checkpoints (CogVideoX-2B / 5B, VideoMAEv2, ...) are read from `${CKPT_DIR}`, which
defaults to `<repo root>/checkpoints`; see `download.sh` and `download_vfm.sh`. All paths
are resolved through `finetune/paths.py`, so nothing is hard-coded to a particular machine.

```bash
export CKPT_DIR=/path/to/your/checkpoints
export VFM_CKPT_DIR=${CKPT_DIR}
```

> `finetune/pretrained_projector.pth` and `finetune/teacher_pca_matrix.pth` are **not** in
> git (58 MB / 21 MB); they are shipped as release assets.

---

## Quickstart

**Train** (CogVideoX-5B + LoRA, GW-Relational alignment, 8 GPUs):

```bash
bash configs/gw_relational/train_5b_lora_gw_relational.sh
```

**Merge the DeepSpeed checkpoint and generate videos:**

```bash
python zero_to_fp32.py ./checkpoint-4000 ../transformer --safe_serialization
# GW checkpoints contain a training-only `downsampler_cogvideo_output` tensor;
# drop it (or pass --gw_model) before loading with a stock CogVideoX pipeline.
python inference/generate.py --model_path <merged> --gw_model --upsampled \
       --input_file videophy.txt --output_dir out/
```

**Evaluate** on VideoPhy:

```bash
cd evaluation/videophy
python utils/prepare_data.py --input_csv <csv> --output_folder <out>
python videocon/training/pipeline_video/entailment_inference.py \
       --input_csv <out>/sa_testing.csv --output_csv <out>/sa.csv \
       --checkpoint ${VIDEOPHY_CKPT}
python calculate_mean.py
```

---

## GW Mapping

Given student features $X \in \mathbb{R}^{N \times D_1}$ and teacher features
$Y \in \mathbb{R}^{N \times D_2}$ with $D_1 \ne D_2$, we solve

$$T^\* = \arg\min_{T \in \Pi(a,b)} \sum_{i,j,k,l} \big|C_X[i,k] - C_Y[j,l]\big|^2 T[i,j]T[k,l] - \varepsilon H(T), \qquad C_X[i,k] = 1 - \cos(X_{:,i}, X_{:,k})$$

row-normalize $T^\*$ into a projection, map $\hat Y = Y \hat T^\top \in \mathbb{R}^{N \times D_1}$,
and apply a per-token cosine loss. Full derivation, hyper-parameter table, complexity
analysis and practical guidance live in
**[`finetune/gw_relational/README.md`](finetune/gw_relational/README.md)**.

```bash
pytest finetune/gw_relational/tests -q      # 29 unit tests
python finetune/gw_relational/example.py    # recovers a hidden dimension permutation
```

---

## Results

<!-- TODO: fill in from evaluation/. Numbers below are placeholders. -->

| Method | Projector | Extra params | VideoPhy SA $\uparrow$ | VideoPhy PC $\uparrow$ |
|---|---|---|---|---|
| CogVideoX-5B (baseline) | — | — | — | — |
| + REPA (`cosine_similarity`) | MLP | ~1.5M | — | — |
| + OT dim align | None | 0 | — | — |
| **+ REPA w/ GW Mapping (ours)** | **None** | **0** | — | — |

<!-- TODO: fill in from evaluation/. Numbers above are placeholders. -->

Training logs for the released runs live under `finetune/output_dir*/` and are not tracked
by git.

---

## Acknowledgements

The training framework is derived from
[CogVideoX-Factory](https://github.com/a-r-r-o-w/cogvideox-factory) and
[diffusers](https://github.com/huggingface/diffusers). We thank the authors of
REPA, VideoMAEv2, VJEPA and VideoPhy. See `LICENSE`.
