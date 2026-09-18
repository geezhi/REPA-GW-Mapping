# VideoREPA

Representation alignment for video diffusion transformers — aligning the internal features
of a CogVideoX denoiser to a frozen video foundation model in order to make video
generation physically plausible.

This repository studies **how** to bridge the student / teacher feature gap. It contains one
baseline (REPA-style learned projector) and a family of projector-free alignment objectives.
The flagship method is **GW-Relational**, which replaces the projector with a
Gromov-Wasserstein transport plan.

---

## Contents

* [Alignment objectives](#alignment-objectives)
* [Repository layout](#repository-layout)
* [Installation](#installation)
* [Quickstart](#quickstart)
* [GW-Relational](#gw-relational)
* [Results](#results)
* [Citation](#citation)

---

## Alignment objectives

All objectives tap the CogVideoX transformer at layer $l$ (default 18) and align the
resulting tokens to a frozen encoder (default VideoMAEv2, $D_2 = 768$).

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

## GW-Relational

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

| Method | Projector | VideoPhy SA $\uparrow$ | VideoPhy PC $\uparrow$ |
|---|---|---|---|
| CogVideoX-5B (baseline) | — | — | — |
| + REPA (`cosine_similarity`) | MLP | — | — |
| + OT dim align | None | — | — |
| **+ GW-Relational (ours)** | **None** | — | — |

Training logs for the released runs are kept at the workspace root
(`5b_lora_gw_align_full.log`, `2b_gw_dim_align_videophy2_eval.log`, ...).

---

## Citation

```bibtex
@article{videorepa2025,
  title   = {VideoREPA: Learning Physical Plausibility in Video Generation
             via Relational Alignment with Foundation Models},
  author  = {<authors>},
  journal = {<venue>},
  year    = {2025}
}
```

---

## Acknowledgements

The training framework is built on
[CogVideoX-Factory](https://github.com/a-r-r-o-w/cogvideox-factory) and
[diffusers](https://github.com/huggingface/diffusers). We thank the authors of
REPA, VideoMAEv2, VJEPA and VideoPhy. See `LICENSE`.
