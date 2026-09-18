"""Filesystem roots used across this repository.

Everything that used to be hard-coded as ``/efs/...`` now goes through this module so
the repository is portable and its releases do not leak machine-specific paths.

Environment overrides
---------------------
``REPA_GW_ROOT``    repository root      (default: the parent of ``finetune/``)
``VFM_CKPT_DIR``    pretrained weights   (default: ``<repo root>/checkpoints``)

Typical layout under ``VFM_CKPT_DIR``::

    checkpoints/
    ├── cogvideox-2b/
    ├── cogvideox-5b/
    ├── VideoMAEv2/vit_b_k710_dl_from_giant.pth
    ├── VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth
    └── vjepa_l/vitl16.pth.tar
"""

from __future__ import annotations

import os
from typing import Union

__all__ = ["REPO_ROOT", "CKPT_DIR", "repo", "ckpt"]

#: Absolute path of the repository root (the directory that contains ``finetune/``).
REPO_ROOT: str = os.environ.get(
    "REPA_GW_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

#: Absolute path of the directory holding pretrained weights.
CKPT_DIR: str = os.environ.get("VFM_CKPT_DIR", os.path.join(REPO_ROOT, "checkpoints"))

PathLike = Union[str, "os.PathLike[str]"]


def repo(*parts: str) -> str:
    """Join ``parts`` onto the repository root."""
    return os.path.join(REPO_ROOT, *parts)


def ckpt(*parts: str) -> str:
    """Join ``parts`` onto the checkpoint directory."""
    return os.path.join(CKPT_DIR, *parts)
