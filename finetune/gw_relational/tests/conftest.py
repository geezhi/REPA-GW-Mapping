"""Pytest configuration: make the repository importable from anywhere."""

import sys
from pathlib import Path

# tests/ -> gw_relational/ -> finetune/ -> VideoREPA/
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FINETUNE_ROOT = Path(__file__).resolve().parents[2]

for _path in (_REPO_ROOT, _FINETUNE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
