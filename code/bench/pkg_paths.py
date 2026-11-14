# -*- coding: utf-8 -*-
"""包内路径垫片：让 code/bench 脚本在 release 树内直接运行。"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
PKG_ROOT = BENCH.parents[1]
CODE = PKG_ROOT / "code"
VISION = CODE / "vision"
NEWMODEL = CODE / "newmodel"
DATA_COMPARISON = PKG_ROOT / "data" / "comparison"
DATA_COMPARISON_RAW = DATA_COMPARISON / "raw"
DATA_STATIONARY = PKG_ROOT / "data" / "stationary"
DATA_DUAL_BLIND = PKG_ROOT / "data" / "dual_blind_b15"
DATA_MULTI_LENS = PKG_ROOT / "data" / "multi_lens"
FITS_ROOT = PKG_ROOT / "results" / "fits"
RESULTS = PKG_ROOT / "results"
WEIGHTS_CKPT = PKG_ROOT / "weights" / "50distill_last.pt"


def comparison_raw_path(stem: str) -> Path:
    """包内 raw 统一放在 data/comparison/raw/{stem}_raw.png。"""
    return DATA_COMPARISON_RAW / f"{stem}_raw.png"

# 仪器工作常数（与正文 / Table 9 一致）
SCALE_M_PER_PX = 7.188752e-6
LAMBDA_M = 589.3e-9

NAMES = ("classical", "v1.3", "Hough", "FRFT")
READABLE = ("Easy", "Open-arc")
TABLE1_EXPECTED = {
    "classical": [57, 58, 58, 58],
    "v1.3": [47, 57, 59, 60],
    "Hough": [0, 1, 1, 1],
    "FRFT": [30, 39, 41, 50],
}
GATES = (0.5, 1.0, 2.0, None)


def setup_import_path() -> Path:
    """把 bench / vision / newmodel 父目录加入 sys.path。"""
    # newmodel 包名在 Windows 上可按 NEWMODEL 导入（大小写不敏感）
    for path in (BENCH, VISION, CODE, PKG_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return PKG_ROOT
