# -*- coding: utf-8 -*-
"""从表 1 冻结圆名单 + human/band 重判 S1.2，并按圆心门汇总。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from band_score import load_band
from pkg_paths import (
    DATA_COMPARISON,
    FITS_ROOT,
    GATES,
    NAMES,
    READABLE,
    TABLE1_EXPECTED,
)
from score_s12 import judge_s12


def _passes(score: dict[str, Any], spacing: float, gate: float | None) -> bool:
    if not score.get("success"):
        return False
    if gate is None:
        return True
    err = score.get("center_error_px")
    if err is None or spacing <= 1e-6:
        return False
    return float(err) <= gate * spacing + 1e-9


def load_readable_frames(
    session: Path | None = None,
    fits_root: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """每条路线：可读 69 帧的 human + radii + center。"""
    session = session or DATA_COMPARISON
    fits_root = fits_root or FITS_ROOT
    frames: dict[str, list[dict[str, Any]]] = {}
    for name in NAMES:
        rows = json.loads((fits_root / name / "results.json").read_text(encoding="utf-8"))
        entries: list[dict[str, Any]] = []
        for row in rows:
            if row.get("subset") not in READABLE:
                continue
            stem = row["stem"]
            human = load_band(session / f"{stem}_band.json", session / f"{stem}_human.json")
            entries.append(
                {
                    "stem": stem,
                    "subset": row["subset"],
                    "human": human,
                    "radii": list(row.get("radii_px") or []),
                    "center": None if row.get("center_xy") is None else tuple(row["center_xy"]),
                }
            )
        frames[name] = entries
    return frames


def score_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return judge_s12(entry["human"], entry["radii"], entry["center"])


def pass_vector(frames: dict[str, list[dict[str, Any]]], name: str, gate: float | None) -> list[int]:
    out: list[int] = []
    for entry in frames[name]:
        score = score_entry(entry)
        spacing = float(entry["human"]["spacing"])
        out.append(1 if _passes(score, spacing, gate) else 0)
    return out


def pass_counts(frames: dict[str, list[dict[str, Any]]], gate: float | None) -> dict[str, int]:
    return {name: sum(pass_vector(frames, name, gate)) for name in NAMES}


def table1_matrix(frames: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, list[int]]:
    frames = frames or load_readable_frames()
    return {name: [pass_counts(frames, g)[name] for g in GATES] for name in NAMES}


def check_table1(matrix: dict[str, list[int]] | None = None) -> tuple[bool, dict[str, list[int]]]:
    matrix = matrix or table1_matrix()
    ok = all(matrix[name] == TABLE1_EXPECTED[name] for name in NAMES)
    return ok, matrix
