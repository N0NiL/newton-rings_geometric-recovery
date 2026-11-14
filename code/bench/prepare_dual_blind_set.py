"""从 Open-arc 真子集中随机抽 15 帧，准备第二标注员双盲会话目录。

只复制分析画布与极简 meta（subset=Open-arc），绝不复制 *_human.json / *_band.json /
*_human_preview.png，避免第二标注员看到第一标注员结果。
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SESSION = Path(__file__).resolve().parent / "sessions" / "comparison"
_OUT = Path(__file__).resolve().parent / "sessions" / "dual_blind_b15"
_SEED = 20261111
_N = 15

# 文件名带 Open-arc、但人工复核为 Illegible 的 13 帧，不进入双盲池
_ILLEGIBLE_NAMED_OPEN = {
    "000042_Open-arc",
    "000043_Open-arc",
    "000044_Open-arc",
    "000045_Open-arc",
    "000046_Open-arc",
    "000047_Open-arc",
    "000048_Open-arc",
    "000050_Open-arc",
    "000060_Open-arc",
    "000061_Open-arc",
    "000064_Open-arc",
    "000065_Open-arc",
    "000068_Open-arc",
}


def _true_open_arc_stems() -> list[str]:
    stems: list[str] = []
    for canvas in sorted(_SESSION.glob("*Open-arc_canvas.png")):
        stem = canvas.name.replace("_canvas.png", "")
        if stem in _ILLEGIBLE_NAMED_OPEN:
            continue
        meta = canvas.with_name(f"{stem}.json")
        subset = "Open-arc"
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                subset = str(payload.get("subset") or "Open-arc")
            except json.JSONDecodeError:
                pass
        if subset == "Broken":
            subset = "Illegible"
        if subset != "Open-arc":
            continue
        human = canvas.with_name(f"{stem}_human.json")
        if not human.is_file():
            continue
        stems.append(stem)
    return stems


def main() -> int:
    pool = _true_open_arc_stems()
    if len(pool) < _N:
        raise SystemExit(f"Open-arc 真子集不足 {_N} 帧（现有 {len(pool)}）")
    rng = random.Random(_SEED)
    chosen = sorted(rng.sample(pool, _N))

    if _OUT.exists():
        shutil.rmtree(_OUT)
    _OUT.mkdir(parents=True)

    for stem in chosen:
        src_canvas = _SESSION / f"{stem}_canvas.png"
        dst_canvas = _OUT / f"{stem}_canvas.png"
        shutil.copy2(src_canvas, dst_canvas)
        # 极简 meta：只告诉软件这是 Open-arc，不含任何人工点/圆心
        (_OUT / f"{stem}.json").write_text(
            json.dumps(
                {
                    "subset": "Open-arc",
                    "dual_blind": True,
                    "annotator_slot": "B",
                    "source_session": "comparison",
                    "note": "双盲对照会话：本目录不得放置第一标注员 *_human.json。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema": "dual_blind_openarc_v1",
        "seed": _SEED,
        "n": _N,
        "pool_size": len(pool),
        "source_session": str(_SESSION.relative_to(_ROOT)).replace("\\", "/"),
        "blind_session": str(_OUT.relative_to(_ROOT)).replace("\\", "/"),
        "protocol": {
            "annotator_A": "sessions/comparison/*_human.json（第一标注员，评测主标注）",
            "annotator_B": "sessions/dual_blind_b15/*_human_B.json（第二标注员，独立双盲）",
            "blindness": "B 会话目录不含 A 的 human/band/preview；软件窗口不加载 A 标注。",
            "metric": "两位标注员外推极心欧氏距离 / 条纹间距 s（分析画布像素当量；与正文 0.5s/1s/2s 门同口径）",
            "s_note": "像素门控时 s 取相邻暗纹半径差的帧内估计或全局条纹间距像素当量；评分脚本默认用 A/B 牛顿半径差中位数估计条纹间距（px）。",
        },
        "stems": chosen,
        "pool_all": pool,
    }
    (_OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 仓库根也留一份便于备忘录引用
    memo_dir = _ROOT / "materials"
    memo_dir.mkdir(exist_ok=True)
    (memo_dir / "dual_blind_b15_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"种子={_SEED}，从 {len(pool)} 帧 Open-arc 中抽 {_N} 帧")
    print(f"输出目录：{_OUT}")
    for stem in chosen:
        print(f"  {stem}")
    print("已写入 manifest.json 与 materials/dual_blind_b15_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
