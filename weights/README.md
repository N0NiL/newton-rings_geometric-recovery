# Weights

Inference-only MA-Net (student) checkpoint used in the paper.

| Field | Value |
|-------|-------|
| File | `weights/50distill_last.pt` |
| Size | 607 060 252 bytes (≈ 0.565 GiB) |
| MD5 | `5941203ce7b5384ba7a389c0a2606c83` |
| Contents | `student_state_dict` + inference `config` (preprocess / model / physics) + `contract` / `roi_hash`; no teacher, optimizer, or training metadata |

Verified bit-exact against the full training checkpoint on frame `000000_Easy_raw.png`
(CUDA; centerline / boundary / center / direction / `prob` max abs diff = 0).
A second pass removed the training metadata; its output again matches bit-exactly
(CPU; all heads, physics outputs, and the 414 contours identical). See `slim_verify_000000_Easy.json`.

Download: https://github.com/N0NiL/newton-rings_geometric-recovery/releases/download/v0.1.0/50distill_last.pt
(MD5 5941203ce7b5384ba7a389c0a2606c83; place it in weights/)

Place the file next to this README before re-running MA-Net inference
(`run_comparison_now.py`, `run_easy_translate.py`, …).

Training code, the teacher network, and the 333 training frames are **not** released.

ROI mask used at inference: `VisionAnalysis/roi.png` (relative to the package root).
