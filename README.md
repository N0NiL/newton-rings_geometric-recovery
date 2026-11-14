# Robust geometric recovery of Newton’s rings under non-ideal imaging — reproduction package

This package contains the data, frozen results, and code behind the manuscript
**“Robust geometric recovery of Newton’s rings under non-ideal imaging”**
(H. Lin), submitted to *Optics and Lasers in Engineering*.
With it, the main numbers in the paper can be checked without a GPU, and the
fits can be regenerated from the recorded frames.

中文说明见 [README_zh.md](README_zh.md).

## Requirements

- Python 3.11 (the paper tables were produced with 3.11)
- Packages pinned in `requirements.txt`:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is optional and only speeds up MA-Net inference. Some console messages
printed by the bench scripts are in Chinese. They do not affect the results.

## One-click reproduce

Creates a local `.venv`, installs `requirements.txt`, runs the four quick checks
below, and requires the log line `Table 1 match.`. Prints **全部通过** on success.

```bat
reproduce.bat
reproduce.bat --with-manet
```

```bash
chmod +x reproduce.sh
./reproduce.sh
./reproduce.sh --with-manet
```

`--with-manet` also runs a one-frame MA-Net fit (`run_comparison_now.py --limit 1`)
and needs `weights/50distill_last.pt`. Logs go to `results/reproduce_logs/`.

## Quick check (no GPU, no weights)

Same four commands as the one-click script (run them yourself if the venv is already set up):

```bash
python scripts/reproduce_tables.py
python code/eval/stats.py
python code/bench/run_score_sensitivity.py
python code/eval/eval_c1_c4_reviews.py --limit 1
```

`reproduce_tables.py` re-scores the frozen circle lists and should end with
`Table 1 match.`. It also recomputes the multi-lens radii of Sec. 3.6.
`stats.py` recomputes the Wilson intervals, exact McNemar tests, Wilcoxon tests,
and the Holm correction from the same files.

## Repository layout

```
data/comparison/       93 labelled frames: analysis canvases, raw frames (raw/),
                       subset sidecars, human pole/ring annotations, band annotations
data/stationary/       40-frame stationary sequence (Tables 7–8)
data/dual_blind_b15/   second-annotator set, 15 open-arc frames (Sec. 3.2)
data/multi_lens/       ring lists of the five multi-lens trials (Sec. 3.6)
results/fits/          frozen circle lists of the four routes
results/timing.json    per-frame run times (Table 6)
results/static.json    stationary-sequence fits (Tables 7–8)
results/easy_translate/     translation experiment, d = 0–200 px (Sec. 3.3)
results/score_sensitivity/  ±20% perturbation of the scoring constants (Sec. 4)
results/hough_oracle/       24-configuration Hough parameter sweep (Sec. 2.5)
results/c1_c4/              Monte Carlo pole perturbation; γ scaling and ablations
results/dual_blind/         pole differences between the two annotators
code/vision/           three-stage solver and background-subtraction frontend
code/newmodel/         MA-Net inference
code/bench/            scoring, comparison routes (Hough, FRFT-type), experiment drivers
code/eval/             statistics and Monte Carlo evaluation
docs/criteria.md       definition of the pass criteria and matching rules
weights/               MA-Net inference checkpoint (distributed separately, see weights/README.md)
```

Route names in the result files map to the paper as follows:

| File name | Route in the paper |
|-----------|--------------------|
| `classical` | background-subtraction frontend + three-stage solver |
| `v1.3` | MA-Net frontend + three-stage solver |
| `Hough` | circle Hough transform |
| `FRFT` | global r²-spectrum (FRFT-type) search |

## Paper item → command → artifact

| Paper item | Command | Artifact |
|------------|---------|----------|
| Table 1 (pass counts at 0.5 Λ / 1 Λ / 2 Λ / pole untested) | `python scripts/reproduce_tables.py` | re-scored from `results/fits/*` |
| Tables 2–5, 9 | frozen results (existence checked by `reproduce_tables.py`) | `results/fits/`, `results/c1_c4/` |
| Table 6 | frozen results | `results/timing.json` |
| Tables 7–8 | frozen results | `results/static.json`, `data/stationary/` |
| Sec. 3.2, second annotator (median 0.54 Λ, 73.3% within 1 Λ) | `python code/bench/score_dual_blind.py` | `results/dual_blind/annotator_AB_pole_delta.json` |
| Sec. 3.2, Monte Carlo pole perturbation (seed 42) | `python code/eval/eval_c1_c4_reviews.py` | `results/c1_c4/part1_results.json` |
| Sec. 3.3, translation experiment | `python code/bench/run_easy_translate.py` (needs weights) | `results/easy_translate/` |
| Sec. 3.6, γ scaling | frozen results | `results/c1_c4/part2_results.json` → `radial_scaling_generalization` |
| Sec. 3.6, multi-lens R | `python scripts/reproduce_tables.py` | `data/multi_lens/confidence_db_multilens.json` |
| Sec. 4, scoring-constant sensitivity | `python code/bench/run_score_sensitivity.py` | `results/score_sensitivity/` |
| Sec. 2.5, Hough parameter sweep | `python code/bench/run_hough_oracle.py` | `results/hough_oracle/` |
| Wilson / McNemar / Wilcoxon / Holm | `python code/eval/stats.py` | computed from `results/fits/` |

## Regenerating the fits

The scripts in `code/bench/` resolve all paths inside this package through
`code/bench/pkg_paths.py`. A one-frame smoke test without MA-Net:

```bash
python code/bench/run_comparison_now.py --limit 1 --skip-manet
python code/bench/run_v162_timing_static.py --limit 1 --skip-manet
```

Smoke runs write to `results/smoke_*` and leave the frozen results untouched.
A full run regenerates the circle lists of Tables 1–5 (`run_comparison_now.py`)
or the timings and stationary fits of Tables 6–8 (`run_v162_timing_static.py`)
and writes them to `results/`, replacing the frozen copies. The MA-Net route
and the translation experiment need the checkpoint at `weights/50distill_last.pt`.
Run times depend on the machine, so Table 6 will not be reproduced exactly.
MA-Net reruns can also differ from the frozen fits at the sub-pixel level
(e.g. 0.01 px in the pole on frame 000000) because of GPU and library versions;
the pass/fail decisions are unaffected on the frames we checked.

## MA-Net

Only the **inference-only** checkpoint of the distilled ResNet-50 MA-Net student
is provided (`weights/50distill_last.pt`, ≈607 MB / 0.565 GiB;
MD5 `5941203ce7b5384ba7a389c0a2606c83`). Teacher weights, optimizer state, and training metadata
were stripped from the training dump; one-frame output matches the full
checkpoint bit-exactly (see `weights/README.md`). The file fits under GitHub’s
2 GiB Release limit. Training code and the 333 training frames are not included.
Student inference uses `code/newmodel/models/baseline.py`.

## Instrument constants

| Quantity | Value |
|----------|-------|
| Wavelength λ | 589.3 nm (sodium D line) |
| Pixel scale p | 7.189 µm/pixel on the 768 × 768 analysis canvas |

Both are working constants of the instrument software. They are not an
independent metrological calibration of the relay optics, so lens radii
derived from them serve only as a plausibility check, as in the paper.

## Frame IDs and subsets

- File names use six-digit frame IDs where available (e.g. `000018_Open-arc_canvas.png`);
  the paper shortens them to four digits (frame 0018).
- The subset (`Easy`, `Open-arc`, `Illegible`) is given by the `subset` field of
  each sidecar `*.json`, not by the file name. Thirteen files named `*_Open-arc*`
  are labelled `Illegible`.

## Multi-lens trials (Sec. 3.6)

The five trials were exported from the result log of the instrument software,
which stores the pole and the ring list of each fit but not the interferogram
images. For each trial, the lens radius is

R = median over adjacent orders of Δ(r²) · p² / λ,

where r is the ring radius in pixels from `fit.rings` and p is the pixel scale.
The paper reports the median of the two trials of each specimen
(1.091 m and 1.075 m) and the reinstalled original lens (1.916 m).
`scripts/reproduce_tables.py` prints all five per-trial values.
The field `R_m_from_platform` is a library-averaged value kept by the instrument
software and is not the value used in the paper.

## Not included

- The laboratory acquisition and teaching platform
- The earlier 40-pixel crop experiment, which the translation experiment replaces
- Internal revision notes of the scoring rules (the final rules are in `docs/criteria.md`)
- Manuscript-preparation files
- MA-Net training data, training code, and the teacher network

## License

- Code (`code/`, `scripts/`, `reproduce.bat`, `reproduce.sh`): GNU GPL v3.0 or later, see [LICENSE](LICENSE).
  You may use, modify, and redistribute the code, including commercially, but any
  distributed derivative must be released under the same license with its source code.
- Data, results, and the inference checkpoint: CC BY 4.0, see [LICENSE-DATA](LICENSE-DATA)

## Citation

Please cite the article; citation metadata are in [CITATION.cff](CITATION.cff).
