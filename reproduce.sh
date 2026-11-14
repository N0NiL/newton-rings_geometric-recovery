#!/usr/bin/env bash
# One-click quick reproduction (Linux/macOS). Optional: ./reproduce.sh --with-manet
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1

WITH_MANET=0
for arg in "$@"; do
  if [[ "$arg" == "--with-manet" ]]; then
    WITH_MANET=1
  fi
done

echo "=== Newton-rings geometry: quick reproduce ==="
echo "Root: $(pwd)"
echo

# ---- 1) Python 3.11 ----
PY=""
for cand in python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)'; then
      PY="$cand"
      break
    fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "[FAIL] Python 3.11 not found. Install 3.11 and ensure python3.11 is on PATH."
  exit 1
fi
echo "[OK] using: $PY"
"$PY" -c 'import sys; print("     ", sys.version)'
echo

# ---- 2) venv + requirements ----
if [[ ! -x .venv/bin/python ]]; then
  echo "[..] creating .venv"
  if ! "$PY" -m venv .venv; then
    echo "[FAIL] could not create .venv"
    exit 1
  fi
else
  echo "[OK] .venv already exists"
fi
VPY="$(pwd)/.venv/bin/python"
echo "[..] pip install -r requirements.txt"
if ! "$VPY" -m pip install --upgrade pip; then
  echo "[FAIL] pip upgrade failed"
  exit 1
fi
if ! "$VPY" -m pip install -r requirements.txt; then
  echo "[FAIL] pip install -r requirements.txt failed"
  exit 1
fi
echo "[OK] dependencies installed"
echo

LOGDIR="$(pwd)/results/reproduce_logs"
mkdir -p "$LOGDIR"
FAILED=""

run_step() {
  local step="$1"
  local logfile="$2"
  shift 2
  echo "[..] step ${step}: python $*"
  if "$VPY" "$@" >"$logfile" 2>&1; then
    echo "[OK] step ${step}"
    return 0
  fi
  local rc=$?
  echo "[FAIL] step ${step} exited with code ${rc}"
  echo "       see ${logfile}"
  return 1
}

# ---- 3) four quick checks ----
if ! run_step 1 "$LOGDIR/01_reproduce_tables.log" scripts/reproduce_tables.py; then
  FAILED="${FAILED} [1]reproduce_tables"
fi
if grep -F -q "Table 1 match." "$LOGDIR/01_reproduce_tables.log" 2>/dev/null; then
  echo "[OK] Table 1 match. found in step 1 log"
else
  echo "[FAIL] step 1: did not print \"Table 1 match.\""
  FAILED="${FAILED} [1]Table1_match"
fi

if ! run_step 2 "$LOGDIR/02_stats.log" code/eval/stats.py; then
  FAILED="${FAILED} [2]stats"
fi

if ! run_step 3 "$LOGDIR/03_score_sensitivity.log" code/bench/run_score_sensitivity.py; then
  FAILED="${FAILED} [3]score_sensitivity"
fi

if ! run_step 4 "$LOGDIR/04_c1_c4_limit1.log" code/eval/eval_c1_c4_reviews.py --limit 1; then
  FAILED="${FAILED} [4]eval_c1_c4"
fi

# ---- 4) optional MA-Net smoke ----
if [[ "$WITH_MANET" -eq 1 ]]; then
  echo
  echo "=== --with-manet: one-frame MA-Net smoke ==="
  if [[ ! -f weights/50distill_last.pt ]]; then
    echo "[FAIL] missing weights/50distill_last.pt"
    FAILED="${FAILED} [manet]missing_weights"
  else
    if ! run_step manet "$LOGDIR/manet_comparison_limit1.log" code/bench/run_comparison_now.py --limit 1; then
      FAILED="${FAILED} [manet]comparison"
    fi
  fi
else
  echo
  echo "[skip] MA-Net smoke (pass --with-manet to enable)"
fi

echo
if [[ -n "$FAILED" ]]; then
  echo "========================================"
  echo "FAILED steps:${FAILED}"
  echo "Logs: ${LOGDIR}"
  echo "========================================"
  exit 1
fi

echo "========================================"
echo "全部通过"
echo "All quick checks passed."
echo "Logs: ${LOGDIR}"
echo "========================================"
exit 0
