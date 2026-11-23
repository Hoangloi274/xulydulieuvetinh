#!/usr/bin/env bash
# Training on the server: packed stores on local disk, then the model. Steps, in the order given:
#   pack    pack the 120 km radar and satellite of 2025 into the local packed stores (resumable)
#   stage   copy packed stores built elsewhere (r300 + ltg) from $UPLOADED into them
#           (pack and stage can run in either order: each adds its own arrays to the same monthly stores)
#   check   200 training steps, printing samples/s and the data-wait share (nothing kept)
#   train   train into runs/$RUN (ends with runs/$RUN/model.pt, calibrated; skipped if it exists)
#
#   bash scripts/server_train.sh pack stage check
#   nohup bash scripts/server_train.sh train > runs/train.log 2>&1 &
#   RUN=seed1 EXTRA="--seed 1" nohup bash scripts/server_train.sh train > runs/seed1.log 2>&1 &
#
# Why local disk: /workspace is NFS, too slow for random reads. The packed 2025 set is ~11 GB.
# Host RAM: the container is capped at 19.5 GB; 8 workers x ~0.7 GB + prefetched batches + main ~ 12 GB.
# Do not run a second training or inference job beside it.
set -euo pipefail

abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$(abspath "${RAW:-$PROJECT/../raw-data}")"
UPLOADED="$(abspath "${UPLOADED:-$RAW/packed}")"
CACHE="$(abspath "${CACHE:-/root/nowcast-cache/packed}")"  # local disk
RADAR="$(abspath "${RADAR:-$RAW/radar3d}")"
SAT="$(abspath "${SAT:-$RAW/satellite}")"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"   # uv is in ~/.local/bin on the server, not on PATH
TORCH="${TORCH:-cu118}"   # torch build: the pyproject extra (cu118 for the server's driver)
MONTHS="${MONTHS:-2025-01 2025-12}"
START="${START:-2025-01-01T01:00}"
END="${END:-2025-12-31T21:50}"
RUN="${RUN:-train}"
EXTRA="${EXTRA:-}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
COMMON="--epochs 40 --batch-size 32 --crop 192 --num-workers 8 --val-workers 2"
py() { "$UV" run --project "$PROJECT" --extra "$TORCH" python "$@"; }   # syncs .venv with uv.lock first

need() { [ -e "$1" ] || { echo "missing: $1 ($2)" >&2; exit 1; }; }
step() { echo; echo "=== $(date '+%F %T') $*"; }
free_gb() { df -P --block-size=1G "$1" | awk 'NR==2 {print $4}'; }

stage() {
  step "stage $UPLOADED -> $CACHE"
  need "$UPLOADED" "packed stores to stage"
  mkdir -p "$CACHE"
  local need_gb have_gb
  need_gb=$(( $(du -s --block-size=1G "$UPLOADED" | cut -f1) + 1 ))
  have_gb=$(free_gb "$CACHE")
  if (( have_gb - need_gb < MIN_FREE_GB )); then
    echo "local disk: $have_gb GB free, need $need_gb GB and must leave $MIN_FREE_GB GB; set CACHE= elsewhere" >&2; exit 1
  fi
  cp -rn "$UPLOADED"/. "$CACHE"/
  echo "staged; $(free_gb "$CACHE") GB free on the local disk"
}

pack() {
  step "pack r120 + sat for $MONTHS into $CACHE"
  need "$RADAR" "120 km column radar stores"; need "$SAT" "satellite stores"
  mkdir -p "$CACHE"
  if (( $(free_gb "$CACHE") - 12 < MIN_FREE_GB )); then   # r120 + sat of a year: ~9 GB, plus room for r300 + ltg
    echo "local disk: $(free_gb "$CACHE") GB free, packing needs ~12 GB and must leave $MIN_FREE_GB GB; set CACHE= elsewhere" >&2; exit 1
  fi
  # shellcheck disable=SC2086
  py -W ignore -m lightning_nowcast.pack --out "$CACHE" --months $MONTHS --sources r120 sat \
    --radar120 "$RADAR" --sat "$SAT" --workers 2
  py - "$CACHE" <<'PYEOF'
import sys, zarr
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("????_??.zarr")):
    g = zarr.open_group(str(p), mode="r")
    missing = [k for k in ("r120", "r300", "sat", "ltg") if k not in g]
    ok = {k: f"{g[k + '_ok'][:].mean():.0%}" for k in ("r120", "r300", "sat") if k + "_ok" in g}
    print(p.name, "MISSING " + ",".join(missing) if missing else "complete", ok)
PYEOF
  echo "$(du -sh "$CACHE" | cut -f1) packed; $(free_gb "$CACHE") GB free on the local disk"
}

check() {
  step "check"
  local dir="$PROJECT/runs/check"
  # shellcheck disable=SC2086
  py -W ignore -m lightning_nowcast.train --packed "$CACHE" --run-dir "$dir" --start "$START" --end "$END" $COMMON $EXTRA \
    --epochs 1 --steps-per-epoch 200 --max-val 64 --no-calibrate 2>&1 | grep --line-buffered -E "^(device|model|epoch 1:)"
  rm -rf "$dir"
}

train() {
  local dir="$PROJECT/runs/$RUN"
  if [ -f "$dir/model.pt" ]; then echo "$RUN: done ($dir)"; return; fi
  step "train -> $dir"
  # shellcheck disable=SC2086
  py -W ignore -m lightning_nowcast.train --packed "$CACHE" --run-dir "$dir" --start "$START" --end "$END" $COMMON $EXTRA
}

for s in "$@"; do
  case "$s" in stage|pack|check|train) ;; *) echo "unknown step '$s'; steps: pack stage check train" >&2; exit 2 ;; esac
done
need "$UV" "uv (https://docs.astral.sh/uv/); set UV=..."
cd "$PROJECT"
printf '%-9s %s\n' project "$PROJECT" uploaded "$UPLOADED" cache "$CACHE" radar "$RADAR" sat "$SAT" run "$RUN"
for s in "$@"; do "$s"; done
step "finished: $*"
