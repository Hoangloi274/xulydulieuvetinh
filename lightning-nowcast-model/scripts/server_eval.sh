#!/usr/bin/env bash
# Verify a checkpoint on the 2026 test periods (GPU inference, then virtual-station scoring).
#   bash scripts/server_eval.sh runs/train/model.pt
#   bash scripts/server_eval.sh models/lightning_nowcast.pt runs/train/model.pt     # several models, side by side
# Writes runs/verify/<first model>/: per period, stations within 250 km at all hours and in the 07-19 duty shift.
# Forecast caches go to runs/verify/cache/ (~1.2 GB per period and model; delete when done).
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
(( $# )) || { echo "usage: server_eval.sh CKPT [CKPT ...]" >&2; exit 2; }
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"   # uv is in ~/.local/bin on the server, not on PATH
TORCH="${TORCH:-cu118}"
PACKED="${PACKED:-$PROJECT/data/stores/packed_test}"
CSV="${CSV:-$PROJECT/data/lightning/strokes_2025-01_2026-09_300km.csv}"
py() { "$UV" run --project "$PROJECT" --extra "$TORCH" python "$@"; }
cd "$PROJECT"
name() { local d; d="$(basename "$(dirname "$1")")"; echo "${d}_$(basename "${1%.pt}")"; }
C=runs/verify/cache
OUT="runs/verify/$(name "$1")"
mkdir -p "$C" "$OUT"
declare -A FIRST=([2026-05-29]=2026-05-29T00:00 [2026-06-17]=2026-06-17T00:00)
declare -A LAST=([2026-05-29]=2026-06-06T22:50 [2026-06-17]=2026-06-21T22:50)
for p in 2026-05-29 2026-06-17; do
  caches=()
  for ck in "$@"; do
    cache="$C/$(name "$ck")_$p.npz"
    [ -f "$cache" ] || py -W ignore -m lightning_nowcast.forecast_cache --ckpt "$ck" --packed "$PACKED" \
      --start "${FIRST[$p]}" --end "${LAST[$p]}" --out "$cache" --batch 8
    caches+=("$(name "$ck")=$cache")
  done
  for d in all 7-19; do
    duty=(); [ "$d" = all ] || duty=(--duty-hours "$d")
    py -W ignore -m lightning_nowcast.verify --cache "${caches[@]}" --strokes-csv "$CSV" --out "$OUT/${p}_$d" "${duty[@]}" > /dev/null
  done
  echo "$p: done"
done
ls -la "$OUT"
