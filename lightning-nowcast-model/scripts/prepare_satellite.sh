#!/usr/bin/env bash
# Himawari stores for extra months, streamed day by day so raw files never pile up (~25 GB per day of
# raw HSD would otherwise accumulate to ~100 GB per month): download one UTC day -> convert it into the
# monthly store -> delete the day's raw files. Re-running resumes: days already in a store are skipped.
#
#   bash scripts/prepare_satellite.sh 2025-09 2025-10 2025-11 2025-12
#
# Environment: SAT (output stores, default data/stores/satellite), RAW_TMP (scratch for one day of raw
# files, default data/raw/hsd-tmp), WORKERS (conversion
# processes, 8), SLOT_WORKERS (time slots downloaded at once, 6).
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py() { uv run --project "$PROJECT" python "$@"; }
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
SAT="$(abspath "${SAT:-$PROJECT/data/stores/satellite}")"
RAW_TMP="$(abspath "${RAW_TMP:-$PROJECT/data/raw/hsd-tmp}")"
WORKERS="${WORKERS:-8}"
SLOT_WORKERS="${SLOT_WORKERS:-6}"   # time slots downloaded at once
[ $# -gt 0 ] || { echo "usage: $0 YYYY-MM [YYYY-MM ...]" >&2; exit 2; }
py -c "import satpy, boto3" || { echo "the project environment needs satpy and boto3 (uv sync)" >&2; exit 1; }
mkdir -p "$SAT" "$RAW_TMP"

last_time() {  # last stored time of a month store, or empty
  py - "$1" <<'PYEOF'
import sys, xarray as xr
from pathlib import Path
p = Path(sys.argv[1])
if p.exists():
    try:
        t = xr.open_zarr(p, consolidated=False)["time"].values
        print(str(t[-1])[:16] if t.size else "")
    except Exception:
        print("")
PYEOF
}

for ym in "$@"; do
  y=${ym%-*}; m=${ym#*-}
  store="$SAT/himawari_patch_${y}_${m}.zarr"
  ndays=$(date -d "$y-$m-01 +1 month -1 day" +%d)
  for d in $(seq -w 1 "$ndays"); do
    day="$y-$m-$d"; next=$(date -d "$day +1 day" +%F)
    end="${next}T00:00"
    [ -z "${TEST_HOURS:-}" ] || end="$(date -d "$day +${TEST_HOURS} hour" +%FT%H:%M)"   # testing only
    last=$(last_time "$store")
    if [ -n "$last" ] && [[ "$last" > "${day}T23:40" ]]; then echo "$day: already in $(basename "$store")"; continue; fi
    echo "=== $(date '+%F %T') $day: download"
    py -m lightning_nowcast.himawari_download --start "${day}T00:00" --end "$end" --out-dir "$RAW_TMP" \
      --max-workers 16 --slot-workers "$SLOT_WORKERS" 2>&1 | grep -E "^(miss|fail|summary)" || true
    echo "=== $(date '+%F %T') $day: convert"
    py -m lightning_nowcast.himawari_zarr --raw-dir "$RAW_TMP" --out-dir "$SAT" \
      --start "${day}T00:00" --end "$end" --center-lat 10.6595 --center-lon 106.7283 \
      --workers "$WORKERS" --no-cache --resume 2>&1 | grep -E "^\[" || true
    rm -rf "${RAW_TMP:?}/$y/$m/$d"
    [ -z "${TEST_HOURS:-}" ] || break
  done
done
echo "=== $(date '+%F %T') finished: $*"
