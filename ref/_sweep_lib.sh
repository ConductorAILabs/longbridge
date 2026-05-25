# Shared helpers for sweep_*.sh scripts.
# Source from a sweep script after `set -u` and `cd "$(dirname "$0")"`.
#
# Provides:
#   sweep_init <out_root>          — mkdir the sweep's top-level output dir
#   sweep_run  <tag> <outdir> <cfg> [opts]
#       Runs inference.py against the config and prints a one-line status.
#       opts (space-separated flags, any order):
#         denoise      — also write <stem>_denoised.mp4 via ffmpeg hqdn3d.
#
# Each sweep_*.sh still writes its own YAML inline — the YAML bodies differ
# enough that table-driving them would be larger than the duplication it
# removes. The duplication this lib targets is the runner+reporter block.

sweep_init() {
  mkdir -p "$1"
}

sweep_run() {
  local tag="$1" outdir="$2" cfg="$3"
  shift 3
  local denoise=0
  for opt in "$@"; do
    case "$opt" in
      denoise) denoise=1 ;;
      *) echo "sweep_run: unknown opt '$opt'" >&2; return 64 ;;
    esac
  done

  local T0 RET DT MP4 SZ PP
  T0=$(date +%s)
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
    --config_path "$cfg" > "$outdir/run.log" 2>&1
  RET=$?
  DT=$(($(date +%s) - T0))
  MP4=$(ls "$outdir"/*.mp4 2>/dev/null | head -1)

  if [ "$RET" -eq 0 ] && [ -n "$MP4" ]; then
    SZ=$(stat -f %z "$MP4")
    echo "  ✓ OK ${DT}s  $((SZ/1024))KB"
    if [ "$denoise" -eq 1 ]; then
      PP="${MP4%.mp4}_denoised.mp4"
      ffmpeg -y -loglevel error -i "$MP4" \
        -vf "hqdn3d=8:6:9:9,unsharp=5:5:0.8:5:5:0.4" \
        -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p "$PP" 2>/dev/null
      echo "    + denoised: $((`stat -f %z "$PP"` / 1024))KB"
    fi
    return 0
  elif [ "$RET" -eq 137 ]; then
    echo "  ✗ OOM ${DT}s"
    return 1
  else
    echo "  ✗ FAIL exit=$RET ${DT}s"
    tail -5 "$outdir/run.log" | sed 's/^/    /'
    return 1
  fi
}
