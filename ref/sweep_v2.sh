#!/bin/bash
set -u
cd "$(dirname "$0")"
mkdir -p v2_outputs

declare -a SUBJECTS=(
  "faucet:test_prompts/long_structured.txt"
  "toilet:test_prompts/subj_toilet.txt"
  "sink:test_prompts/subj_sink.txt"
  "shower:test_prompts/subj_shower.txt"
  "tub:test_prompts/subj_tub.txt"
)

for entry in "${SUBJECTS[@]}"; do
  tag="${entry%%:*}"
  prompt_file="${entry##*:}"
  outdir="v2_outputs/$tag"
  mkdir -p "$outdir"
  cfg="$outdir/config.yaml"
  cat > "$cfg" <<YAML
model_kwargs:
  model_name: Wan2.2-TI2V-5B
  timestep_shift: 5.0
  num_frame_per_block: 8
  local_attn_size: 32
use_ema: false
output_folder: $outdir
num_samples: 1
save_with_index: true
data:
  data_path: $prompt_file
  image_or_video_shape: [1, 8, 48, 32, 56]   # 512x896 — known to fit with fp32 VAE
inference:
  sampling_steps: 4
  sink_size: 8
  guidance_scale: 1.0
  multi_shot_sink: true                       # critical for clean output
  multi_shot_rope_offset: 8
  streaming_vae: false
  async_vae: false
  vae_type: wan
  vae_device: "mps"
checkpoints:
  generator_ckpt: longlive_models/LongLive-2.0-5B/model_bf16.pt
logging:
  seed: 0
YAML
  echo "=== $tag ==="
  T0=$(date +%s)
  LONGBRIDGE_VAE_FP32=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. \
    ../.venv/bin/python inference.py --config_path "$cfg" > "$outdir/run.log" 2>&1
  RET=$?
  DT=$(($(date +%s) - T0))
  RAW=$(ls "$outdir"/*.mp4 2>/dev/null | head -1)
  if [ "$RET" -eq 0 ] && [ -n "$RAW" ]; then
    KB=$(stat -f %z "$RAW" | awk '{print int($1/1024)}')
    echo "  ✓ raw ${DT}s  ${KB}KB"
    # Apply static_bg post-process
    CLEAN="$outdir/clean.mp4"
    ../.venv/bin/python scripts/static_bg.py "$RAW" "$CLEAN" --threshold 25 2>&1 | tail -1
    echo "  ✓ cleaned -> $CLEAN"
  else
    echo "  ✗ FAIL exit=$RET ${DT}s"
    tail -5 "$outdir/run.log" | sed 's/^/    /'
  fi
done
echo "=== sweep done ==="
