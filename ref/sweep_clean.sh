#!/bin/bash
set -u
cd "$(dirname "$0")"
mkdir -p clean_outputs

# 704x1280 base, vary steps + guidance to fight background noise
declare -a CONFIGS=(
  # tag           steps  guidance  shift
  "s4_g1_default   4     1.0       5.0"
  "s8_g1           8     1.0       5.0"
  "s8_g3           8     3.0       5.0"
  "s12_g3         12     3.0       5.0"
)

for entry in "${CONFIGS[@]}"; do
  read -r tag steps guide shift <<<"$entry"
  outdir="clean_outputs/$tag"
  mkdir -p "$outdir"
  cfg="$outdir/config.yaml"
  cat > "$cfg" <<YAML
model_kwargs:
  model_name: Wan2.2-TI2V-5B
  timestep_shift: $shift
  num_frame_per_block: 8
  local_attn_size: 32

use_ema: false
output_folder: $outdir
num_samples: 1
save_with_index: true

data:
  data_path: test_prompts/single.txt
  image_or_video_shape:
  - 1
  - 8
  - 48
  - 44
  - 80

inference:
  sampling_steps: $steps
  sink_size: 8
  guidance_scale: $guide
  multi_shot_sink: false
  multi_shot_rope_offset: 0
  streaming_vae: false
  async_vae: false
  vae_type: wan
  vae_device: "mps"

checkpoints:
  generator_ckpt: longlive_models/LongLive-2.0-5B/model_bf16.pt

logging:
  seed: 0
YAML
  echo "=== $tag (steps=$steps, guide=$guide, shift=$shift) ==="
  T0=$(date +%s)
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
    --config_path "$cfg" > "$outdir/run.log" 2>&1
  RET=$?
  DT=$(($(date +%s) - T0))
  MP4=$(ls "$outdir"/*.mp4 2>/dev/null | head -1)
  if [ "$RET" -eq 0 ] && [ -n "$MP4" ]; then
    SZ=$(stat -f %z "$MP4"); echo "  ✓ OK ${DT}s  $((SZ/1024))KB"
    # Open for visual comparison
    open "$MP4"
  elif [ "$RET" -eq 137 ]; then
    echo "  ✗ OOM ${DT}s"
  else
    echo "  ✗ FAIL exit=$RET ${DT}s"; tail -5 "$outdir/run.log" | sed 's/^/    /'
  fi
done
echo "=== sweep done ==="
