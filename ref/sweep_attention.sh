#!/bin/bash
set -u
cd "$(dirname "$0")"
mkdir -p attn_outputs

NEG="noisy, grainy, low quality, artifacts, blurry, soft, oversaturated, distorted"

# Try the unexplored attention/sink/tscale knobs the research surfaced
declare -a CONFIGS=(
  # tag             sink  attn   tscale
  "sink4_attn32     4     32     1.0"
  "sink8_attn16     8     16     1.0"
  "sink4_attn16     4     16     1.0"
  "sink4_tscale1.1  4     32     1.1"
  "best_combo       4     16     1.1"
)

for entry in "${CONFIGS[@]}"; do
  read -r tag sink attn tscale <<<"$entry"
  outdir="attn_outputs/$tag"
  mkdir -p "$outdir"
  cfg="$outdir/config.yaml"
  # Build config — only include inference_t_scale when != 1.0
  TSCALE_LINE=""
  if [ "$tscale" != "1.0" ]; then
    TSCALE_LINE="inference_t_scale: $tscale"
  fi
  cat > "$cfg" <<YAML
model_kwargs:
  model_name: Wan2.2-TI2V-5B
  timestep_shift: 5.0
  num_frame_per_block: 8
  local_attn_size: $attn

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
  sampling_steps: 4
  sink_size: $sink
  guidance_scale: 1.0
  multi_shot_sink: false
  multi_shot_rope_offset: 0
  streaming_vae: false
  async_vae: false
  vae_type: wan
  vae_device: "mps"
  negative_prompt: "$NEG"
  local_attn_size: $attn
$TSCALE_LINE

checkpoints:
  generator_ckpt: longlive_models/LongLive-2.0-5B/model_bf16.pt

logging:
  seed: 0
YAML
  echo "=== $tag (sink=$sink, attn=$attn, t_scale=$tscale) ==="
  T0=$(date +%s)
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
    --config_path "$cfg" > "$outdir/run.log" 2>&1
  RET=$?
  DT=$(($(date +%s) - T0))
  MP4=$(ls "$outdir"/*.mp4 2>/dev/null | head -1)
  if [ "$RET" -eq 0 ] && [ -n "$MP4" ]; then
    SZ=$(stat -f %z "$MP4"); echo "  ✓ OK ${DT}s  $((SZ/1024))KB"
  else
    echo "  ✗ FAIL exit=$RET ${DT}s"; tail -5 "$outdir/run.log" | sed 's/^/    /'
  fi
done
echo "=== sweep done ==="
