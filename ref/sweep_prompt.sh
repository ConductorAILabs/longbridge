#!/bin/bash
set -u
cd "$(dirname "$0")"
mkdir -p prompt_outputs

declare -a CONFIGS=(
  # tag                   prompt_file
  "long_default           test_prompts/long_structured.txt"
  "short_default          test_prompts/single.txt"
)

for entry in "${CONFIGS[@]}"; do
  read -r tag prompt_file <<<"$entry"
  outdir="prompt_outputs/$tag"
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
  image_or_video_shape:
  - 1
  - 8
  - 48
  - 44
  - 80

inference:
  sampling_steps: 4
  sink_size: 8
  guidance_scale: 1.0
  multi_shot_sink: true
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
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
    --config_path "$cfg" > "$outdir/run.log" 2>&1
  RET=$?
  DT=$(($(date +%s) - T0))
  MP4=$(ls "$outdir"/*.mp4 2>/dev/null | head -1)
  if [ "$RET" -eq 0 ] && [ -n "$MP4" ]; then
    SZ=$(stat -f %z "$MP4"); echo "  ✓ OK ${DT}s  $((SZ/1024))KB"
    # Also apply ffmpeg denoise post-process for comparison
    PP="${MP4%.mp4}_denoised.mp4"
    ffmpeg -y -loglevel error -i "$MP4" \
      -vf "hqdn3d=8:6:9:9,unsharp=5:5:0.8:5:5:0.4" \
      -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p "$PP" 2>/dev/null
    echo "    + denoised: $((`stat -f %z "$PP"` / 1024))KB"
  else
    echo "  ✗ FAIL exit=$RET ${DT}s"; tail -5 "$outdir/run.log" | sed 's/^/    /'
  fi
done
echo "=== done ==="
