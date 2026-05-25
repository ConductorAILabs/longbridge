#!/bin/bash
set -u
cd "$(dirname "$0")"
mkdir -p sweep_outputs

declare -a CONFIGS=(
  # tag             F   H   W    pixel
  "512x896          8   32  56   512x896"
  "640x1152         8   40  72   640x1152"
  "704x1280         8   44  80   704x1280"
  "704x1280-2blk   16   44  80   704x1280_16f"
)

for entry in "${CONFIGS[@]}"; do
  read -r tag F H W pixel <<<"$entry"
  outdir="sweep_outputs/$tag"
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
  data_path: test_prompts/single.txt
  image_or_video_shape:
  - 1
  - $F
  - 48
  - $H
  - $W

inference:
  sampling_steps: 4
  sink_size: 8
  guidance_scale: 1.0
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
  echo "=== $tag (latent ${F}x${H}x${W}, pixel ${pixel}) ==="
  T0=$(date +%s)
  PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
    --config_path "$cfg" > "$outdir/run.log" 2>&1
  RET=$?
  DT=$(($(date +%s) - T0))
  MP4=$(ls "$outdir"/*.mp4 2>/dev/null | head -1)
  if [ "$RET" -eq 0 ] && [ -n "$MP4" ]; then
    SZ=$(stat -f %z "$MP4"); echo "  ✓ OK ${DT}s  $((SZ/1024))KB  -> $MP4"
  elif [ "$RET" -eq 137 ]; then
    echo "  ✗ OOM ${DT}s — stopping"; break
  else
    echo "  ✗ FAIL exit=$RET ${DT}s"; tail -8 "$outdir/run.log" | sed 's/^/    /'
  fi
done
echo "=== sweep done ==="
