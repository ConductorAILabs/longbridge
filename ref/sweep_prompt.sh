#!/bin/bash
set -u
cd "$(dirname "$0")"
source ./_sweep_lib.sh
sweep_init prompt_outputs

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
  sweep_run "$tag" "$outdir" "$cfg" denoise
done
echo "=== done ==="
