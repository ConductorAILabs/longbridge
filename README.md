# longbridge

**Run NVIDIA LongLive 2.0 5B on Apple Silicon (Mac, MPS) — no CUDA, no NVFP4, no flash-attn.**

A PyTorch+MPS bridge for [NVlabs/LongLive 2.0](https://github.com/NVlabs/LongLive). In-place patches to the upstream code that swap CUDA-only paths for Mac-friendly equivalents. The model runs natively in bf16 on Apple Silicon — base Wan 2.2 TI2V-5B + LongLive 2.0 fine-tune.

> First public MLX/MPS port of LongLive 2.0. Verified end-to-end on M5 Max with 128GB unified memory.

Built by **[Conductor AI Labs](https://www.conductorailabs.com)** as part of our work on local-first AI video infrastructure.

## What this is

LongLive 2.0 is a frame-level autoregressive long-video model from NVIDIA. The
official inference stack assumes:

- CUDA + NCCL distributed launch
- Optional NVFP4 quantization on Blackwell GPUs
- flash-attn 2/3
- fp64 math in RoPE / time embeddings

Apple Silicon has none of these. This repo carries the smallest set of patches
needed to run the bf16 inference path on MPS:

- Skip distributed init when CUDA isn't available
- Lazy device picking (`cuda` → `mps` → `cpu`)
- Fall back to `scaled_dot_product_attention` when flash-attn is missing
- Replace `view_as_complex` + fp64 RoPE with explicit real arithmetic on MPS
- Replace MPS-incompatible `.double()` / `torch.float64` with fp32
- Use `psutil` for memory reporting instead of `torch.cuda.memory_stats`
- Stub training-only deps (`decord`, `x_clip_loss`) that aren't on PyPI for arm64

See [`PATCHES.md`](./PATCHES.md) for the full patch catalog with file:line references, and [`FINDINGS.md`](./FINDINGS.md) for what we learned about LongLive's quality ceiling during this work.

## Setup

```bash
# 1. Clone
git clone https://github.com/ConductorAILabs/longbridge.git
cd longbridge

# 2. Python 3.12 venv (3.14 has no PyTorch 2.12 wheels yet)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Download model weights (~32GB total)
.venv/bin/huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir ref/wan_models/Wan2.2-TI2V-5B
.venv/bin/huggingface-cli download Efficient-Large-Model/LongLive-2.0-5B \
  --local-dir ref/longlive_models/LongLive-2.0-5B
```

If the xet CDN throttles (downloads stalling at 0 MB/s), disable it:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/huggingface-cli download ...
```

## Inference

```bash
cd ref

# Generate. LONGBRIDGE_VAE_FP32=1 forces fp32 VAE decode — fights the
# bf16 chroma noise that MPS introduces in the high-compression Wan 2.2 VAE.
# Costs ~1.5GB extra RAM, so drop to 512x896 or 640x1152 res unless you
# have headroom at 704x1280.
LONGBRIDGE_VAE_FP32=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. \
  ../.venv/bin/python inference.py \
  --config_path configs/inference_mac.yaml

# Post-process: lock background to temporal median (kills inter-frame wall
# flicker, keeps subject motion). Usually drops "noise" complaint to zero.
../.venv/bin/python scripts/static_bg.py \
  videos/mac_test/rank0-*.mp4 \
  videos/mac_test/clean.mp4 \
  --threshold 25
```

Output lands in `ref/videos/mac_test/`.

## What works, what doesn't

**Works:**
- bf16 inference at native 704×1280 resolution
- ~3 min per 1.2-second clip on M5 Max (8 latent frames, 4 denoising steps)
- All LongLive 2.0 mechanics: frame sink, KV-recache, causal AR

**Doesn't work:**
- NVFP4 quantization (Blackwell-only; skip via bf16 checkpoint)
- Sequence parallel inference (`inference_sp.py` — single-GPU only here)
- Streaming VAE / async VAE (the `cached_decode` path isn't implemented on `WanVAE_`)
- Long sequences at native res: 16+ frames at 704×1280 OOMs on 128GB
  (drop to ~512×896 for longer clips, or wait for VAE tiling)

## Resolution / memory ceiling

| Pixel res | Latent shape | Status on 128GB unified RAM |
|---|---|---|
| 352×640   | `[1, 8, 48, 22, 40]` | ✓ (~80s) |
| 512×896   | `[1, 8, 48, 32, 56]` | ✓ (~110s) |
| 640×1152  | `[1, 8, 48, 40, 72]` | ✓ (~180s) |
| 704×1280  | `[1, 8, 48, 44, 80]` | ✓ (~190s) — native, recommended |
| 704×1280 × 16 frames | `[1, 16, 48, 44, 80]` | ✗ OOM |

Latent H and W must be **even** — the patch embed expects it.

## Quality recipe (two big wins, in order)

### 1. Background-flicker post-process

The single biggest visual win. LongLive 2.0 / Wan 2.2 generate each frame
somewhat independently, so flat-color regions (walls, backdrops) get
slightly different speckle each frame — reads as constant background
flicker even when prompt and config are perfect. Foreground stays locked
because chrome reflections + water have strong anchoring features.

Fix: compute per-pixel temporal median across all frames, then for each
frame, blend toward median wherever motion magnitude is low. Wall snaps to
a single static value; subject motion passes through.

```bash
python scripts/static_bg.py raw.mp4 clean.mp4 --threshold 25
```

Typical result: 85-97% of every frame locks to median. Subject still moves.
No model re-run.

### 2. Prompt structure

After sweeping dozens of config combinations, the **single biggest quality
fix is prompt structure**, not any hyperparameter:

| | Short terse prompt | Long structured prompt |
|---|---|---|
| Background | Noisy chaotic dark fill | Clean uniform gray |
| Subject | Hard to read | Crisp chrome reflections |
| Frame consistency | Drift | Stable across frames |

The model fills unspecified background space with hallucinated detail — exactly
what reads as "noise." Tell it explicitly what the background should look like
("soft gray seamless backdrop", "uniform, out of focus") and the noise goes
away.

See [`test_prompts/long_structured.txt`](./ref/test_prompts/long_structured.txt)
for the proven recipe (~130 words, present tense, single subject, explicit
background descriptors, explicit camera direction).

Also confirmed: `multi_shot_sink: true` in the config (upstream issue #20 / PR
#21 fix). The provided `configs/inference_mac.yaml` has both right.

See [`FINDINGS.md`](./FINDINGS.md) for the full research log of every knob
we tried and what each one did (or didn't) do.

## License & attribution

This repository carries **two licenses**, both requiring attribution:

| Component | License |
|---|---|
| Upstream NVlabs/LongLive + Wan 2.2 code in `ref/` | Apache 2.0 |
| **Conductor AI Labs patches + additions** (every `# Mac/MPS bridge:` line, `ref/scripts/`, `ref/configs/inference_mac.yaml`, `ref/test_prompts/long_structured.txt`, `ref/sweep_prompt.sh`, `ref/_sweep_lib.sh`, all docs) | **CC BY 4.0** |

You may use, share, adapt — including commercially — **provided you credit Conductor AI Labs** for the Apple Silicon bridge work.

**Required attribution** (or substantially equivalent):

> Apple Silicon (MPS) bridge for NVIDIA LongLive 2.0 by **[Conductor AI Labs](https://www.conductorailabs.com)** · Source: https://github.com/ConductorAILabs/longbridge · CC BY 4.0

Place this in your README or NOTICE file when redistributing. For papers, blog posts, demos, or videos that describe a derived work, include an equivalent credit line.

**Model weights** (downloaded separately):
- Wan 2.2 TI2V-5B base: Apache 2.0
- LongLive-2.0-5B fine-tune: CC BY-NC-SA 4.0 (research only — non-commercial)

For commercial output, use Wan 2.2 base alone (mlx-video supports it natively); the LongLive fine-tune is research-only by NVIDIA's license.

### Cite

```bibtex
@misc{longbridge2026,
  title  = {longbridge: NVIDIA LongLive 2.0 on Apple Silicon},
  author = {Conductor AI Labs},
  year   = {2026},
  howpublished = {\url{https://github.com/ConductorAILabs/longbridge}},
  note   = {CC BY 4.0},
}
```

## Acknowledgements

- Upstream model + reference code: [NVlabs/LongLive](https://github.com/NVlabs/LongLive)
- Base diffusion model: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- Patching playbook borrowed from prior MPS bridge work on SANA-WM
