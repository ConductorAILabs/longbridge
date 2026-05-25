# longbridge

**Run NVIDIA LongLive 2.0 5B on Apple Silicon (Mac, MPS) — no CUDA, no NVFP4, no flash-attn.**

A PyTorch+MPS bridge for [NVlabs/LongLive 2.0](https://github.com/NVlabs/LongLive). 16 in-place patches to the upstream code that swap CUDA-only paths for Mac-friendly equivalents. The model runs natively in bf16 on Apple Silicon — base Wan 2.2 TI2V-5B + LongLive 2.0 fine-tune.

Verified working: M-series Macs with 96GB+ unified memory.

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

See [`PATCHES.md`](./PATCHES.md) for the full list with file:line references.

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
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
  --config_path configs/inference_mac.yaml
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

## Licensing

- **Code** (this repo + upstream LongLive 2.0): Apache 2.0
- **Model weights**:
  - Wan 2.2 TI2V-5B base: Apache 2.0
  - **LongLive-2.0-5B fine-tune: CC-BY-NC-SA 4.0** (non-commercial)

If you need commercial output, use the Wan 2.2 base alone (mlx-video has it natively).
The LongLive layer is research-only.

## Acknowledgements

- Upstream model + reference code: [NVlabs/LongLive](https://github.com/NVlabs/LongLive)
- Base diffusion model: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- Patching playbook borrowed from prior MPS bridge work on SANA-WM
