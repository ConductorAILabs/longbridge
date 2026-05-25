# Patch catalog — LongLive 2.0 5B on Apple Silicon / MPS

Every patch in this list is already applied to `ref/`. This document exists so
the bridge can be re-derived if NVlabs updates upstream and we need to rebase.

## Categories

1. **Import-blocking** — fail at module load on CUDA-less boxes
2. **Device-picking** — replace hardcoded `cuda` paths with `cuda → mps → cpu`
3. **MPS-incompatible ops** — fp64, complex128, view_as_complex
4. **Missing dependencies** — newer transformers / torchvision / no-arm64-wheel decord
5. **Runtime CUDA-only** — `torch.cuda.*` calls + flash-attn

## 1. `utils/memory.py:8-15` — CUDA-at-import guard

`gpu = torch.device(f'cuda:{torch.cuda.current_device()}')` runs at import time
and crashes on CUDA-less boxes. Wrapped with `is_available()` check, falls back
to MPS then CPU.

## 2. `utils/memory.py:78-117` — Memory queries

`get_cuda_free_memory_gb()` calls `torch.cuda.memory_stats()` which returns an
empty dict on MPS, then `KeyError 'active_bytes.all.current'`. Returns
`psutil.virtual_memory().available` on non-CUDA backends.

`log_gpu_memory()` similarly: `torch.cuda.get_device_properties` is CUDA-only —
use `psutil.virtual_memory().total` on MPS/CPU.

`move_model_to_device_with_memory_preservation` — gate `torch.cuda.empty_cache()`
on CUDA availability.

## 3. `inference.py:8-17` — torchvision write_video shim

`torchvision.io.write_video` removed in torchvision 0.27. Added try/except
fallback to `imageio.v3.imwrite(... fps=..., codec='libx264')`.

## 4. `inference.py:228-247` — Device picking + skip NCCL

- Added `_pick_device()` helper (cuda → mps → cpu).
- Distributed init block now gated on `"LOCAL_RANK" in os.environ AND
  torch.cuda.is_available()` — single-device path runs on MPS without NCCL.
- Single-device fallback uses `_pick_device()` instead of hard `torch.device("cuda")`.

## 5. `wan_5b/modules/t5.py:472-487` — T5EncoderModel default arg

`def __init__(..., device=torch.cuda.current_device(), ...)` evaluates at
class-definition time and crashes import. Changed to `device=None` with
device-pick logic in body.

## 6. `wan_5b/modules/causal_model.py:5-11` — `x_clip_loss` import

`from transformers.models.x_clip.modeling_x_clip import x_clip_loss` — removed
in newer transformers, was only used in training paths. Wrapped in try/except
with a NotImplementedError stub.

## 7. `utils/dataset.py:14-19` — Optional decord

`import decord` — no macOS arm64 wheel. Only used for training-time video
dataset loading. Wrapped in try/except so inference path doesn't need it.

## 8. `pipeline/self_forcing_training.py:6-15` — write_video shim

Same imageio fallback as `inference.py`. Pipeline `__init__.py` imports both
inference and training pipelines, so even inference-only callers hit this
import.

## 9. `utils/wan_5b_wrapper.py:31-40` — WanTextEncoder MPS move

`if torch.cuda.is_available(): self.text_encoder = self.text_encoder.cuda()`
left the encoder on CPU when MPS was the target. Added MPS branch:
`elif torch.backends.mps.is_available(): self.text_encoder.to('mps')`.

## 10. `utils/wan_5b_wrapper.py:42-53` — `device` property

Was hard-coded `torch.cuda.current_device()`. Now reads
`next(self.text_encoder.parameters()).device` to follow the actual model.

## 11. `utils/wan_5b_wrapper.py:370-376, 397-403` — fp32 not fp64

Two `.double().to(...)` calls in `_convert_flow_pred_to_x0` and
`_convert_x0_to_flow_pred`. MPS doesn't support fp64. Switch to fp32 on MPS.

## 12. `utils/scheduler.py:38-44, 65-72, 95-104` — fp32 schedulers

Three `.double()` calls in the noise / flow / velocity converters. Same fix:
`hi_dtype = float32 if device.type == 'mps' else float64`.

## 13. `utils/position_embedding_utils.py:80-83` — RoPE freqs fp32

`torch.angle(freqs_t[1]).to(torch.float64)` and `torch.arange(..., dtype=float64)`
both fail on MPS. Use fp32 — precision loss is negligible for sinusoidal time
embeddings at our sequence lengths.

## 14. `model/base.py:458-470` — Scheduler timesteps `.cuda()` → `.to(device)`

Three `.cuda()` calls on scheduler timesteps. Replaced with `.to(_d)` where `_d`
is `unipc_timesteps.device`.

## 15. `wan_5b/modules/causal_model.py:1327-1336` — fp64/complex128 freqs on MPS

`self.freqs.to(device)` failed when moving complex128 to MPS. On MPS: downcast
complex128 → complex64 and fp64 → fp32 before the device move.

## 16. `wan_5b/modules/causal_model.py:43-100` — `causal_rope_apply` real math

`torch.view_as_complex(x.to(torch.float64).reshape(...))` — both fp64 AND
view_as_complex are MPS-incompatible. Added explicit real-arithmetic branch
for MPS: split freqs into (real, imag), compute
`(a+ib)(c+id) = (ac-bd) + i(ad+bc)` in fp32, stack & flatten. CUDA path preserved.

## 17. `wan_5b/modules/model.py:15-22` — `sinusoidal_embedding_1d` fp32

`position.type(torch.float64)` → gated to fp32 on MPS. Precision loss is
negligible for sinusoidal time embeddings.

## 18. `wan_5b/modules/attention.py:51-72` — flash_attention SDPA fallback

`flash_attention()` is called directly from `model.py` cross-attention and
`causal_model.py:149`, bypassing the `attention()` dispatcher's SDPA fallback.
When neither FA2 nor FA3 is installed, route to
`scaled_dot_product_attention` at the top of `flash_attention()`. CUDA-device
assertion is gated on `torch.cuda.is_available()`.

## Known unpatched (not exercised on inference path)

- `utils/wan_5b_wrapper.py:265` — `torch.cuda.empty_cache()`. Will silently
  fail on MPS but only logs a warning since it's wrapped in a `low_memory`
  branch we don't enter.
- `pipeline/causal_diffusion_inference.py:417,556,560,568,583` — CUDA streams
  for async/streaming VAE. Disabled via `streaming_vae: false` in config.
- `model/dmd.py:218-222` — `.double()` in DMD distillation loss. Training-only.
- NVFP4 paths in `fouroversix/` — only activated when `model_quant=True`.

## Run command

```bash
cd ref
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. ../.venv/bin/python inference.py \
  --config_path configs/inference_mac.yaml
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is required — a small number of ops silently
fall back to CPU. Without the env, those raise NotImplementedError.

## Lineage

This bridge derives from a similar playbook used to run SANA-WM on MPS. The
overall pattern: keep upstream code intact, stub CUDA-only deps at import
boundaries, replace MPS-incompatible ops in-place, wrap with subprocess.
