# Findings — LongLive 2.0 5B output quality on Apple Silicon

This document records what we learned about LongLive 2.0's quality ceiling
while porting it to MPS. Goal: save the next person time when they hit the
same "noisy background" complaint.

## TL;DR

**Prompt structure is the biggest quality lever, not any hyperparameter.**

After sweeping ~20 config combinations chasing "background noise" complaints,
we found the actual cause: the model fills unspecified background space with
hallucinated detail. Tell it explicitly what the background should look like
(e.g., "soft gray seamless backdrop, uniform, out of focus") and the "noise"
goes away. See `test_prompts/long_structured.txt` for the working recipe.

Beyond prompt structure:

LongLive 2.0 5B is built for **speed and long-form interactivity**, not photorealism.
Its quality ceiling is roughly Wan 2.1 1.3B-tier (the base model lineage), and that
ceiling is the same on CUDA H100s as it is on MPS — verified by:

- The paper's own ablations (Tables 4, 7) showing 4-step BF16 is the published
  recipe at VBench 85.06 (essentially tied with raw Wan 2.1 1.3B at 84.87)
- Upstream issue #51 where authors can't reproduce paper Table 1 scores even on
  CUDA
- Community signal: zero ComfyUI nodes, zero "wow" posts in 7 months, near-zero
  Reddit/Twitter/Discord discussion

If your bar is sharp photorealistic single shots, look at Wan 2.2 base alone at
40 UniPC steps (mlx-video supports it natively). LongLive's value is multi-shot
narrative + minute-plus runtime + 20 FPS interactivity.

## The fix: structured prompts with explicit background

Short prompt ("close-up macro of water dripping from chrome faucet"):
- Model hallucinates a chaotic dark background → reads as "noise"
- Frame-to-frame the noise drifts → reads as "jittery"
- All sweeping in this state did nothing

Long structured prompt (~130 words, explicit camera move + background +
lighting + composition rules, "no human hands or other objects"):
- Background renders as clean uniform gray exactly as described
- Subject is crisp, frame-to-frame stable
- No post-process denoise needed

Why: distilled few-step diffusion has no spare capacity to invent backgrounds.
It commits early. Underspecified prompts → it commits to noise. Overspecified
prompts → it commits to the structure you described. Same model, same weights,
dramatic quality difference.

## What we swept (none of it moved the noise needle visibly)

| Knob | Range tested | Result |
|---|---|---|
| `sampling_steps` | 4 → 12 | no diff (model is 4-step distilled) |
| `guidance_scale` | 1.0 → 3.0 | >1.5 burns out (CFG amplifies on distilled) |
| `negative_prompt` | empty / clean / structured | no observable diff |
| `timestep_shift` | 3.0 → 5.0 | no diff (5.0 is the trained value) |
| `sink_size` | 4, 8 | no diff at single-block inference |
| `local_attn_size` | 16, 32 | no diff |
| `inference_t_scale` | 1.0, 1.1 | no diff |
| `multi_shot_sink` | false → true | **the fix that mattered for spec correctness** (issue #20) — but no visible diff at single-block |
| Resolution | 352×640 → 704×1280 native | sharper proportionally, expected |

## The one config change everyone should make

Upstream NVlabs/LongLive issue #20 + PR #21 documents that
`multi_shot_sink: false` (the prior default) corrupts the KV cache sink
region permanently across frames — exactly the "noisy background drift"
pattern people complain about. Maintainer `AndysonYs` flipped the default
to `true` and confirmed: *"Global_sink == True leads to better consistency."*

The provided `configs/inference_mac.yaml` has this set correctly. If you fork
this repo or write your own config, **keep `multi_shot_sink: true` and
`multi_shot_rope_offset: 8`.**

## Research path summary

We ran three parallel research agents to confirm no hidden quality knob exists:

1. **Paper deep-dive** (LongLive 2.0 arXiv 2605.18739, LongLive 1.0
   arXiv 2509.22622): the recipe is fully public. There is no separate "HQ
   mode." The 24.8 FPS figure IS the high-quality preset. No `cfg_rescale`,
   no two-stage refinement, no overshoot scheduling, no post-process. Authors'
   only quality advice in the docs is **prompt hygiene** (~300 tokens/shot,
   present tense, single subject/location, repeat background descriptors).

2. **Community scan** (Reddit, Twitter, HuggingFace, ComfyUI marketplace):
   essentially no community signal. Zero ComfyUI custom nodes (huge red flag
   for a 7-month-old model). The only working community recipe is
   [Daydream Scope](https://github.com/daydreamlive/scope) — they confirm
   832×480 train resolution for 1.3B, long detailed prompts mandatory, and
   describe LongLive's quality as fundamentally Wan 2.1 1.3B-tier.

3. **GitHub repo audit** (NVlabs/LongLive issues + PRs + commits, last 30
   days): no Mac/MPS discussion exists. The only relevant fix is the
   `multi_shot_sink` flag flip above. fouroversix's "PyTorch reference"
   backend is documented as "should not be used in real-world use cases" —
   only NVFP4 + Triton paths are tuned for quality on Blackwell.

## Possible improvements (not yet tried)

These are speculative and may or may not help:

1. **MPS attention softmax fp32 accumulation** — MPS historically accumulates
   bf16 matmul outputs in fp16 for some ops. Forcing fp32 accumulation in
   attention softmax could reduce noise. Untested.

2. **`mg_lightvae_v2` VAE** — different decoder. Weights are not on HF; would
   need to find a published mirror.

3. **Wan 2.2 base alone at 40 steps via mlx-video** — skip the LongLive
   fine-tune entirely. mlx-video supports Wan 2.2 TI2V-5B natively. Much
   slower per clip but visibly cleaner per the Scope team's experience.

4. **Post-process ffmpeg `hqdn3d` denoise** — won't add detail but cleans up
   the worst frame-to-frame chroma noise. Cheap, fast, no model change.

## What we'd recommend for production

For Conductor AI Labs' own work:

- **Real product shots:** Wan 2.2 base @ 40 steps + Real-ESRGAN upscale,
  not LongLive
- **Long-form narrative B-roll where coherence beats fidelity:** LongLive 2.0
  5B with the corrected config in this repo
- **Speed-critical interactive use cases:** LongLive 2.0 5B NVFP4 on a
  Blackwell card (not Mac — Mac path is research-only)

This bridge stays useful as research-grade Mac infra and as a working
playbook for porting other CUDA+PyTorch video models to MPS.
