# ComfyUI Sol-Attn (SM120 / RTX 5090) — MiniMax H3

Sol-Attn (Sparsified On-the-fly Attention) sparse-attention acceleration, built specifically for **MiniMax H3 video generation** on **NVIDIA RTX 5090 (SM120)**.

This plugin performs Sol-Attn's sparse routing through PyTorch's `flex_attention` (which lowers to FlashAttention-3-style kernels via `torch.compile`), giving significant speedups over standard SDPA on long sequences.

> **v2 (2026-08)** — two new quality-preserving options, guided by recommendations from [Kijai](https://github.com/kijai) (see [Credits](#credits)):
> - **`dense_steps` / `step_off`** — run the last denoising step(s) fully dense, avoiding the worst of the noise at the end of sampling.
> - **`exact_kv` / `exact_kv_and_rows`** — MiniMax-H3 only; keep the packed prefix (text/cond/reference/audio) rows exact.

## Features

- **SM120 native optimization**: compiled `flex_attention` kernel — no serial loop, no OOM
- **Real speedup**: measured faster than SDPA on long sequences (>8k tokens)

| Sequence | SDPA | This plugin | Speedup | Density |
|---------|------|-------------|---------|---------|
| 4096  | 2.35ms | 1.61ms | 1.46x | 0.15 |
| 8192  | 9.24ms | 2.35ms | 3.93x | 0.11 |
| 16384 | 36.47ms | 5.28ms | 6.91x | 0.09 |
| 32768 | 144.5ms | 16.06ms | 9.00x | 0.08 |

(MiniMax H3's real config: H=56 heads, bfloat16, BTHD layout, post-warmup)

- **Automatic fallback**: `flex_attention` → Triton reference kernel → standard SDPA; failures never interrupt generation
- **Zero compile-time dependencies**: pure Python + torch for SM120 — no CUTLASS / CuTe DSL required

## Platform & GPU Compatibility

- **Cross-platform**: the code is OS-agnostic (no hardcoded Linux paths). Works on **Windows**, Linux, and macOS.
- **GPU**: acceleration requires **RTX 5090 (SM120)** or another Blackwell consumer GPU.
  - Other GPUs (H100/SM90, B200/SM100, RTX 40/30-series) still run correctly but **fall back to SDPA** (no speedup) — this release ships only the SM120 backend.
- **PyTorch ≥ 2.6** is required for full `flex_attention` performance (tested on 2.11.0+cu130).
- **Note for Windows users**: the bundled `inductor_fix.py` only acts when it detects a stale torch 2.11 install bug; on a normal install it is a no-op.

## Requirements

- NVIDIA RTX 5090 (SM120) or other Blackwell consumer GPU
- **PyTorch ≥ 2.6** (tested on 2.11.0+cu130)
- CUDA 12.x / 13.x
- ComfyUI

## Installation

1. Place this folder in ComfyUI's custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/KingGore/ComfyUI_sol-attn_Blackwell.git
```

Or download the whole folder and place it under `ComfyUI/custom_nodes/` (the folder should be named `ComfyUI_sol-attn_Blackwell`).

2. Restart ComfyUI (no pip install needed — dependencies are bundled with torch).

On startup the log should show:

```
[Sol-Attn] ✓ GPU: NVIDIA GeForce RTX 5090 (SM120) — Sol-Attn via flex_attention (Python) path
[SolAttnFlex] flex_attention kernel compiled (warmup done)
```

## Usage

Add the node **Sol-Attn MiniMax H3 Patcher 🚀** to your MiniMax H3 workflow:

```
[Load Diffusion Model] → [Sol-Attn MiniMaxH3 Patcher] → [KSampler / sampler node]
```

Node parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable / disable |
| `tau` | 1.0 | Sparsity temperature; larger = skip more KV blocks (faster but less accurate). Recommended 0.8–1.5 |
| `thresh_type` | diag | Threshold estimation; `diag` is faster, `exact` is more precise |
| `exact_mode` | off | **MiniMax-H3 only.** `off` / `exact_kv` / `exact_kv_and_rows`. Keep the packed prefix (text / cond / reference / audio rows) exact — see below. No effect on other models. |
| `dense_steps` | 0 | Run the **last N** denoising steps fully dense (no sparse). The final steps carry the least noise, so their sparse error is the most visible. |
| `step_off` | 0.0 | Alternative to `dense_steps`: run the last *fraction* of steps dense by schedule position (e.g. `0.5` = last half). |
| `sink_tokens` | 0 | Exact prefix token count used by `exact_mode`. `0` auto-derives it from the H3 packed layout (the video segment start). Usually leave at 0. |

### The two new quality features (v2)

#### 1. Last steps dense — `dense_steps` / `step_off`

Sol-Attn's sparse error is concentrated in the *early* denoising steps (high noise overrides the detail). At the very end of sampling the noise is weakest, so a sparse approximation there is the most visible. Setting `dense_steps=1` (or `step_off=1.0`) runs the final step(s) with full, exact attention — removing the last of the clamping noise at a tiny cost.

#### 2. Exact KV sink — `exact_mode` (MiniMax-H3 only)

MiniMax H3's packed sequence is `[text | conditioned rows | audio | video]`. The rows before the video are the *conditioning* the model must honor faithfully — and the generated **audio stream** lives at the end of that prefix. Two levels:

| `exact_mode` | Cost | Effect |
|--------------|------|--------|
| `exact_kv` | ~+3% | Every query attends to the prefix KV rows exactly (no sparsification on the conditioning). |
| `exact_kv_and_rows` | ~+20% | Also runs the prefix *query* rows dense, making the generated audio stream exact. |

> **Note:** `exact_mode` applies only to the packed sequence. The model's tiny token-refiner attention (text-only) stays sparse. Other models are unaffected — the override is MiniMax-H3 specific.

### Recommended starting points

- Max quality for the extra ~20%: `exact_mode=exact_kv_and_rows`, `dense_steps=1`
- Cheap quality boost: `exact_mode=exact_kv`, `dense_steps=1`
- Baseline (pure v1 behavior): `exact_mode=off`, `dense_steps=0`

## How It Works

Sol-Attn's core idea: use each KV block's mean vector `kc` for a cheap routing pass, identify the "important" KV blocks, and only compute exact attention over those.

This plugin computes the routing decision at 128-token granularity in pure torch, then executes the selected blocks with `flex_attention`'s compiled kernel. Compared to the original Sol-Attn Triton reference implementation (which processes each KV block serially — 3–17x slower and sometimes OOM), this approach achieves full FlashAttention-3 performance on SM120.

**v2 routing changes:** the mask builder now takes an optional *exact KV sink* — a prefix of the sequence whose KV blocks are always selected for every query (and, with `exact_kv_and_rows`, whose query rows run fully dense). This is how the conditioning and audio rows stay exact while the large video portion keeps its sparse speedup.

## File Structure

```
ComfyUI_sol-attn_Blackwell/
├── __init__.py          # Plugin entry: inductor fix + flex warmup + node registration
├── inductor_fix.py      # Fixes a torch 2.11 torch.compile crash (duplicate template)
├── sol_attn_flex.py     # SM120 main backend (flex_attention + Sol-Attn routing + exact sink)
├── minimax_h3_patch.py  # MiniMax H3 optimized_attention_override hook + step/sink wiring
├── sol_attn_node.py     # ComfyUI nodes (MiniMax H3 Patcher with exact_mode / dense_steps / step_off)
├── sol_attn_loader.py   # Loader (with fallback)
└── sol_attn/
    ├── __init__.py
    ├── interface.py
    ├── preprocess.py
    └── triton_ref/
        └── fwd.py       # Fallback backend (Triton reference)
```

## Known Limitations

- This release targets **SM120 (RTX 5090)** and **MiniMax H3**. The H100 (SM90) / B200 (SM100) CuTe DSL backends are not included.
- Only MiniMax H3 self-attention is intercepted (`skip_reshape=True` + head_dim=128 + bfloat16). Cross-attention and other dtypes fall back automatically.
- The first generation has a one-time `torch.compile` latency (pre-warmed at plugin startup, ~3–4s).
- `exact_kv_and_rows` costs ~+20% over the sparse baseline; on very long sequences watch VRAM (the dense prefix rows are bounded by the prefix size, not the video length).

## Credits

Big thanks to **[@Kijai](https://github.com/kijai)** for the two design suggestions that shaped [v2](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell/releases):

1. **"Do the last step or a few last steps without it"** — the insight that the final denoising steps carry the least noise, so their sparse error is the most visible — implemented here as `dense_steps` / `step_off`.
2. **MiniMax-H3 only `exact_kv` / `exact_kv_and_rows`** — keeping the packed conditioning/audio rows exact — implemented as the `exact_mode` option.

## References

- [Sol-Attn paper](https://arxiv.org/abs/2501.17209)
- [NVLabs/Sana source](https://github.com/NVlabs/Sana)
- [PyTorch flex_attention docs](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)