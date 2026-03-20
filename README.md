# AP ClipSEG Light

A lightweight ComfyUI custom node pack that generates segmentation masks from plain-text prompts using [CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg).

## Features

- **Text-driven masking** — describe what you want masked in plain language (e.g. `face`, `sky`, `person`)
- **Multi-prompt averaging** — separate prompts with `|` to blend multiple heatmaps: `face | head | eyes`
- **Soft or hard masks** — output a feathered sigmoid heatmap or a crisp binary mask
- **numpy 2.x safe** — all image I/O goes through torch/PIL, no numpy array ops
- **Auto model caching** — models are kept in GPU/CPU memory after the first load; no reload cost on subsequent runs
- **numpy 2 / TF crash fix** — automatically stubs broken TensorFlow imports caused by transformers ≥ 4.50 on numpy 2.x systems

## Requirements

| Package | Minimum version |
|---|---|
| `torch` | any recent |
| `torchvision` | ≥ 0.15 |
| `transformers` | any recent |
| `Pillow` | any recent |

Models are downloaded automatically from the HuggingFace Hub on first use:

| Model | Size | Quality |
|---|---|---|
| `clipseg_rd64` (default) | ~350 MB | Higher |
| `clipseg_rd16` | ~100 MB | Faster |

## Installation

```bash
# Option A – ComfyUI Manager (search "AP ClipSEG Light")
# Option B – manual
git clone https://github.com/adampolczynski/ComfyUI_AP_ClipSEG_Light \
  ComfyUI/custom_nodes/AP_ClipSEG_Light
pip install transformers Pillow
```

## Node: AP Text Prompt Mask V3

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image batch |
| `prompt` | STRING | Text description of the region to mask. Use `\|` to separate multiple prompts |
| `threshold` | FLOAT (0.01–0.99) | Heatmap threshold. Lower → larger mask, higher → tighter mask |
| `smooth_radius` | INT (0–32) | Gaussian blur radius applied before thresholding |
| `soft_mask` | BOOLEAN | `True` = feathered sigmoid output, `False` = binary mask |
| `invert` | BOOLEAN | Invert the output mask |
| `model` | ENUM | `clipseg_rd64` or `clipseg_rd16` |
| `device` | ENUM | `auto` / `cuda` / `cpu` |
| `unload_after_run` | BOOLEAN | Free model VRAM after each run |

**Output:** `MASK` — float32 tensor `[B, H, W]` in `[0, 1]`

## License

MIT — see [LICENSE](LICENSE).
# ComfyUI_AP_ClipSEG_Light
