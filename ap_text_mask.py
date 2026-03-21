"""AP_TextPromptMask_V3 — numpy-2-safe text-prompted mask generation.

Uses CLIPSeg via HuggingFace transformers.  All image I/O goes through
torch/PIL to avoid numpy entirely, so this works with numpy 2.x.

Model auto-downloads to the HuggingFace hub cache on first use (~350 MB for
rd64-refined, ~100 MB for rd16).  Models are cached in GPU/CPU memory after
the first run to keep subsequent frames fast.

TensorFlow stub
---------------
transformers ≥ 4.50 lazily imports image_transforms.py which does
``import tensorflow as tf``.  On systems where TF is built against the old
numpy C-ABI (numpy < 2 ABI) this raises a SystemError at import time even
though CLIPSeg never uses TensorFlow at runtime.  _ensure_tensorflow_importable()
pre-empts the crash by inserting an empty stub module into sys.modules before
transformers is touched.
"""

import sys
import types
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# TensorFlow compatibility shim (must run before any transformers import)
# ---------------------------------------------------------------------------

def _ensure_tensorflow_importable() -> None:
    """
    TensorFlow compiled for numpy <2 raises SystemError when imported under
    numpy 2.x.  transformers 4.50+ unconditionally imports tensorflow from
    image_transforms.py even when only CLIPSeg (pure PyTorch) is used.

    If TF is broken or absent we insert a minimal stub so:
      1. The import chain completes without a C-extension crash.
      2. Runtime calls like isinstance(x, tf.Tensor) return False correctly.
    CLIPSeg inference never calls any real TF functions.
    """
    if "tensorflow" in sys.modules:
        # Already loaded — check it has the minimum attrs transformers needs.
        tf = sys.modules["tensorflow"]
        if not hasattr(tf, "Tensor"):
            tf.Tensor = type("Tensor", (), {})  # dummy class
        return

    try:
        import tensorflow  # noqa: F401
        # Loaded OK — still ensure Tensor exists (shouldn't be needed, but safe).
        if not hasattr(tensorflow, "Tensor"):
            tensorflow.Tensor = type("Tensor", (), {})
    except Exception:
        stub = types.ModuleType("tensorflow")
        # Provide the attrs transformers' generic.py and image_transforms.py access.
        stub.Tensor = type("Tensor", (), {})
        stub.__version__ = "0.0.0"
        sys.modules["tensorflow"] = stub
        for _sub in (
            "tensorflow.python",
            "tensorflow.python.ops",
            "tensorflow.python.framework",
        ):
            sys.modules.setdefault(_sub, types.ModuleType(_sub))


_ensure_tensorflow_importable()


# ---------------------------------------------------------------------------
# Local helpers (previously imported from ap_temporal_common)
# ---------------------------------------------------------------------------

CATEGORY_TEMPORAL = "AP ClipSEG Light"


def _ensure_bhwc(t: torch.Tensor) -> torch.Tensor:
    """Ensure image tensor is [B, H, W, C].  ComfyUI already sends BHWC but
    guard against bare [H, W, C] inputs."""
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_CLIPSEG_MODEL_IDS = {
    "clipseg_rd64": "CIDAS/clipseg-rd64-refined",
    "clipseg_rd16": "CIDAS/clipseg-rd16",
}

# Global model cache keyed by (model_key, device_str).
_MODEL_CACHE: dict = {}


def _load_clipseg(model_key: str, device: torch.device):
    cache_key = (model_key, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' library is required for AP TextPrompt Mask.  "
            "Install it with:  pip install transformers"
        ) from exc

    model_id = _CLIPSEG_MODEL_IDS[model_key]
    print(f"[APTextMask] Loading CLIPSeg model '{model_id}' → {device} …")
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model = model.to(device).eval()

    _MODEL_CACHE[cache_key] = (processor, model)
    print(f"[APTextMask] Model ready.")
    return processor, model


def _resolve_device(device_pref: str, default_device: torch.device) -> torch.device:
    p = str(device_pref).strip().lower()
    if p == "auto":
        return default_device
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _frame_to_pil(frame_hwc: torch.Tensor) -> Image.Image:
    """Convert a single [H,W,C] float32 tensor [0,1] to PIL RGB.  Numpy-2 safe."""
    rgb = frame_hwc.clamp(0.0, 1.0).mul(255.0).byte().cpu()
    # torch → numpy → PIL; basic byte array ops work fine in numpy 2.
    return Image.fromarray(rgb.numpy(), mode="RGB")


@torch.no_grad()
def _run_clipseg_frame(
    pil_image: Image.Image,
    prompts: list[str],
    processor,
    model,
    device: torch.device,
) -> torch.Tensor:
    """
    Run CLIPSeg for one image and one or more prompts.
    Returns a [H_orig, W_orig] float32 logit-heatmap averaged across prompts.
    """
    W_orig, H_orig = pil_image.size

    # Replicate the image once per prompt so the processor sees paired inputs.
    images = [pil_image] * len(prompts)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # [num_prompts, H_logit, W_logit]

    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    # Average across prompts.
    heatmap = logits.float().mean(dim=0)  # [H_logit, W_logit]

    # Resize to original image size.
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(H_orig, W_orig),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    return heatmap


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """Normalise logits to [0,1] per-frame using sigmoid."""
    return torch.sigmoid(heatmap)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AP_TextPromptMask_V3:
    DESCRIPTION = (
        "Generates a segmentation mask from a text prompt using CLIPSeg.\n\n"
        "Works with numpy 2.x.\n\n"
        "prompt:\n"
        "  Main text description of the region to segment, e.g. 'face' or 'person'.\n"
        "  Separate multiple prompts with '|' to average their heatmaps:\n"
        "  e.g.  'face | head | eyes'\n\n"
        "threshold:\n"
        "  Sigmoid-normalised heatmap value above which a pixel is considered foreground.\n"
        "  Lower = larger mask, higher = tighter mask.\n\n"
        "smooth_radius:\n"
        "  Gaussian blur applied to the heatmap before thresholding.  Helps produce\n"
        "  cleaner soft edges.  0 = no smoothing.\n\n"
        "model:\n"
        "  clipseg_rd64  – higher quality, ~350 MB download, slower\n"
        "  clipseg_rd16  – faster, ~100 MB, slightly lower quality\n\n"
        "device:\n"
        "  auto = use cuda when available, else cpu.  Models are cached in memory so\n"
        "  subsequent frames do not reload weights.\n\n"
        "unload_after_run:\n"
        "  Free the model from memory after each execution.  Enable to save VRAM when\n"
        "  this node is not used every frame."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "face",
                        "multiline": False,
                        "tooltip": "Text description of the region to mask. Use '|' to separate multiple prompts.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.40,
                        "min": 0.01,
                        "max": 0.99,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Sigmoid heatmap threshold. Lower → larger mask, higher → tighter mask.",
                    },
                ),
                "smooth_radius": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Gaussian blur radius applied to the heatmap before thresholding.",
                    },
                ),
                "soft_mask": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True  = output the smoothed sigmoid heatmap (soft/feathered edges).\n"
                            "False = hard binary mask at the threshold value."
                        ),
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the output mask.",
                    },
                ),
                "model": (
                    list(_CLIPSEG_MODEL_IDS.keys()),
                    {
                        "default": "clipseg_rd64",
                        "tooltip": "CLIPSeg variant. rd64 = better quality, rd16 = faster.",
                    },
                ),
                "mask_dilate": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Dilate (expand) the final mask by this many pixels. 0 = off.",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Gaussian blur applied to the final mask. 0 = off.",
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device. 'auto' uses CUDA when available.",
                    },
                ),
            },
            "optional": {
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload the model from memory after each run to free VRAM.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    CATEGORY = CATEGORY_TEMPORAL

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run (prompt / threshold can change without tensor changes).
        return float("nan")

    def generate_mask(
        self,
        image,
        prompt,
        threshold=0.40,
        smooth_radius=4,
        soft_mask=True,
        invert=False,
        mask_dilate=0,
        mask_blur=0,
        model="clipseg_rd64",
        device="auto",
        unload_after_run=False,
    ):
        imgs = _ensure_bhwc(image).float()  # [B, H, W, C]
        B, H, W, _ = imgs.shape

        # Parse prompts.
        raw_prompts = [p.strip() for p in str(prompt).split("|") if p.strip()]
        if not raw_prompts:
            raw_prompts = ["object"]

        # Resolve compute device — default to the image tensor's device when auto.
        default_dev = imgs.device
        compute_dev = _resolve_device(device, default_dev)

        processor, seg_model = _load_clipseg(str(model), compute_dev)

        masks_out = []
        for b in range(B):
            pil_img = _frame_to_pil(imgs[b])  # PIL RGB

            heatmap = _run_clipseg_frame(pil_img, raw_prompts, processor, seg_model, compute_dev)
            heatmap = heatmap.to(dtype=torch.float32)

            # Optional Gaussian smoothing.
            if int(smooth_radius) > 0:
                r = int(smooth_radius)
                k = 2 * r + 1
                # Build a simple box kernel as a cheap / dependency-free Gaussian approx.
                # For a real Gaussian, we use successive box blurs (3 passes ≈ Gaussian).
                kernel = torch.ones(1, 1, k, k, device=heatmap.device, dtype=torch.float32) / (k * k)
                pad = r
                hm = heatmap.unsqueeze(0).unsqueeze(0)
                for _ in range(3):  # 3 box-blur passes ≈ Gaussian
                    hm = F.conv2d(F.pad(hm, [pad, pad, pad, pad], mode="reflect"), kernel)
                heatmap = hm.squeeze(0).squeeze(0)

            prob = _normalize_heatmap(heatmap)  # [0, 1]

            if bool(soft_mask):
                # Remap: prob=threshold → 0, prob=1 → 1; below threshold is black (0).
                t = float(threshold)
                mask = ((prob - t) / (1.0 - t + 1e-6)).clamp(0.0, 1.0)
            else:
                mask = (prob >= float(threshold)).float()

            if bool(invert):
                mask = 1.0 - mask

            # Dilate (expand) the mask.
            if int(mask_dilate) > 0:
                d = int(mask_dilate)
                k = 2 * d + 1
                m = mask.unsqueeze(0).unsqueeze(0)
                m = F.max_pool2d(m, kernel_size=k, stride=1, padding=d)
                mask = m.squeeze(0).squeeze(0)

            # Blur the final mask.
            if int(mask_blur) > 0:
                r = int(mask_blur)
                k = 2 * r + 1
                blur_k = torch.ones(1, 1, k, k, device=mask.device, dtype=torch.float32) / (k * k)
                m = mask.unsqueeze(0).unsqueeze(0)
                for _ in range(3):  # 3 box-blur passes ≈ Gaussian
                    m = F.conv2d(F.pad(m, [r, r, r, r], mode="reflect"), blur_k)
                mask = m.squeeze(0).squeeze(0).clamp(0.0, 1.0)

            masks_out.append(mask)

        result = torch.stack(masks_out, dim=0)  # [B, H, W]

        if bool(unload_after_run):
            cache_key = (str(model), str(compute_dev))
            _MODEL_CACHE.pop(cache_key, None)
            del seg_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (result,)


NODE_CLASS_MAPPINGS = {
    "AP_CLIPSeg_TextMask": AP_TextPromptMask_V3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AP_CLIPSeg_TextMask": "◧ AP CLIPSeg Text Mask",
}
