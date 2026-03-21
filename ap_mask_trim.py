"""AP_MaskTrim — trim a mask by percentage along X / Y axes.

crop_x:  positive → trim from right,  negative → trim from left.
crop_y:  positive → trim from bottom, negative → trim from top.
Values are percentages (-100..100).  The output mask has the same spatial
dimensions as the input; trimmed pixels fade to 0 over the feather zone.

auto_align: when True, the trim axes rotate to match the mask's principal
orientation (PCA major/minor axis of the masked region).  crop_y then trims
the "bottom" of the masked shape regardless of how it is rotated in the image.

Post-processing: optional dilate and blur applied to the trimmed mask.
"""

import math

import torch
import torch.nn.functional as F


def _trim_weight(size: int, cut_px: int, from_end: bool, feather_px: int,
                 device: torch.device) -> torch.Tensor:
    """Return a 1-D weight vector of length `size`.

    Pixels in the trimmed region are 0; the feather zone transitions linearly
    from 1 → 0 over `feather_px` pixels before the hard cut boundary.
    """
    w = torch.ones(size, device=device, dtype=torch.float32)
    if cut_px <= 0:
        return w

    cut_px = min(cut_px, size)
    feather_px = min(feather_px, cut_px)

    if from_end:
        # zero out [size-cut_px : size], feather before that
        zero_start = size - cut_px
        if feather_px > 0:
            fade_start = max(0, zero_start - feather_px)
            idx = torch.arange(fade_start, zero_start, device=device, dtype=torch.float32)
            w[fade_start:zero_start] = 1.0 - (idx - fade_start) / feather_px
        w[zero_start:] = 0.0
    else:
        # zero out [0 : cut_px], feather after that
        if feather_px > 0:
            fade_end = min(size, cut_px + feather_px)
            idx = torch.arange(cut_px, fade_end, device=device, dtype=torch.float32)
            w[cut_px:fade_end] = (idx - cut_px) / feather_px
        w[:cut_px] = 0.0

    return w


def _apply_trim(mask: torch.Tensor, crop_x: int, crop_y: int,
                feather: int) -> torch.Tensor:
    """Apply X/Y trim to a [B, H, W] mask."""
    B, H, W = mask.shape
    device = mask.device

    weight = torch.ones(H, W, device=device, dtype=torch.float32)

    if crop_x != 0:
        cut_px = int(abs(crop_x) / 100.0 * W + 0.5)
        wx = _trim_weight(W, cut_px, from_end=(crop_x > 0), feather_px=feather, device=device)
        weight = weight * wx.unsqueeze(0)       # [1, W] broadcast over H

    if crop_y != 0:
        cut_px = int(abs(crop_y) / 100.0 * H + 0.5)
        wy = _trim_weight(H, cut_px, from_end=(crop_y > 0), feather_px=feather, device=device)
        weight = weight * wy.unsqueeze(1)       # [H, 1] broadcast over W

    return (mask * weight.unsqueeze(0)).clamp(0.0, 1.0)


def _mask_centroid_and_angle(mask_2d: torch.Tensor):
    """Return (cy, cx, angle_rad) for the masked region.

    angle_rad is the angle the major axis makes with the vertical (y-axis).
    Computed via weighted PCA on pixel coordinates.
    """
    H, W = mask_2d.shape
    total = mask_2d.sum().item()
    if total < 1.0:
        return H / 2.0, W / 2.0, 0.0

    ys = torch.arange(H, device=mask_2d.device, dtype=torch.float32)
    xs = torch.arange(W, device=mask_2d.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    cy = (mask_2d * yy).sum().item() / total
    cx = (mask_2d * xx).sum().item() / total

    dy = yy - cy
    dx = xx - cx

    muu = (mask_2d * dy * dy).sum().item() / total  # var along y
    mvv = (mask_2d * dx * dx).sum().item() / total  # var along x
    muv = (mask_2d * dy * dx).sum().item() / total  # covariance

    # Larger eigenvalue of [[muu, muv],[muv, mvv]] in (y,x) coordinates.
    half_diff = (muu - mvv) / 2.0
    disc = math.sqrt(max(0.0, half_diff ** 2 + muv ** 2))
    lambda_max = (muu + mvv) / 2.0 + disc

    # Eigenvector direction (ey, ex): proportional to (muv, lambda_max - muu).
    ey = muv
    ex = lambda_max - muu
    angle = math.atan2(ex, ey) if (abs(ey) > 1e-6 or abs(ex) > 1e-6) else 0.0
    return cy, cx, angle


def _trim_coord_weight(
    coord_map: torch.Tensor,
    mask_ref: torch.Tensor,
    crop_pct: int,
    feather_px: int,
) -> torch.Tensor:
    """Build an [H, W] weight map that zeros crop_pct% of the mask's extent
    along coord_map, measured only within the masked region.

    crop_pct > 0: trim from the high-coord end.
    crop_pct < 0: trim from the low-coord end.
    """
    binary = mask_ref > 0.05
    if not binary.any():
        return torch.ones_like(coord_map)

    masked_vals = coord_map[binary]
    min_c = masked_vals.min().item()
    max_c = masked_vals.max().item()
    extent = max_c - min_c
    if extent < 1.0:
        return torch.ones_like(coord_map)

    cut_amount = abs(crop_pct) / 100.0 * extent
    fp = float(min(feather_px, cut_amount))
    w = torch.ones_like(coord_map)

    if crop_pct > 0:  # trim from high end
        cut_line = max_c - cut_amount
        w[coord_map > cut_line] = 0.0
        if fp > 1e-3:
            fade_mask = (coord_map >= cut_line - fp) & (coord_map <= cut_line)
            t = ((coord_map[fade_mask] - (cut_line - fp)) / fp).clamp(0.0, 1.0)
            w[fade_mask] = 1.0 - t
    else:  # trim from low end
        cut_line = min_c + cut_amount
        w[coord_map < cut_line] = 0.0
        if fp > 1e-3:
            fade_mask = (coord_map >= cut_line) & (coord_map <= cut_line + fp)
            t = ((coord_map[fade_mask] - cut_line) / fp).clamp(0.0, 1.0)
            w[fade_mask] = t

    return w


def _apply_trim_aligned(
    mask: torch.Tensor, crop_x: int, crop_y: int, feather: int
) -> torch.Tensor:
    """Trim the mask along its own principal axes (PCA orientation).

    Builds rotated coordinate maps aligned with the major/minor eigenvectors
    of the masked region, then evaluates the trim entirely in pixel space —
    no actual image rotation needed.
    """
    B, H, W = mask.shape
    device = mask.device

    ref = mask.float().mean(0)  # mean across batch for stable axis detection
    cy, cx, angle = _mask_centroid_and_angle(ref)

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    dy = yy - cy
    dx = xx - cx

    # Rotate coordinate frame by -angle so major axis aligns with vertical.
    # In (y, x) 2D: y' = y·cos θ − x·sin θ,  x' = y·sin θ + x·cos θ
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    dy_r = dy * cos_a - dx * sin_a  # along major axis
    dx_r = dy * sin_a + dx * cos_a  # along minor axis

    weight = torch.ones(H, W, device=device, dtype=torch.float32)

    if crop_y != 0:
        weight = weight * _trim_coord_weight(dy_r, ref, crop_y, feather)
    if crop_x != 0:
        weight = weight * _trim_coord_weight(dx_r, ref, crop_x, feather)

    return (mask * weight.unsqueeze(0)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AP_MaskTrim:
    DESCRIPTION = (
        "Trims a mask region by percentage along the X and/or Y axis.\n\n"
        "crop_x:\n"
        "  Positive value → trim that percentage from the RIGHT edge.\n"
        "  Negative value → trim from the LEFT edge.\n"
        "  Example: 30 removes the rightmost 30 % of the mask.\n\n"
        "crop_y:\n"
        "  Positive value → trim from the BOTTOM.\n"
        "  Negative value → trim from the TOP.\n"
        "  Example: -40 removes the top 40 % of the mask.\n\n"
        "auto_align:\n"
        "  When True, trim axes rotate to match the mask's principal orientation\n"
        "  (PCA major/minor axes of the masked region).  crop_y then always trims\n"
        "  the 'bottom' of the masked shape, regardless of how it is rotated in\n"
        "  the image — useful for tilted heads, diagonal limbs, etc.\n\n"
        "feather:\n"
        "  Soft-edge width in pixels at the cut boundary. 0 = hard cut.\n\n"
        "mask_dilate:\n"
        "  Expands the trimmed mask by this many pixels (morphological dilation).\n\n"
        "mask_blur:\n"
        "  Gaussian blur applied to the final mask after trimming and dilation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "crop_x": (
                    "INT",
                    {
                        "default": 0,
                        "min": -100,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "Trim % from right (positive) or left (negative). 0 = no trim.",
                    },
                ),
                "crop_y": (
                    "INT",
                    {
                        "default": 0,
                        "min": -100,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "Trim % from bottom (positive) or top (negative). 0 = no trim.",
                    },
                ),
                "auto_align": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When True, trim axes follow the mask's principal orientation (PCA).\n"
                            "crop_y trims the 'bottom' of the masked shape even when it is rotated."
                        ),
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Soft-edge width in pixels at the cut boundary.",
                    },
                ),
                "mask_dilate": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Dilate (expand) the trimmed mask by this many pixels.",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Gaussian blur applied to the final mask.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "trim_mask"
    CATEGORY = "AP ClipSEG Light"

    def trim_mask(
        self,
        mask: torch.Tensor,
        crop_x: int = 0,
        crop_y: int = 0,
        auto_align: bool = False,
        feather: int = 10,
        mask_dilate: int = 0,
        mask_blur: int = 0,
    ) -> tuple:
        # Ensure [B, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if bool(auto_align):
            result = _apply_trim_aligned(mask, int(crop_x), int(crop_y), int(feather))
        else:
            result = _apply_trim(mask, int(crop_x), int(crop_y), int(feather))

        # Dilate
        if int(mask_dilate) > 0:
            d = int(mask_dilate)
            k = 2 * d + 1
            m = result.unsqueeze(1)          # [B, 1, H, W]
            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=d)
            result = m.squeeze(1)

        # Blur
        if int(mask_blur) > 0:
            r = int(mask_blur)
            k = 2 * r + 1
            blur_k = torch.ones(1, 1, k, k, device=result.device, dtype=torch.float32) / (k * k)
            m = result.unsqueeze(1)          # [B, 1, H, W]
            for _ in range(3):
                m = F.conv2d(F.pad(m, [r, r, r, r], mode="reflect"), blur_k)
            result = m.squeeze(1).clamp(0.0, 1.0)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "AP_MaskTrim": AP_MaskTrim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AP_MaskTrim": "AP Mask Trim",
}
