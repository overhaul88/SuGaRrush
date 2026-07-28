"""Object-mask photometric loss for SuGaR (loss-masked Gaussian isolation).

The `--gaussian-prune` path prunes the trained Gaussians to the object before SuGaR. But
SuGaR *re-optimises* its input Gaussians against the GT images, and with the background
removed an unmasked loss pulls those object Gaussians outward to explain the (now
unrepresentable) background. Restricting the photometric loss to the U2Net silhouette
(``SUGAR_MASK_LOSS=1``) zeroes background gradients so the Gaussians stay on the object.

Two refinements make masked training on a *small* object viable:

* **Area-normalized L1** (``L1 = sum(M*|I-Î|) / (sum(M)*C)``): a plain ``.mean()`` divides by
  the whole tensor, so with the object at ~5 % of the frame the object gradients are divided
  by ~20 and starve. Normalizing by the mask area keeps them at full magnitude regardless of
  how much of the frame the object fills.
* **Eroded-mask SSIM**: SSIM uses an 11x11 window (``loss_utils.ssim``). At the mask edge that
  window straddles the artificial black boundary and registers a huge structural edge, pulling
  Gaussians to the silhouette and tearing topology. Eroding the mask inward ~5 px before the
  SSIM term keeps the window entirely inside the object. The L1 term still uses the full mask.

``SUGAR_MASK_LOSS_WEIGHT`` (default 1.0) is a global multiplier; ``SUGAR_SSIM_ERODE`` (default 5)
is the erosion radius.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sugar_utils.loss_utils import ssim


def mask_loss_enabled():
    return os.environ.get("SUGAR_MASK_LOSS", "0") == "1"


def mask_loss_weight():
    return float(os.environ.get("SUGAR_MASK_LOSS_WEIGHT", "1.0"))


def mask_ssim_erode():
    return int(os.environ.get("SUGAR_SSIM_ERODE", "5"))


def build_mask_bank(cam_list, masks_dir, image_height, image_width):
    """CPU uint8 tensor (N, H, W) of binary object masks aligned to ``cam_list`` order.

    If a camera already carries an ``object_mask`` (set by the focal-crop loader, already at the
    render resolution and aligned to the cropped GT), that is used directly. Otherwise the mask is
    loaded from ``masks_dir`` by ``image_name`` and resized (nearest) to the render resolution. A
    missing mask falls back to all-ones (that view is left unmasked) with a warning.
    """
    n = len(cam_list)
    bank = torch.ones((n, image_height, image_width), dtype=torch.uint8)
    missing = 0
    for i, cam in enumerate(cam_list):
        cam_mask = getattr(cam, "object_mask", None)
        if cam_mask is not None:
            bank[i] = torch.as_tensor(np.asarray(cam_mask) > 0, dtype=torch.uint8)
            continue
        stem = os.path.splitext(os.path.basename(cam.image_name))[0]
        mp = os.path.join(masks_dir, stem + ".png")
        if not os.path.isfile(mp):
            missing += 1
            continue
        m = Image.open(mp).convert("L").resize((image_width, image_height), Image.NEAREST)
        bank[i] = torch.from_numpy((np.asarray(m) > 127).astype(np.uint8))
    frac = bank.float().mean().item()
    note = "" if not missing else f", {missing} missing -> left unmasked"
    print(f"[mask-loss] loaded {n - missing}/{n} object masks "
          f"(mean coverage {100 * frac:.1f}%{note})", flush=True)
    return bank


def erode_mask(mask, radius):
    """Erode a binary mask inward by ``radius`` px. mask: (1,1,H,W) float in {0,1}.

    Implemented as a max-pool on the inverted mask (GPU-native), so a pixel survives only if
    every pixel within ``radius`` is inside the object.
    """
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=radius)


def masked_photometric_loss(pred_rgb, gt_rgb, mask, dssim_factor, weight=1.0, ssim_erode=5):
    """Area-normalized masked L1 + eroded-mask D-SSIM.

    pred_rgb / gt_rgb: (B, 3, H, W). mask: (1, 1, H, W) in {0, 1}, broadcast over channels.
    """
    m = mask
    c = pred_rgb.shape[-3]
    l1 = (m * (pred_rgb - gt_rgb).abs()).sum() / (m.sum() * c + 1e-8)
    if dssim_factor > 0:
        ms = erode_mask(m, ssim_erode)
        dssim = 1.0 - ssim(pred_rgb * ms, gt_rgb * ms)
        return weight * ((1.0 - dssim_factor) * l1 + dssim_factor * dssim)
    return weight * l1
