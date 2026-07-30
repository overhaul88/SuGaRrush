"""Keep only the sharpest frame per consecutive temporal window.

Motion blur is the biggest quality killer for a hand-held capture, so we
over-extract frames with ffmpeg and then keep the sharpest one in each
consecutive window. Sharpness is the variance of the Laplacian, the standard
no-reference focus measure: blur suppresses high-frequency detail, which
lowers that variance.

An optional bilateral-filter pass is deliberately applied *after* selection.
It therefore cannot change which viewpoints survive, and it only pays the
filtering cost for frames that are copied into the COLMAP scene.

    python scripts/select_sharp.py --src $SCENE/_allframes --dst $SCENE/input --target 200
    python scripts/select_sharp.py --src $SCENE/_allframes --dst $SCENE/input --target 200 \
        --bilateral
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sharpness(path, work_width=640):
    """Variance of the Laplacian, computed on a downscaled grayscale image."""
    with Image.open(path) as im:
        im = im.convert("L")
        if im.width > work_width:
            h = max(1, round(im.height * work_width / im.width))
            im = im.resize((work_width, h), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return float(laplace(a).var())


def write_selected_frame(src, dst, bilateral=False, diameter=3,
                         sigma_color=10.0, sigma_space=1.0, jpeg_quality=95):
    """Copy a selected frame, optionally applying an edge-preserving bilateral filter."""
    if not bilateral:
        shutil.copy2(src, dst)
        return

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "--bilateral requires OpenCV (cv2). The end-to-end pipeline runs "
            "this mode in the 'seg' environment, where OpenCV is available."
        ) from exc

    image = cv2.imread(src, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read selected frame: {src}")
    filtered = cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        borderType=cv2.BORDER_REFLECT_101,
    )

    ext = os.path.splitext(dst)[1].lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(dst, filtered, params):
        raise RuntimeError(f"OpenCV could not write filtered frame: {dst}")


def main():
    ap = argparse.ArgumentParser(description="Keep the sharpest frame per window.")
    ap.add_argument("--src", required=True, help="directory of over-extracted frames")
    ap.add_argument("--dst", required=True, help="output directory (COLMAP 'input')")
    ap.add_argument("--target", type=int, default=200, help="approximate number of frames to keep")
    ap.add_argument("--work-width", type=int, default=640, help="width used for the sharpness measure")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel scoring workers")
    ap.add_argument("--bilateral", action="store_true",
                    help="bilateral-filter selected frames before writing them")
    ap.add_argument("--bilateral-diameter", type=int, default=3,
                    help="bilateral neighbourhood diameter; must be a positive odd integer")
    ap.add_argument("--bilateral-sigma-color", type=float, default=10.0,
                    help="bilateral colour sigma in 8-bit intensity units")
    ap.add_argument("--bilateral-sigma-space", type=float, default=1.0,
                    help="bilateral spatial sigma in pixels")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG quality used when filtered frames are re-encoded")
    ap.add_argument("--dry-run", action="store_true", help="report only, copy nothing")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(EXTS))
    if not files:
        raise SystemExit(f"No images found in {args.src}")
    if args.target < 1:
        raise SystemExit("--target must be >= 1")
    if args.bilateral_diameter < 1 or args.bilateral_diameter % 2 == 0:
        raise SystemExit("--bilateral-diameter must be a positive odd integer")
    if args.bilateral_sigma_color <= 0 or args.bilateral_sigma_space <= 0:
        raise SystemExit("--bilateral sigmas must be > 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    paths = [os.path.join(args.src, f) for f in files]
    print(f"Scoring {len(files)} frames from {args.src} ...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scores = list(pool.map(sharpness, paths, [args.work_width] * len(paths), chunksize=8))

    if len(files) <= args.target:
        print(f"Only {len(files)} frames available (<= target {args.target}); keeping all.")
        keep = list(range(len(files)))
    else:
        # Consecutive, non-overlapping windows keep the selection evenly spread
        # over the capture, so we never drop a whole viewpoint just because that
        # part of the orbit was slightly blurrier than the rest.
        window = len(files) / args.target
        keep = []
        for i in range(args.target):
            lo = int(round(i * window))
            hi = int(round((i + 1) * window))
            hi = min(max(hi, lo + 1), len(files))
            if lo >= len(files):
                break
            best = lo + int(np.argmax(scores[lo:hi]))
            keep.append(best)
        keep = sorted(set(keep))

    kept_scores = [scores[i] for i in keep]
    print(f"Keeping {len(keep)} frames.")
    print(f"  sharpness kept   : min {min(kept_scores):.6f}  median {np.median(kept_scores):.6f}  max {max(kept_scores):.6f}")
    print(f"  sharpness overall: min {min(scores):.6f}  median {np.median(scores):.6f}  max {max(scores):.6f}")

    if args.dry_run:
        print("Dry run; nothing written.")
        return

    os.makedirs(args.dst, exist_ok=True)
    if args.bilateral:
        print(
            "Applying bilateral filter to selected frames only: "
            f"d={args.bilateral_diameter}, sigmaColor={args.bilateral_sigma_color:g}, "
            f"sigmaSpace={args.bilateral_sigma_space:g}, JPEG quality={args.jpeg_quality}"
        )
    width = max(5, len(str(len(keep))))
    for n, idx in enumerate(keep, start=1):
        ext = os.path.splitext(files[idx])[1].lower()
        write_selected_frame(
            paths[idx],
            os.path.join(args.dst, f"{n:0{width}d}{ext}"),
            bilateral=args.bilateral,
            diameter=args.bilateral_diameter,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
            jpeg_quality=args.jpeg_quality,
        )
    print(f"Wrote {len(keep)} frames to {args.dst}")


if __name__ == "__main__":
    main()
