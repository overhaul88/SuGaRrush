"""Keep only the sharpest frame per sliding window.

Motion blur is the biggest quality killer for a hand-held capture, so we
over-extract frames with ffmpeg and then keep the sharpest one in each
consecutive window. Sharpness is the variance of the Laplacian, the standard
no-reference focus measure: blur suppresses high-frequency detail, which
lowers that variance.

    python scripts/select_sharp.py --src $SCENE/_allframes --dst $SCENE/input --target 200
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


def main():
    ap = argparse.ArgumentParser(description="Keep the sharpest frame per window.")
    ap.add_argument("--src", required=True, help="directory of over-extracted frames")
    ap.add_argument("--dst", required=True, help="output directory (COLMAP 'input')")
    ap.add_argument("--target", type=int, default=200, help="approximate number of frames to keep")
    ap.add_argument("--work-width", type=int, default=640, help="width used for the sharpness measure")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel scoring workers")
    ap.add_argument("--dry-run", action="store_true", help="report only, copy nothing")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(EXTS))
    if not files:
        raise SystemExit(f"No images found in {args.src}")
    if args.target < 1:
        raise SystemExit("--target must be >= 1")

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
    width = max(5, len(str(len(keep))))
    for n, idx in enumerate(keep, start=1):
        ext = os.path.splitext(files[idx])[1].lower()
        shutil.copy2(paths[idx], os.path.join(args.dst, f"{n:0{width}d}{ext}"))
    print(f"Wrote {len(keep)} frames to {args.dst}")


if __name__ == "__main__":
    main()
