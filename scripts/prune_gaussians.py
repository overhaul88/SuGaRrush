"""Visual-hull prune of a trained 3DGS checkpoint against U2-Net object masks.

Why this exists
---------------
Vanilla 3DGS reconstructs the whole scene -- object plus floor and background.
Handing all of that to SuGaR spends the mesh budget on clutter we do not want.
The object masks already say, per view, which pixels belong to the object, so we
carry carve_mesh.py's visual-hull idea over from meshes to Gaussians: forward-
project every Gaussian centre into every camera and keep a centre only if it
lands inside the object silhouette in a sufficient fraction of the views that
see it. Poses and the 3DGS attributes are left untouched -- background Gaussians
are simply dropped, leaving a point_cloud.ply SuGaR can consume directly.

    python scripts/prune_gaussians.py --gs-dir output/vanilla_gs/object4 \
        --scene scenes/object4 --out-ply pruned.ply

Masks are produced separately (see scripts/gen_masks.py) because the
segmentation model lives in its own conda env.

Two tests, because there are two ways to not belong
---------------------------------------------------
The agreement ratio alone is not a volumetric test. It is `n_in / n_frust`, and
`n_frust` counts only the views in which the centre lands inside the image
rectangle -- out-of-frame views never enter the denominator. That is deliberate
(a view that never looked cannot be evidence of background; hull_complete.py's
carve_hull rests on the same principle) but on its own it lets a distant centre
be judged by whichever handful of cameras happened to frame it. Measured on
object6: 477 centres a median 7.1 object-diagonals away survived at ratio 0.604,
clearing the 0.60 threshold by 0.004, and `--min-views 8` is no defence against
N = 236. They reached the coarse mesh, which came out spanning [22.7, 4.8, 16.3]
for a 1.66-unit object in 30 components; 411 of them became the "background mesh"
SuGaR merges in, and the 66 that fell inside the camera bbox polluted the Poisson
depth search into rejecting depths 9/8/7 (159/71/21 components) and settling for
depth 6 at 17k triangles.

So test the two failure modes separately, because they are disjoint:

  SUPPORT   n_frust >= sigma * N -- was this location interrogated at all? Kills
            geometry no camera meaningfully looked at (the distant junk).
  AGREEMENT n_in / n_frust >= keep_ratio -- of the cameras that did look, did they
            call it object? Kills geometry the cameras looked at and rejected
            (the table under the object).

The support gate makes the agreement ratio well-posed by bounding its denominator
below, rather than replacing it. Neither test subsumes the other: the table has
full support and fails agreement; the distant junk has high agreement and fails
support.

Sigma is measured, not chosen -- see calibrate_support. Hand-setting a constant
here is exactly the mistake that cost two runs in the hull work (a borrowed 0.25
inflated the hull 21.7% in Z), and the object's own worst-supported point is the
quantity that decides what is safe, which varies with how tightly the operator
framed the capture.

Centres, not ellipsoids: this tests Gaussian means. Measured on object6 the kept
Gaussians have max scale a median 0.023 against an object diagonal of 1.94 (1.2%),
so the mean is a faithful stand-in for the splat; SUGAR_SCALE_CLAMP handles the
few outsized ones downstream.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carve_mesh import load_cameras  # noqa: E402

SUGAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SUGAR_DIR, "gaussian_splatting"))
from scene.gaussian_model import GaussianModel  # noqa: E402


def calibrate_support(n_frust, n_views, lo=0.15, hi=0.85, bins=70,
                      min_gap=0.10, max_cut=0.25, default=0.50):
    """Locate the support threshold in the EMPTY BAND between the object and the junk.

    Frame support is bimodal for an orbit capture, and for a structural reason rather than a
    lucky one: the operator keeps the subject framed, so the object stays in frame almost
    always, while anything far enough away to be background drifts out of frame as the camera
    swings around. Measured on object6 (236 views):

        object   support 0.661 .. 1.000   (every object centre is in frame in >= 156 views)
        junk     support 0.000 .. 0.233   (never more than 55)
        bins from 0.25 to 0.65 are EMPTY -- 43% of the range, not one centre of either kind

    Any cut in that band gives an identical answer, which is what makes this robust; the
    threshold is not balancing an error trade-off, it is naming a gap. So find the gap: take
    the widest run of empty histogram bins inside [lo, hi] and cut at its midpoint.

    THE GAP ONLY EXISTS AMONG THE CENTRES THAT ALREADY PASS THE AGREEMENT TEST, so this must be
    called on that subpopulation and the two tests are ordered rather than parallel. On the raw
    541k-Gaussian cloud every support bin is populated -- 181,906 centres below 0.05, 53,040 at
    0.20-0.25, a steady tail all the way up -- because the background is everywhere and at every
    distance, so no band is empty and this returns the fallback. Condition on agreement first and
    the background is gone; what remains is the object plus whatever survived agreement by having
    too small a denominator to be tested, which is precisely the population the support question
    is about.

    Degenerate cases both fail safe. With no junk at all the whole range is empty, the midpoint
    lands below every object centre and nothing is removed. With no clean gap -- a capture framed
    so loosely that the object itself trails down into low support -- there is no bimodality to
    exploit, so fall back to `default` and, more importantly, refuse to cut more than `max_cut`
    of the cloud: an over-aggressive support gate would remove measured surface, which is the
    expensive direction of error (stage 7b can fill a wound, but it cannot know it should).

    Returns (sigma, n_removed_by_support, diagnostic_string).
    """
    s = np.asarray(n_frust, np.float64) / max(n_views, 1)
    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(s, bins=edges)

    best_len, best_run = 0, None
    i = 0
    while i < bins:
        if hist[i] == 0:
            j = i
            while j < bins and hist[j] == 0:
                j += 1
            if j - i > best_len:
                best_len, best_run = j - i, (i, j)
            i = j
        else:
            i += 1

    width = (hi - lo) * best_len / bins if best_run else 0.0
    if best_run is not None and width >= min_gap:
        sigma = float((edges[best_run[0]] + edges[best_run[1]]) / 2)
        how = f"empty band {edges[best_run[0]]:.3f}-{edges[best_run[1]]:.3f} (width {width:.3f})"
    else:
        sigma = float(default)
        how = (f"no empty band wider than {min_gap:g} in [{lo:g},{hi:g}]; "
               f"distribution is not bimodal -> falling back to default")

    cut = int((s < sigma).sum())
    if cut > max_cut * len(s):
        # Never let a mis-read distribution delete a quarter of the cloud.
        sigma = float(np.percentile(s, 100.0 * max_cut))
        how += f"; capped at the {100*max_cut:g}th percentile to bound the cut"
        cut = int((s < sigma).sum())
    return sigma, cut, how


def support_histogram(n_frust, n_views, sigma, bins=20):
    """Print the support distribution, so the gap the threshold sits in is visible and not
    merely asserted. The hull work's habit: render the artifact, do not trust the summary."""
    s = np.asarray(n_frust, np.float64) / max(n_views, 1)
    hist, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    scale = 46.0 / max(hist.max(), 1)
    lines = ["  frame support histogram (| marks the chosen cut):"]
    for k in range(bins):
        mark = " <== cut" if edges[k] <= sigma < edges[k + 1] else ""
        bar = "#" * int(round(hist[k] * scale))
        lines.append(f"    {edges[k]:.2f}-{edges[k+1]:.2f} {hist[k]:>7,} {bar}{mark}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs-dir", required=True, help="vanilla 3DGS checkpoint dir")
    ap.add_argument("--scene", required=True, help="COLMAP scene dir (has sparse/0 and masks/)")
    ap.add_argument("--out-ply", required=True, help="where to write the pruned point_cloud.ply")
    ap.add_argument("--iteration", type=int, default=7000)
    ap.add_argument("--min-views", type=int, default=8,
                    help="centres seen by fewer cameras than this are dropped")
    ap.add_argument("--keep-ratio", type=float, default=0.6,
                    help="fraction of observing views that must see the centre inside the mask")
    ap.add_argument("--min-support", type=float, default=0.0,
                    help="fraction of ALL masked views in which a centre must land in frame. "
                         "0 = AUTO: cut in the empty band between the object and the distant "
                         "junk (see calibrate_support). A ratio test whose denominator is only "
                         "the views that happened to frame the point is not a volumetric test; "
                         "this is what bounds that denominator")
    ap.add_argument("--dilate", type=int, default=2,
                    help="dilate masks by N px to tolerate slight pose error")
    ap.add_argument("--report-json", default=None,
                    help="write the prune's measured statistics here")
    args = ap.parse_args()

    ply = os.path.join(args.gs_dir, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    masks_dir = os.path.join(args.scene, "masks")

    t0 = time.time()
    model = GaussianModel(3)  # 3DGS default SH degree -> f_rest_0..44
    model.load_ply(ply)
    X = model.get_xyz.detach().cpu().numpy()
    print(f"loaded {len(X):,} gaussians from {ply} ({time.time()-t0:.1f}s)")

    cams = load_cameras(args.scene)
    print(f"cameras: {len(cams)}")

    t1 = time.time()
    seen = np.zeros(len(X), np.int32)
    inside = np.zeros(len(X), np.int32)
    used = 0
    for ci, c in enumerate(cams):
        mp = os.path.join(masks_dir, os.path.splitext(c["name"])[0] + ".png")
        if not os.path.isfile(mp):
            continue
        m = np.array(Image.open(mp).convert("L")) > 127
        if args.dilate > 0:
            # Cheap binary dilation: max-pool over a (2d+1) window via shifts.
            d = args.dilate
            acc = m.copy()
            for dy in range(-d, d + 1):
                for dx in range(-d, d + 1):
                    acc |= np.roll(np.roll(m, dy, 0), dx, 1)
            m = acc
        mh, mw = m.shape
        sx, sy = mw / c["w"], mh / c["h"]

        # Project every centre into this camera at once (vectorised, no per-Gaussian loop).
        P = X @ c["R"].T + c["t"]
        z = P[:, 2]
        front = z > 1e-6
        u = np.full(len(X), -1.0); v = np.full(len(X), -1.0)
        u[front] = c["fx"] * P[front, 0] / z[front] + c["cx"]
        v[front] = c["fy"] * P[front, 1] / z[front] + c["cy"]
        px = np.round(u * sx).astype(np.int64)
        py = np.round(v * sy).astype(np.int64)
        vis = front & (px >= 0) & (px < mw) & (py >= 0) & (py < mh)
        seen += vis
        idx = np.nonzero(vis)[0]
        inside[idx] += m[py[idx], px[idx]]
        used += 1
        if ci % 50 == 0:
            print(f"  view {ci}/{len(cams)}", flush=True)
    print(f"used {used} masked views ({time.time()-t1:.1f}s)")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(seen > 0, inside / np.maximum(seen, 1), 0.0)

    # Order matters: agreement first, then support calibrated on what agreement left behind.
    # See calibrate_support -- the raw cloud has no empty band because the background occupies
    # every support level, and it is the agreement test's job to remove it.
    agreeing = ratio >= args.keep_ratio
    if args.min_support > 0:
        sigma, how = float(args.min_support), "set explicitly"
    else:
        sigma, _, how = calibrate_support(seen[agreeing], used)
    print(f"support threshold sigma = {sigma:.3f} of {used} views "
          f"({int(np.ceil(sigma*used))} views); {how}")
    print(support_histogram(seen[agreeing], used, sigma))

    supported = seen >= sigma * used
    keep = supported & agreeing & (seen >= args.min_views)
    n_keep, n_tot = int(keep.sum()), len(X)
    # Report the two tests separately: they target disjoint failure modes, so a run in which one
    # of them does nothing is a signal about the capture, not a redundancy to be optimised away.
    print(f"  fails support only  : {int((~supported & agreeing).sum()):,}  "
          f"(interrogated by too few cameras -- the distant junk)")
    print(f"  fails agreement only: {int((supported & ~agreeing).sum()):,}  "
          f"(cameras looked and called it background -- floor, table, backdrop)")
    print(f"  fails both          : {int((~supported & ~agreeing).sum()):,}")
    print(f"kept {n_keep:,} / {n_tot:,} gaussians ({100*n_keep/max(n_tot,1):.1f}%) "
          f"[support>={sigma:.3f}, ratio>={args.keep_ratio}, views>={args.min_views}]")
    if n_keep == 0:
        raise SystemExit("prune removed everything; lower --keep-ratio or --min-support")

    ext_before = X.max(0) - X.min(0)
    ext_after = X[keep].max(0) - X[keep].min(0)
    print(f"extent {np.round(ext_before, 3)} -> {np.round(ext_after, 3)}")

    if args.report_json:
        import json
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(dict(
                n_gaussians=n_tot, n_kept=n_keep, n_views=used, sigma=sigma, how=how,
                keep_ratio=args.keep_ratio, min_views=args.min_views,
                fails_support_only=int((~supported & agreeing).sum()),
                fails_agreement_only=int((supported & ~agreeing).sum()),
                fails_both=int((~supported & ~agreeing).sum()),
                support_p0=float(np.min(seen[keep]) / used),
                support_p50=float(np.median(seen[keep]) / used),
                extent_before=[round(float(v), 4) for v in ext_before],
                extent_after=[round(float(v), 4) for v in ext_after],
            ), f, indent=2)
        print(f"wrote report {args.report_json}")

    t2 = time.time()
    km = torch.from_numpy(keep).to(model._xyz.device)
    for attr in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        t = getattr(model, attr)
        setattr(model, attr, nn.Parameter(t[km].detach().requires_grad_(True)))
    model.save_ply(args.out_ply)
    print(f"wrote {args.out_ply} ({time.time()-t2:.1f}s)")


if __name__ == "__main__":
    main()
