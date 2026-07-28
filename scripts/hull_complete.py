"""Complete the wound with the VISUAL HULL instead of a smoothness prior.

Why this exists
---------------
After the observation-confidence filter (scripts/observation_confidence.py) the mesh is honest but
open: everything the cameras never certified has been removed. Something has to fill that wound,
and the choice of filler is the whole ballgame.

Every filler tried before this one invents its answer out of a prior rather than out of the data:

  * screened Poisson over the hole minimises a smooth energy, so its minimiser is a DOME (Sec Q4);
  * pymeshfix triangulates a big non-planar rim into a folded, pinched tent;
  * a planar cap fits one plane to the rim -- correct only when the rim IS planar. On object4 the
    rim snakes over 52% of the object diagonal (planarity deviation mean 0.138, and only 1.9% of it
    lies near the outer extreme), so the least-squares plane passes straight through the solid and
    the "cap" becomes a plate that slices a corner off the cube. That plate was 327 triangles
    carrying 23.7% of the surface area.

The visual hull is different in kind: it is a deterministic function of the observations. Under
Sec 4 that makes it admissible -- completing with it adds no information the cameras did not
supply, so I(C;D|O) = 0. For a box seen from an orbit it yields planar side walls and a flat
bottom for free, which is exactly the piecewise-planar answer Sec Q4 argues for, obtained by
measurement instead of by iterating a normal flow toward it.

Measured on object4: hull extent [0.808 0.917 0.973] against the measured mesh's
[0.824 0.920 0.972] -- the hull is tight, not a loose bound.

Method
------
1. Carve an occupancy grid against the (dilated) masks BY VOTE, not by strict intersection: a voxel
   dies only when more than (1 - keep_ratio) of the views call it background. A strict intersection
   lets one bad mask delete real geometry -- on object6 it removed an upper body whose median mask
   agreement was 0.924 over 118 views. A voxel outside a view's image is not counted as dissent:
   absence of evidence is not evidence.
2. Take the hull's boundary voxels as oriented surface samples, placed to SUB-VOXEL accuracy by a
   Newton step onto the 0.5 isolevel of the smoothed occupancy. Snapping them to voxel centres
   instead leaves a visible staircase that Poisson faithfully reproduces.
3. Keep only hull samples farther than tau from the measured surface -- i.e. only inside the wound.
   Where the measured surface exists it remains the evidence; the hull is merely an outer bound.
4. Solve one screened-Poisson problem over measured + wound samples, so the result is closed by
   construction (Sec 8) rather than closed by a repair operator afterwards.
5. Snap the observed part of the result back onto the measured surface (Sec 7.4: observed geometry
   does not move). Poisson would otherwise smooth it by ~1% of the diagonal.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observation_confidence import load_cameras, auto_stride


def calibrate_dissent(cams, masks_dir, mesh, dilate, pct=95.0, margin=1.5,
                      floor=0.04, ceil_=0.30, n_pts=6000, quiet=False):
    """Derive the carving tolerance from KNOWN-REAL geometry instead of guessing a constant.

    The measured surface is, by construction, geometry the cameras already agreed on. Sampling it
    and histogramming each sample's dissent fraction therefore measures this capture's segmentation
    noise floor directly, and the carve threshold can be set just above it.

    Measured on object6 (118 views): samples ON the measured surface dissent in a median 0.009 and
    a p90 0.059 of the views, while points 0.02*diag outside it dissent in a median 0.204. The two
    populations separate cleanly -- but only if the threshold sits between them. A strict
    intersection (0) deleted real geometry with 0.924 agreement; a hand-set 0.25 inflated the hull
    by 21.7% in Z and made 91.6% of hull samples read as wound. Neither number was measured; this
    one is.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=n_pts, use_triangle_normal=True)
    P = np.asarray(pcd.points)
    n_see = np.zeros(len(P), np.int32)
    n_out = np.zeros(len(P), np.int32)
    for cam in cams:
        mp = os.path.join(masks_dir, os.path.splitext(cam["name"])[0] + ".png")
        if not os.path.isfile(mp):
            continue
        fg = np.array(Image.open(mp).convert("L")) > 127
        if dilate:
            fg = binary_dilation(fg, iterations=dilate)
        mh, mw = fg.shape
        sx, sy = mw / cam["w"], mh / cam["h"]
        Q = P @ cam["R"].T + cam["t"]
        z = Q[:, 2]
        ok = z > 1e-6
        zz = np.where(ok, z, 1.0)
        px = np.round((cam["fx"] * Q[:, 0] / zz + cam["cx"]) * sx).astype(np.int64)
        py = np.round((cam["fy"] * Q[:, 1] / zz + cam["cy"]) * sy).astype(np.int64)
        inb = ok & (px >= 0) & (px < mw) & (py >= 0) & (py < mh)
        j = np.nonzero(inb)[0]
        n_see[j] += 1
        n_out[j[~fg[py[j], px[j]]]] += 1
    m = n_see > 0
    if not m.any():
        return floor
    f = n_out[m] / n_see[m]
    noise = float(np.percentile(f, pct))
    tol = float(np.clip(margin * noise, floor, ceil_))
    if not quiet:
        print(f"    dissent on KNOWN-REAL surface: median {np.median(f):.3f} "
              f"p{pct:g} {noise:.3f}  ->  carve tolerance {tol:.3f} "
              f"({margin:g}x the noise floor, clamped to [{floor:g}, {ceil_:g}])")
    return tol


def carve_hull(scene, masks_dir, lo, hi, n, dilate, stride, keep_ratio=0.0, mesh=None,
               chunk=3_000_000, quiet=False):
    """Space-carve an occupancy grid against the silhouettes, by VOTE rather than by intersection.

    The textbook visual hull is a strict intersection: a voxel dies the moment one view calls it
    background. That makes the hull maximally fragile to segmentation error, because a single bad
    mask deletes real geometry no matter how many views disagree -- and U2Net masks are not perfect.
    Measured on object6: the figurine's upper body has median mask agreement 0.924 over 118 views,
    yet strict intersection removed it, because ~9 views dissented. The hull came out 30% short in Y
    (0.995 vs the measured 1.419), the fusion then had no support up there, and the "completed" mesh
    arrived with 3 components and 78 boundary edges.

    So carve by VOTE: keep a voxel unless the views that disagree exceed a tolerance. A voxel out of
    frame in a view is not counted as dissent (absence of evidence is not evidence), which keeps the
    result a conservative OUTER bound -- exactly what a completion prior must be.

    The tolerance is calibrated against the measured surface rather than hand-set (see
    calibrate_dissent). Hand-setting it failed in both directions on this object: 0 deleted a real
    upper body, 0.25 inflated the hull by 21.7% in Z.
    """
    step = (hi - lo) / n
    cams = load_cameras(scene)
    stride = auto_stride(len(cams), stride)
    cams = cams[::stride]
    n_masked = sum(1 for c in cams
                   if os.path.isfile(os.path.join(masks_dir,
                                                  os.path.splitext(c["name"])[0] + ".png")))
    tol = (1.0 - keep_ratio) if keep_ratio > 0 else calibrate_dissent(
        cams, masks_dir, mesh, dilate, quiet=quiet)
    max_out = max(1, int(np.ceil(tol * max(n_masked, 1))))
    if not quiet:
        print(f"    carving by vote: a voxel needs {max_out} dissenting views of {n_masked} "
              f"to be removed (tolerance {tol:.3f})")
    out_cnt = np.zeros(n ** 3, np.uint16)
    occ = np.ones(n ** 3, bool)
    # Decode voxel indices per chunk rather than materialising ix/iy/iz for the whole grid: at
    # grid 256 those three int64 arrays are ~400 MB held for the entire carve, and this host has
    # 7.7 GB total and has already lost the WSL VM once to a memory spike.
    nn = n * n

    used = 0
    for ci, cam in enumerate(cams):
        mp = os.path.join(masks_dir, os.path.splitext(cam["name"])[0] + ".png")
        if not os.path.isfile(mp):
            continue
        fg = np.array(Image.open(mp).convert("L")) > 127
        if dilate:
            fg = binary_dilation(fg, iterations=dilate)
        mh, mw = fg.shape
        sx, sy = mw / cam["w"], mh / cam["h"]
        live = np.nonzero(occ)[0]
        if not len(live):
            break
        for s in range(0, len(live), chunk):
            idx = live[s:s + chunk]
            P = np.stack([lo[0] + (idx // nn + 0.5) * step[0],
                          lo[1] + ((idx // n) % n + 0.5) * step[1],
                          lo[2] + (idx % n + 0.5) * step[2]], 1) @ cam["R"].T + cam["t"]
            z = P[:, 2]
            ok = z > 1e-6
            zz = np.where(ok, z, 1.0)
            px = np.round((cam["fx"] * P[:, 0] / zz + cam["cx"]) * sx).astype(np.int64)
            py = np.round((cam["fy"] * P[:, 1] / zz + cam["cy"]) * sy).astype(np.int64)
            inb = ok & (px >= 0) & (px < mw) & (py >= 0) & (py < mh)
            j = np.nonzero(inb)[0]
            dissent = np.zeros(len(idx), bool)
            dissent[j] = ~fg[py[j], px[j]]
            di = idx[dissent]
            out_cnt[di] += 1
            occ[di[out_cnt[di] >= max_out]] = False
        used += 1
        if not quiet and ci % 40 == 0:
            print(f"    view {ci}/{len(cams)}  occupied {occ.sum():,}", flush=True)
    return occ.reshape(n, n, n), used, step


def hull_samples(vol, lo, step, smooth=1.0):
    """Oriented surface samples on the hull boundary, placed to sub-voxel accuracy.

    The occupancy field is binary, so its 0.5 isolevel sits somewhere inside each boundary voxel.
    Smoothing it slightly and taking one Newton step along the gradient puts the sample on that
    isolevel and gives a normal that is not quantised to the 26 lattice directions -- without this
    the cap shows the voxel staircase.
    """
    f = gaussian_filter(vol.astype(np.float32), smooth)
    pad = np.pad(vol, 1)
    interior = (pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1] &
                pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    surf = vol & ~interior
    si = np.nonzero(surf.ravel())[0]
    n = vol.shape[0]
    P = np.stack([lo[0] + (si // (n * n) + 0.5) * step[0],
                  lo[1] + ((si // n) % n + 0.5) * step[1],
                  lo[2] + (si % n + 0.5) * step[2]], 1)
    gx, gy, gz = np.gradient(f, step[0], step[1], step[2])
    G = np.stack([gx.ravel()[si], gy.ravel()[si], gz.ravel()[si]], 1)
    g2 = np.einsum("ij,ij->i", G, G)
    ok = g2 > 1e-20
    P, G, g2, si = P[ok], G[ok], g2[ok], si[ok]
    # Newton step onto the 0.5 isolevel, clamped to one voxel so a flat gradient cannot fling a point
    resid = 0.5 - f.ravel()[si]
    t = np.clip(resid / g2, -step.max(), step.max())
    P = P + t[:, None] * G
    N = -G / np.sqrt(g2)[:, None]          # occupancy decreases outward
    return P, N


def orient_outward(mesh, centre):
    """Make the winding consistent and globally outward. Must run BEFORE any RaycastingScene is
    built from the mesh, because the inward probe reads primitive_normals and needs them outward."""
    mesh.orient_triangles()
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fnu = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    vote = float(np.sum(area * np.einsum("ij,ij->i", fnu, V[T].mean(1) - centre)))
    if vote < 0:
        mesh.triangles = o3d.utility.Vector3iVector(T[:, ::-1])
    return vote < 0


def outward_oriented_samples(mesh, n_points, centre):
    """Sample the measured surface with normals that all point out of the solid.

    Orienting each normal outward from the centroid is only valid for a star-shaped surface. Inside
    the deep grooves between this cube's stickers the wall normals are nearly tangential to
    (P - centre), so the sign is noise there and Poisson stitches handles across the groove. Make
    the winding consistent first (local and exact), then choose the single global sign by an
    area-weighted vote, which individual ambiguous grooves cannot swing.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=n_points, use_triangle_normal=True)
    return np.asarray(pcd.points), np.asarray(pcd.normals), False


def _topology(mesh):
    """(components, genus, boundary_edges). chi = 2C - 2g - b, so genus is only meaningful once
    the boundary loops are counted -- reporting (2C - chi)/2 on a mesh with holes returns nonsense
    (it read genus -17 on a mesh whose real defect was open boundary)."""
    lbl, cnt, _ = mesh.cluster_connected_triangles()
    c = len(np.asarray(cnt))
    ec = defaultdict(int)
    for t in np.asarray(mesh.triangles):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[(a, b) if a < b else (b, a)] += 1
    nb = sum(1 for v in ec.values() if v == 1)
    chi = mesh.euler_poincare_characteristic()
    loops = 1 if nb else 0                       # cheap proxy; exact count needs ring tracing
    return c, (2 * c - chi - loops) // 2, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="rho-filtered (open) mesh")
    ap.add_argument("--scene", required=True, help="COLMAP scene dir (sparse/0 + masks/)")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=256, help="hull voxel grid resolution per axis")
    ap.add_argument("--keep-ratio", type=float, default=0.0,
                    help="a voxel survives unless more than (1 - keep-ratio) of the views call it "
                         "background. 0 = AUTO: calibrate the tolerance against the measured "
                         "surface's own dissent distribution. 1.0 reproduces the fragile "
                         "strict-intersection hull")
    ap.add_argument("--dilate", type=int, default=2,
                    help="mask dilation in px; matches carve_mesh.py, absorbs segmentation bias")
    ap.add_argument("--view-stride", type=int, default=0,
                    help="use every Nth camera; 0 = auto, ~120 views for any frame count. A fixed "
                         "stride makes the hull depend on capture length: 499 frames at stride 4 "
                         "gave 125 views, a 240-frame capture would give 60 and a looser hull")
    ap.add_argument("--hull-smooth", type=float, default=1.0,
                    help="sigma (in voxels) for smoothing the occupancy field before extracting the "
                         "isosurface. 1.0 measured best on object4; raising it to 1.6 did not "
                         "visibly reduce the residual voxel-frequency ripple (that ripple sits at "
                         "the Poisson cell size, so a finer --grid is the lever) and cost exact "
                         "watertightness")
    ap.add_argument("--probe-frac", type=float, default=0.03,
                    help="inward probe length as a fraction of the diagonal: a hull sample with "
                         "measured surface within this distance inward is redundant, not wound")
    ap.add_argument("--tau-frac", type=float, default=0.012,
                    help="wound threshold as a fraction of the object diagonal")
    ap.add_argument("--depth", default="9", help="Poisson octree depth, or 'auto' to search")
    ap.add_argument("--n-samples", type=int, default=200_000,
                    help="samples drawn from the measured surface")
    ap.add_argument("--target-faces", type=int, default=0, help="0 = no decimation")
    ap.add_argument("--no-snap", action="store_true",
                    help="skip snapping observed vertices back onto the measured surface")
    ap.add_argument("--hull-ply", default=None)
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    masks_dir = args.masks or os.path.join(args.scene, "masks")
    meas = o3d.io.read_triangle_mesh(args.mesh)
    meas.remove_unreferenced_vertices()
    Vm = np.asarray(meas.vertices)
    lo0, hi0 = Vm.min(0), Vm.max(0)
    centre = (lo0 + hi0) / 2
    diag = float(np.linalg.norm(hi0 - lo0))
    tau = args.tau_frac * diag
    half = (hi0 - lo0) / 2 * 1.15
    lo, hi = centre - half, centre + half
    flipped = orient_outward(meas, centre)
    print(f"measured mesh: {len(meas.triangles):,} faces, diagonal {diag:.4f}, tau {tau:.4f}"
          + ("  (global winding flipped: area-weighted vote said inward)" if flipped else ""))

    t0 = time.time()
    vol, used, step = carve_hull(args.scene, masks_dir, lo, hi, args.grid,
                                 args.dilate, args.view_stride, args.keep_ratio, mesh=meas)
    lab, nl = cc_label(vol)
    if nl > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        vol = lab == sizes.argmax()
    hull_vol = float(vol.sum() * np.prod(step))
    print(f"  hull carved from {used} views ({time.time()-t0:.1f}s): {vol.sum():,} voxels, "
          f"volume {hull_vol:.4f}, voxel {step.mean():.5f}")

    Ph, Nh = hull_samples(vol, lo, step, smooth=args.hull_smooth)
    ext_h = Ph.max(0) - Ph.min(0)
    print(f"  hull surface samples {len(Ph):,}   extent {np.round(ext_h,4)} "
          f"vs measured {np.round(hi0-lo0,4)}")
    if args.hull_ply:
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ph))
        p.normals = o3d.utility.Vector3dVector(Nh)
        os.makedirs(os.path.dirname(args.hull_ply) or ".", exist_ok=True)
        o3d.io.write_point_cloud(args.hull_ply, p)

    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(meas))
    d_hull = rc.compute_distance(o3d.core.Tensor(Ph.astype(np.float32))).numpy()

    # "Far from the measured surface" is not the same as "the measured surface is missing here", and
    # conflating them is what let hull looseness leak into the result. A hull that sits a little
    # outside an EXISTING measured face is far from it by that offset, so a pure distance test calls
    # it wound and injects a competing outer shell -- on object6 that made 91.6% of hull samples read
    # as wound and inflated the object by 21.7% in Z.
    #
    # Ask the question directly instead: look inward along the sample's own normal. If measured
    # surface lies just inside, this sample is redundant and the measurement wins. If the probe
    # travels through nothing, the surface really is absent and the hull is all we have. This makes
    # the wound test insensitive to how tightly the hull was carved, so the carve can be safely
    # loose -- and a loose carve is the safe direction, since over-carving removes the support the
    # wound needs and reopens the hole.
    # The probe must also check WHAT it hit. On a thin part -- this figurine's base plate, its limbs
    # -- an inward ray from a genuine wound crosses the hollow and strikes the FAR wall, which a
    # distance-only probe reads as "surface is present", so it discards the sample and leaves the
    # hole unfilled (measured: 215 boundary edges). A face lying just inside this sample is hit on
    # its outward side, so its normal agrees with the sample's; the far wall is hit from behind and
    # its normal opposes. One dot product separates the two.
    probe = max(3.0 * tau, args.probe_frac * diag)
    org = (Ph - 1e-3 * diag * Nh).astype(np.float32)
    rays = np.hstack([org, (-Nh).astype(np.float32)])
    ans = rc.cast_rays(o3d.core.Tensor(rays))
    t_in = ans["t_hit"].numpy()
    hit_n = ans["primitive_normals"].numpy()
    facing = np.einsum("ij,ij->i", hit_n, Nh)
    redundant = (t_in < probe) & (facing > 0.3)
    wound = (d_hull > tau) & ~redundant
    print(f"  hull samples: {(d_hull > tau).sum():,} far from the measured surface, of which "
          f"{redundant[d_hull > tau].sum():,} sit just outside EXISTING surface (probe {probe:.4f})")
    print(f"  hull samples inside the wound: {wound.sum():,} / {len(Ph):,} "
          f"({wound.mean()*100:.1f}%) -- the rest is already measured")

    Pm, Nm, _ = outward_oriented_samples(meas, args.n_samples, centre)
    P = np.vstack([Pm, Ph[wound]])
    N = np.vstack([Nm, Nh[wound]])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.normals = o3d.utility.Vector3dVector(N)
    print(f"  fused cloud {len(P):,} points ({len(Pm):,} measured + {wound.sum():,} hull)")

    depths = [int(args.depth)] if args.depth != "auto" else [9, 8, 7]
    mesh = None
    for dep in depths:
        t0 = time.time()
        m, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=dep, width=0, scale=1.05, linear_fit=False)
        m.remove_duplicated_vertices(); m.remove_duplicated_triangles()
        m.remove_degenerate_triangles()
        lbl, cnt, _ = m.cluster_connected_triangles()
        lbl, cnt = np.asarray(lbl), np.asarray(cnt)
        m.remove_triangles_by_mask(lbl != cnt.argmax())
        m.remove_unreferenced_vertices()
        comp, genus, nb = _topology(m)
        print(f"  poisson depth {dep}: {len(m.triangles):,} faces, genus {genus}, "
              f"boundary edges {nb} ({time.time()-t0:.1f}s)")
        mesh = m
        if args.depth != "auto" or genus <= 4:
            break

    if not args.no_snap:
        V = np.asarray(mesh.vertices)
        cp = rc.compute_closest_points(o3d.core.Tensor(V.astype(np.float32)))["points"].numpy()
        off = np.linalg.norm(cp - V, axis=1)
        snap = off < tau
        V[snap] = cp[snap]
        mesh.vertices = o3d.utility.Vector3dVector(V)
        print(f"  snapped {snap.sum():,} observed vertices back onto the measured surface "
              f"(max move {off[snap].max() if snap.any() else 0:.5f})")

    if args.target_faces and len(mesh.triangles) > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
        mesh.remove_duplicated_vertices(); mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        # Decimation can shed slivers into their own components even when the input was a single
        # closed shell, so re-apply the largest-component filter here rather than leaving the
        # fragments for pymeshfix, which repairs by deleting and is happy to take the object with it.
        lbl2, cnt2, _ = mesh.cluster_connected_triangles()
        lbl2, cnt2 = np.asarray(lbl2), np.asarray(cnt2)
        if len(cnt2) > 1:
            mesh.remove_triangles_by_mask(lbl2 != cnt2.argmax())
            mesh.remove_unreferenced_vertices()
            print(f"  decimation left {len(cnt2)} components; kept the largest "
                  f"({cnt2.max():,} of {cnt2.sum():,} faces)")
        print(f"  decimated to {len(mesh.triangles):,} faces")

    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
    C = V[F].mean(1)
    dd = rc.compute_distance(o3d.core.Tensor(C.astype(np.float32))).numpy()
    obs = dd < tau
    q = np.percentile(dd[obs], [50, 90, 99]) if obs.any() else [float("nan")] * 3
    comp, genus, nb = _topology(mesh)
    stats = {
        "input_faces": int(len(meas.triangles)), "output_faces": int(len(F)),
        "hull_voxel_volume": hull_vol, "hull_views": int(used),
        "hull_samples": int(len(Ph)), "hull_samples_in_wound": int(wound.sum()),
        "extent": np.round(V.max(0) - V.min(0), 4).tolist(),
        "components": int(comp), "genus": int(genus), "boundary_edges": int(nb),
        "watertight": bool(mesh.is_watertight()),
        "frac_invented": float((~obs).mean()),
        "observed_dev_p50": float(q[0]), "observed_dev_p90": float(q[1]),
        "observed_dev_p99": float(q[2]),
    }
    print(f"\ncompleted mesh: {len(V):,} v  {len(F):,} f  components {comp} "
          f"genus {genus}  boundary edges {nb}")
    print(f"  extent {stats['extent']}  watertight {stats['watertight']}")
    print(f"  invented {(~obs).sum():,} faces ({(~obs).mean()*100:.1f}%)")
    print(f"  observed-region deviation from the measured surface: "
          f"p50 {q[0]:.5f}  p90 {q[1]:.5f}  p99 {q[2]:.5f}  ({q[2]/diag*100:.2f}% of diagonal)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"wrote {args.out}")
    if args.report_json:
        with open(args.report_json, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"wrote report {args.report_json}")


if __name__ == "__main__":
    main()
