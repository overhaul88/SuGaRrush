"""Per-face observation confidence rho, and the admissibility filter built on it.

Implements Sec 3 of morphogenetic_completion.md ("Confidence field: recovering the information the
pipeline discards") and the A_obs / wound partition of Sec 4.

    rho(f) = min(1, (1/kappa) * sum_c  v_c(f) * m_c(f) * <n_f, d_c>_+ * s_c(f))

Why this exists
---------------
Surface extraction treats every point it produces as evidence. It is not. In a region no camera
ever saw, nothing constrains the Gaussians, so junk accumulates there -- detached blobs and spikes
off the unobserved face. Screened Poisson then fits that junk with exactly the same weight as a
face seen from 300 views, and the closure wraps it into a smooth bulge. No octree depth, no repair
operator and no closure algorithm can fix that, because the error is not one of discretisation: it
is that non-evidence was admitted as evidence.

rho is the discriminator the pipeline was missing. Observed faces score ~1; floaters and the
genuinely unseen bottom both score ~0. Dropping the latter removes the floaters AND turns the
unobserved region into an honest hole, which the downstream closure spans minimally instead of
ballooning around debris.

Two things that must not be simplified away
-------------------------------------------
* **Visibility is a z-buffer test, not a frustum test.** scripts/carve_mesh.py counts a vertex as
  seen whenever it projects inside the image and lies in front of the camera. A floater tucked
  against the object's silhouette passes that from many views, which is precisely why it survives.
  Occlusion is the whole discriminator, so we rasterise a depth buffer per view and compare.
* **`unobserved` (never seen by any camera) is kept distinct from merely low rho** (Sec 3). They
  need opposite treatment: a grazing, few-view face is weak evidence to be down-weighted; a
  never-seen face is not evidence at all and must go. Conflating them either deletes real geometry
  or keeps the floaters.

Angular coverage is tracked separately from view count: ten frames over a 5 degree baseline
constrain a face far less than three over 90 degrees.
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gaussian_splatting"))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary  # noqa: E402


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]])


def load_cameras(scene):
    ex = read_extrinsics_binary(os.path.join(scene, "sparse/0/images.bin"))
    ic = read_intrinsics_binary(os.path.join(scene, "sparse/0/cameras.bin"))
    cams = []
    for im in ex.values():
        cam = ic[im.camera_id]
        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        elif cam.model == "SIMPLE_PINHOLE":
            fx = fy = cam.params[0]; cx, cy = cam.params[1:3]
        else:
            raise SystemExit(f"Camera model {cam.model} is not undistorted")
        cams.append(dict(name=im.name, R=qvec2rotmat(im.qvec), t=np.asarray(im.tvec),
                         fx=fx, fy=fy, cx=cx, cy=cy, w=cam.width, h=cam.height))
    return sorted(cams, key=lambda c: c["name"])



TARGET_VIEWS = 120


def auto_stride(n_cams, stride, target=TARGET_VIEWS):
    """Pick a camera stride that yields ~`target` views regardless of how many frames were kept.

    A FIXED stride silently changes the evidence with the capture length: object4 kept 499 frames so
    stride 4 gave 125 views, while a 240-frame capture at the same stride gives 60 -- half the
    angular evidence for rho and a looser visual hull, with nothing in the output saying so. Stride 0
    means auto. On 499 cameras this returns 4, reproducing the validated object4 run exactly.
    """
    if stride > 0:
        return stride
    # ROUND, do not floor. Flooring gives stride 1 for any count in [target, 2*target), so a
    # 236-camera capture would use all 236 views while claiming to target 120 -- harmless for
    # quality but a stated behaviour the code did not have. round() keeps 499 -> 4 (the validated
    # object4 stride) and sends 236 -> 2.
    return max(1, int(round(n_cams / max(target, 1))))


def view_crops(cams, V, n_faces, px_per_face=2.0, margin=4, max_side=1400, fixed_scale=0.0):
    """Per-view depth-buffer grids, cropped to the object and resolved so faces are not sub-pixel.

    Visibility here is a rendering: rasterise depth, project each face centroid, keep the face if
    nothing is nearer along its ray. That measures OCCLUSION only while a face owns roughly a
    pixel. Once many faces land inside one pixel, that pixel's depth is won by whichever of them
    is nearest and all its neighbours read as occluded by it -- manufacturing "never seen" on
    surface every camera saw perfectly well.

    The old grid was the FULL frame at a fixed 0.25x, which is doubly wrong. It is not
    resolution-aware, so the same setting means different things at different face counts; and it
    spends its pixels on empty background, because the object covers ~1.2% of the frame here.
    Measured on object6, silhouette a median 24,960 real-camera pixels:

        19,389-face mesh (Poisson depth 6):  6.2 front-facing faces per depth pixel
        72,278-face mesh (Poisson depth 7): 23.2 front-facing faces per depth pixel

    At 23 faces per pixel the filter punched the measured surface into 290 components with 6,270
    boundary edges (from 3 and 440); the closure had to bridge all of it and the final mesh came
    out genus 7 with visible fissures, where the coarser run gave genus 0.

    Cropping to the object's projected bounding box is EXACT, not an approximation: the occluder
    geometry handed to the raycasting scene is untouched, and every ray we query passes through a
    face centroid, which lies inside the crop by construction. It is also cheaper than what it
    replaces -- the crop needs ~72k rays per view at 2 px/face where the old full frame at 0.25x
    spent 127k to get 0.04 px/face.

    Roughly half the faces point at the camera at once, so solve
    bbox_area * s^2 >= px_per_face * n_faces / 2 for s.
    """
    crops = []
    for c in cams:
        P = V @ c["R"].T + c["t"]
        z = P[:, 2]
        ok = z > 1e-6
        if not ok.any():
            crops.append(dict(W=8, H=8, s=1.0, u0=0.0, v0=0.0))
            continue
        uf = c["fx"] * P[ok, 0] / z[ok] + c["cx"]
        vf = c["fy"] * P[ok, 1] / z[ok] + c["cy"]
        # Clamp to the real sensor: geometry projected off-frame is not observable anyway, and an
        # extreme outlier behind the camera would otherwise blow the crop up to nothing useful.
        u0 = float(np.clip(uf.min(), 0, c["w"])) - margin
        v0 = float(np.clip(vf.min(), 0, c["h"])) - margin
        u1 = float(np.clip(uf.max(), 0, c["w"])) + margin
        v1 = float(np.clip(vf.max(), 0, c["h"])) + margin
        bw, bh = max(u1 - u0, 1.0), max(v1 - v0, 1.0)

        s = fixed_scale if fixed_scale > 0 else float(
            np.sqrt(px_per_face * max(n_faces, 1) / 2.0 / (bw * bh)))
        s = min(s, max_side / bw, max_side / bh)
        W = max(8, int(np.ceil(bw * s)))
        H = max(8, int(np.ceil(bh * s)))
        crops.append(dict(W=W, H=H, s=s, u0=u0, v0=v0))
    return crops


def _face_adjacency(T):
    """Faces sharing an edge, padded with -1."""
    from collections import defaultdict
    e = defaultdict(list)
    for fi, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e[(a, b) if a < b else (b, a)].append(fi)
    nbr = [[] for _ in range(len(T))]
    for fs in e.values():
        if len(fs) == 2:
            nbr[fs[0]].append(fs[1]); nbr[fs[1]].append(fs[0])
    w = max((len(x) for x in nbr), default=0)
    out = np.full((len(T), w), -1, dtype=np.int64)
    for i, x in enumerate(nbr):
        out[i, :len(x)] = x
    return out


def solid_angle_coverage(dirs):
    """Rough angular coverage of a set of unit view directions, as a fraction of the sphere.

    Bins directions onto a coarse spherical grid and reports the occupied fraction. Cheap, and only
    needs to distinguish 'a narrow arc' from 'a wide baseline'.
    """
    if len(dirs) == 0:
        return 0.0
    theta = np.arccos(np.clip(dirs[:, 2], -1, 1))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    ti = np.clip((theta / np.pi * 8).astype(int), 0, 7)
    pi_ = np.clip(((phi + np.pi) / (2 * np.pi) * 16).astype(int), 0, 15)
    return len(set(zip(ti.tolist(), pi_.tolist()))) / 128.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--scene", required=True, help="COLMAP scene dir (sparse/0 + masks/)")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--out", default=None, help="filtered mesh (A_obs kept)")
    ap.add_argument("--rho-ply", default=None, help="rho-coloured mesh for inspection")
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--rho0", type=float, default=0.15,
                    help="A_obs threshold; faces below this are the wound W (default 0.15)")
    ap.add_argument("--min-views", type=int, default=2,
                    help="a face visible from fewer than this many cameras is treated as unobserved")
    ap.add_argument("--view-stride", type=int, default=0,
                    help="use every Nth camera (angular coverage matters more than count); "
                         "0 = auto, choosing a stride that yields ~120 views for any frame count")
    ap.add_argument("--raster-scale", type=float, default=0.0,
                    help="depth-buffer resolution as a fraction of the real camera. 0 = AUTO: "
                         "resolve it from the mesh so faces are not sub-pixel (see "
                         "view_crops). A fixed value is only correct for the face count "
                         "it was chosen at")
    ap.add_argument("--px-per-face", type=float, default=2.0,
                    help="target depth-buffer pixels per front-facing face when --raster-scale is "
                         "auto")
    ap.add_argument("--depth-tol", type=float, default=0.01,
                    help="z-buffer tolerance as a fraction of the object diagonal")
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="normaliser: weighted view sum at which a face reaches rho=1 (0 = derive from data)")
    ap.add_argument("--drop-unobserved", action="store_true",
                    help="delete W from the mesh and write --out (otherwise only measure)")
    ap.add_argument("--drop-mode", choices=["unobserved", "rho"], default="unobserved",
                    help="which faces the wound comprises: never-seen only (default, "
                         "conservative) or additionally everything below --rho0")
    ap.add_argument("--passes", type=int, default=3,
                    help="iterative re-observation passes. Removing an occluder is a new "
                         "observation, so one pass deletes real surface that junk was shadowing.")
    ap.add_argument("--junk-component-frac", type=float, default=0.02,
                    help="a component smaller than this fraction of the mesh may be seeded as junk")
    ap.add_argument("--junk-seen-frac", type=float, default=0.2,
                    help="...and is junk only if less than this fraction of it is ever seen")
    ap.add_argument("--keep-ratio", type=float, default=0.75,
                    help="visual-hull test: a face must project inside the silhouette in at least\n"
                         "this fraction of the views that can see it, else it is outside the hull")
    ap.add_argument("--label-smooth", type=int, default=3,
                    help="majority-vote sweeps over the dual graph (cheap Potts MRF, Sec 5)")
    ap.add_argument("--save-npz", default=None, help="dump per-face rho/n_vis for analysis")
    ap.add_argument("--keep-largest", action="store_true",
                    help="after filtering, keep only the largest connected component")
    args = ap.parse_args()

    masks_dir = args.masks or os.path.join(args.scene, "masks")
    mesh = o3d.io.read_triangle_mesh(args.mesh, enable_post_processing=True)
    if len(mesh.triangles) == 0:
        raise SystemExit("empty mesh")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_triangle_normals()

    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    FN = np.asarray(mesh.triangle_normals)
    C = V[T].mean(axis=1)                      # face centroids
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    tol = args.depth_tol * diag
    print(f"mesh: {len(T):,} faces, {len(V):,} verts, diagonal {diag:.3f}")

    _all_cams = load_cameras(args.scene)
    _stride = auto_stride(len(_all_cams), args.view_stride)
    cams = _all_cams[::_stride]

    crops = view_crops(cams, V, len(T), args.px_per_face, fixed_scale=args.raster_scale)
    _s = np.array([cr["s"] for cr in crops])
    _px = np.array([cr["W"] * cr["H"] for cr in crops]) / max(len(T) / 2.0, 1.0)
    print(f"cameras: {len(cams)} of {len(_all_cams)} (stride {_stride}"
          f"{', auto' if args.view_stride <= 0 else ''}), tolerance {tol:.4f}")
    print(f"  depth buffer cropped to the object: median {int(np.median([cr['W'] for cr in crops]))}"
          f"x{int(np.median([cr['H'] for cr in crops]))} px at {np.median(_s):.2f}x the real camera"
          f"  ->  {np.median(_px):.2f} px per front-facing face"
          + ("" if args.raster_scale <= 0 else f"  (--raster-scale {args.raster_scale:g} forced)"))

    def observe(occluder_mask, quiet=False):
        """Accumulate evidence for EVERY face, using only `occluder_mask` faces as occluders.

        Splitting the query set from the occluder set is the whole point. rho computed with the
        junk still in place is self-defeating: a floater standing in front of a corner occludes it,
        the corner reads as never-seen, and we delete real surface for being hidden by the very
        thing we are trying to remove. Measured on object4: 1,813 of 13,467 deleted faces (13.5%)
        become visible from >=5 views once the junk is dropped from the occluder set.
        """
        occ = o3d.geometry.TriangleMesh(mesh)
        occ.remove_triangles_by_mask(~occluder_mask)
        occ.remove_unreferenced_vertices()
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occ))
        return _accumulate(rc, quiet)

    def _accumulate(scene_rc, quiet):
        acc = np.zeros(len(T))                     # weighted evidence sum
        n_vis = np.zeros(len(T), dtype=np.int32)   # visible AND inside the silhouette
        n_geom = np.zeros(len(T), dtype=np.int32)  # visible at all (mask ignored)
        dir_sum = np.zeros((len(T), 3))            # for angular spread
        dir_list = [[] for _ in range(len(T))] if len(T) < 400_000 else None

        used = 0
        for ci, c in enumerate(cams):
            mp = os.path.join(masks_dir, os.path.splitext(c["name"])[0] + ".png")
            if not os.path.isfile(mp):
                continue
            mask = np.array(Image.open(mp).convert("L")).astype(np.float32) / 255.0

            cr = crops[ci]
            W, H, s, u0, v0 = cr["W"], cr["H"], cr["s"], cr["u0"], cr["v0"]
            # Principal point shifted by the crop origin, so the grid covers only the object.
            K = np.array([[c["fx"] * s, 0, (c["cx"] - u0) * s],
                          [0, c["fy"] * s, (c["cy"] - v0) * s],
                          [0, 0, 1]], dtype=np.float64)
            E = np.eye(4); E[:3, :3] = c["R"]; E[:3, 3] = c["t"]

            rays = scene_rc.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(K),
                                                extrinsic_matrix=o3d.core.Tensor(E),
                                                width_px=W, height_px=H)
            depth = scene_rc.cast_rays(rays)["t_hit"].numpy()      # distance from eye per pixel

            # project face centroids into this view
            P = C @ c["R"].T + c["t"]
            z = P[:, 2]
            front = z > 1e-6
            # uf, vf are real-camera pixel coordinates; the crop only changes where they land in
            # the depth buffer, never how the mask is sampled.
            uf = np.full(len(C), -1e9); vf = np.full(len(C), -1e9)
            uf[front] = c["fx"] * P[front, 0] / z[front] + c["cx"]
            vf[front] = c["fy"] * P[front, 1] / z[front] + c["cy"]
            px = np.round((uf - u0) * s).astype(np.int64)
            py = np.round((vf - v0) * s).astype(np.int64)
            inb = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)

            # z-buffer test: the face is visible only if nothing is nearer along its own ray
            dist = np.linalg.norm(P, axis=1)
            vis = np.zeros(len(C), dtype=bool)
            idx = np.nonzero(inb)[0]
            if len(idx):
                dbuf = depth[py[idx], px[idx]]
                vis[idx] = dist[idx] <= (dbuf + tol)

            # mask agreement, sampled at full mask resolution
            mh, mw = mask.shape
            mval = np.zeros(len(C))
            if len(idx):
                mx = np.clip(np.round(uf[idx] / c["w"] * mw).astype(np.int64), 0, mw - 1)
                my = np.clip(np.round(vf[idx] / c["h"] * mh).astype(np.int64), 0, mh - 1)
                mval[idx] = mask[my, mx]

            # Incidence: a face seen edge-on constrains almost nothing. Sec 3 writes this as <n_f,d_c>_+,
            # which presumes outward-oriented normals -- but SuGaR's output winding is unreliable (on
            # object4 ~90% of face normals point inward), and the clamp would then zero out almost every
            # real observation. |<n,d>| is winding-invariant, and facing is already handled correctly by
            # the z-buffer above: a genuinely back-facing patch is occluded by the front surface.
            eye = -c["R"].T @ c["t"]
            d = eye - C
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            incid = np.abs(np.einsum("ij,ij->i", FN, d))

            contrib = vis * mval * incid
            acc += contrib
            hard = vis & (mval > 0.5)
            n_vis += hard
            n_geom += vis
            dir_sum[hard] += d[hard]
            if dir_list is not None:
                for j in np.nonzero(hard)[0]:
                    dir_list[j].append(d[j])
            used += 1
            if ci % 25 == 0 and not quiet:
                print(f"  view {ci}/{len(cams)}", flush=True)

        print(f"used {used} masked views")
        return acc, n_vis, n_geom, dir_sum, dir_list, used


    # ---- iterative re-observation -------------------------------------------------------------
    # Removing an occluder IS a new observation (Sec 4: completion is re-observation), so rho must
    # be iterated rather than computed once. Junk seeds -- small detached components with no
    # observational support -- are removed permanently and never restored, because once alone in the
    # open a floater becomes "visible" and a purely monotone rule would resurrect it. Everything
    # else is re-evaluated each pass against the shrinking occluder set, so real surface that was
    # merely standing in a floater's shadow comes back.
    face_comp = np.asarray(mesh.cluster_connected_triangles()[0])
    comp_size = np.bincount(face_comp, minlength=face_comp.max() + 1)

    # The occluder set must only ever SHRINK, or the scheme oscillates: feeding back the visible set
    # re-adds faces as occluders and re-shadows what the previous pass recovered (measured: 67,971
    # -> 72,613 -> 68,422). Occluders are therefore "everything except junk", and junk only grows,
    # so visibility is monotone non-decreasing and the loop converges.
    junk = np.zeros(len(T), bool)
    acc = n_vis = dir_sum = dir_list = None
    used = 0
    prev_seen = -1
    for p in range(max(1, args.passes)):
        acc, n_vis, n_geom, dir_sum, dir_list, used = observe(~junk, quiet=(p > 0))
        seen = n_vis >= args.min_views
        # A component that is both small and essentially never seen is not the object. Seeding is
        # what keeps the loop honest: without a permanent junk set, a floater left alone in the open
        # becomes "visible" once its neighbours are gone and a monotone rule would resurrect it.
        small = comp_size[face_comp] < args.junk_component_frac * len(T)
        comp_seen = np.zeros(len(comp_size))
        np.add.at(comp_seen, face_comp, seen.astype(float))
        frac_seen = comp_seen[face_comp] / np.maximum(comp_size[face_comp], 1)
        # Visual-hull test. A lump fused to the object is the OUTERMOST surface, so it is visible
        # and reads as observed -- while shadowing the real geometry behind it. That is how a corner
        # got deleted while the lump that hid it survived. What such a lump fails is the silhouette:
        # protruding past the true surface, it projects OUTSIDE the U2Net mask in some views. Faces
        # seen often but agreeing with the silhouette in less than keep-ratio of those views are
        # therefore outside the visual hull and are junk regardless of connectivity. This is the
        # test carve_mesh.py applies on the default path and which --gaussian-prune skipped.
        with np.errstate(invalid="ignore", divide="ignore"):
            hull_ratio = np.where(n_geom > 0, n_vis / np.maximum(n_geom, 1), 1.0)
        outside_hull = (n_geom >= args.min_views) & (hull_ratio < args.keep_ratio)
        new_junk = ((small & (frac_seen < args.junk_seen_frac)) | outside_hull) & ~junk
        if outside_hull.any() and p == 0:
            print(f"  outside visual hull: {outside_hull.sum():,} faces "
                  f"(silhouette agreement < {args.keep_ratio:g})")
        gained = int(seen.sum() - prev_seen) if prev_seen >= 0 else 0
        print(f"  pass {p+1}: visible {seen.sum():,} / {len(T):,}"
              + (f"   (+{gained:,} recovered from occlusion shadow)" if p > 0 else "")
              + (f"   +{new_junk.sum():,} junk seeded" if new_junk.any() else ""))
        prev_seen = seen.sum()
        if not new_junk.any():
            break
        junk |= new_junk

    # kappa normalises so a comfortably covered face reaches 1. A hardcoded constant would not
    # transfer across view counts or object scales, so derive it from this capture unless told
    # otherwise: the 75th percentile of the evidence sum is "comfortably covered" for this object.
    kappa = args.kappa
    if kappa <= 0:
        kappa = max(float(np.percentile(acc, 75)), 1e-9)
        print(f"kappa auto = {kappa:.2f} (75th percentile of the weighted view sum)")
    rho = np.minimum(1.0, acc / kappa)
    q = np.percentile(n_vis, [10, 50, 90])
    print(f"visible-view count per face: p10 {q[0]:.0f}  median {q[1]:.0f}  p90 {q[2]:.0f}  "
          f"(of {used} views used)")
    print(f"weighted evidence sum: median {np.median(acc):.2f}  p90 {np.percentile(acc,90):.2f}")
    unobserved = n_vis < args.min_views
    # angular spread: |sum of unit dirs| / count near 1 means all views from one direction
    with np.errstate(invalid="ignore", divide="ignore"):
        concentration = np.where(n_vis > 0, np.linalg.norm(dir_sum, axis=1) / np.maximum(n_vis, 1), 1.0)
    ang_cov = None
    if dir_list is not None:
        ang_cov = np.array([solid_angle_coverage(np.array(dl)) if dl else 0.0 for dl in dir_list])

    # Sec 3 insists the hard predicate stay distinct from low confidence, and the distinction
    # decides what we may delete. A never-seen face is not evidence -- remove it. A grazing,
    # few-view face is weak evidence that ought to be DOWN-WEIGHTED, but the downstream closure can
    # only keep or drop, so deleting on low rho would throw away real, merely-awkward geometry.
    # Default therefore cuts on the hard predicate only; --drop-mode rho opts into the stricter cut.
    if args.drop_mode == "rho":
        obs = (~unobserved) & (rho >= args.rho0)
    else:
        obs = ~unobserved
    obs &= ~junk

    # Sec 5 solves object/scene assignment as a Potts MRF on the dual graph rather than by
    # thresholding faces independently, because an independent threshold produces speckle: single
    # faces flipped inside an otherwise coherent region. Iterated majority voting over the dual
    # graph is the cheap version of the same idea and removes the speckle in both directions --
    # pinholes punched into measured surface, and stray survivors inside the wound.
    if args.label_smooth > 0:
        nbr_f = _face_adjacency(T)
        for _ in range(args.label_smooth):
            valid = nbr_f >= 0
            votes = np.where(valid, obs[np.where(valid, nbr_f, 0)], False).sum(1)
            deg = valid.sum(1)
            flip_on = (~obs) & (votes >= np.maximum(deg - 1, 1)) & ~junk   # surrounded by observed
            flip_off = obs & (votes == 0) & (deg > 0)                     # island inside the wound
            if not (flip_on.any() or flip_off.any()):
                break
            obs = (obs | flip_on) & ~flip_off
        print(f"  label smoothing ({args.label_smooth} sweeps): A_obs now {obs.sum():,}")
    print(f"\nrho: median {np.median(rho):.3f}  mean {rho.mean():.3f}")
    print(f"  faces never seen (or < {args.min_views} views): {unobserved.sum():,} "
          f"({unobserved.mean()*100:.1f}%)")
    print(f"  faces below rho0={args.rho0}: {(rho < args.rho0).sum():,} ({(rho<args.rho0).mean()*100:.1f}%)")
    print(f"  A_obs (observed): {obs.sum():,} ({obs.mean()*100:.1f}%)   "
          f"wound W: {(~obs).sum():,} ({(~obs).mean()*100:.1f}%)")
    if ang_cov is not None:
        print(f"  angular coverage of observed faces: median {np.median(ang_cov[obs]):.3f} "
              f"of the sphere; wound faces: {np.median(ang_cov[~obs]):.3f}")
    print(f"  mean view-direction concentration (1 = single direction): "
          f"observed {concentration[obs].mean():.2f}, wound {concentration[~obs].mean():.2f}")

    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez(args.save_npz, rho=rho, n_vis=n_vis, unobserved=unobserved, obs=obs,
                 centroids=C, acc=acc)
        print(f"  wrote per-face arrays -> {args.save_npz}")

    if args.rho_ply:
        vis_mesh = o3d.geometry.TriangleMesh(mesh)
        # per-vertex colour from the max rho of incident faces: blue = wound, red = observed
        vr = np.zeros(len(V))
        np.maximum.at(vr, T[:, 0], rho); np.maximum.at(vr, T[:, 1], rho); np.maximum.at(vr, T[:, 2], rho)
        col = np.stack([vr, np.zeros_like(vr), 1.0 - vr], axis=1)
        vis_mesh.vertex_colors = o3d.utility.Vector3dVector(col)
        os.makedirs(os.path.dirname(os.path.abspath(args.rho_ply)), exist_ok=True)
        o3d.io.write_triangle_mesh(args.rho_ply, vis_mesh)
        print(f"  wrote rho visualisation -> {args.rho_ply}")

    stats = {
        "mesh": args.mesh, "n_faces": int(len(T)), "n_views_used": used,
        "rho0": args.rho0, "min_views": args.min_views, "kappa": float(kappa),
        "frac_unobserved": float(unobserved.mean()), "frac_below_rho0": float((rho < args.rho0).mean()),
        "frac_A_obs": float(obs.mean()), "rho_median": float(np.median(rho)),
    }

    if args.drop_unobserved and args.out:
        out_mesh = o3d.geometry.TriangleMesh(mesh)
        out_mesh.remove_triangles_by_mask(~obs)
        out_mesh.remove_unreferenced_vertices()
        if args.keep_largest and len(out_mesh.triangles):
            lbl, cnt, _ = out_mesh.cluster_connected_triangles()
            if len(np.asarray(cnt)):
                keep = int(np.argmax(np.asarray(cnt)))
                n_before = len(out_mesh.triangles)
                out_mesh.remove_triangles_by_mask(np.asarray(lbl) != keep)
                out_mesh.remove_unreferenced_vertices()
                print(f"  largest component: {n_before:,} -> {len(out_mesh.triangles):,} triangles "
                      f"({len(np.asarray(cnt))} components before)")
        if len(out_mesh.triangles) == 0:
            raise SystemExit("rho filter removed everything -- lower --rho0")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        o3d.io.write_triangle_mesh(args.out, out_mesh)
        stats["n_faces_kept"] = int(len(out_mesh.triangles))
        print(f"kept {len(out_mesh.triangles):,} / {len(T):,} faces -> {args.out}")

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"  wrote report {args.report_json}")


if __name__ == "__main__":
    main()
