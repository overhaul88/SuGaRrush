"""Fit CAD primitives to a segmented scan mesh, and infer how they mate.

Stage 7b + 7d. Consumes the surface groups found by segment_parts.py, abstracts
each to a parametric part, then reasons about contacts between parts.

Design commitments, and why:

  * **Thickness gating.** Sheet-metal / Meccano parts share one stock thickness.
    On object1 the plane-pair thicknesses cluster hard at 0.016-0.027 with a tail
    at 0.04-0.10; the tail is spurious pairings across unrelated surfaces. Taking
    the modal thin value and rejecting outliers is a strong, cheap, class-correct
    filter.
  * **Single-sided plates are allowed, and flagged.** The scan never saw under the
    cargo shelf, so that plate has no antiparallel partner. Refusing to emit it
    would lose a real part; inventing a thickness silently would be dishonest. We
    borrow the modal thickness and mark the part `thickness_inferred: true`.
  * **Nothing that fails to fit is faked.** Unfit regions are reported as leftover
    area, not extruded into confident geometry.

    python scripts/fit_primitives.py --mesh object1_final.ply --out parts.json
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_parts import (face_adjacency, grow_regions, plane_of,  # noqa: E402
                           merge_coplanar, min_area_rect)


def fit_cylinder(pts, nrms):
    """Axis is the direction every surface normal is perpendicular to.

    For a cylinder the normals span the plane orthogonal to the axis, so the
    axis is the least-explained direction of the normal set -- the smallest
    singular vector. Radius and centre then come from a circle fit in that plane.
    """
    _, s, vt = np.linalg.svd(nrms - 0, full_matrices=False)
    axis = vt[2]
    axis /= np.linalg.norm(axis)
    # basis perpendicular to the axis
    a = np.array([1.0, 0, 0])
    if abs(a @ axis) > 0.9:
        a = np.array([0, 1.0, 0])
    u = a - axis * (a @ axis); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    P = np.c_[pts @ u, pts @ v]
    # algebraic circle fit (Kasa)
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-12)))
    resid = float(np.abs(np.linalg.norm(P - [cx, cy], axis=1) - r).mean())
    t = pts @ axis
    centre = cx * u + cy * v + axis * ((t.min() + t.max()) / 2)

    # Angular coverage is the discriminator that separates a real cylinder from a
    # noisy blob. A genuine barrel has surface normals fanning right around the
    # axis; a lump of scan noise fits a circle algebraically but its normals only
    # occupy a narrow wedge. Without this test the fitter happily reports the flat
    # mast as a 1.8-long cylinder.
    nn = nrms - np.outer(nrms @ axis, axis)
    keep = np.linalg.norm(nn, axis=1) > 1e-9
    ang = np.arctan2(nn[keep] @ v, nn[keep] @ u)
    hist = np.histogram(ang, bins=36, range=(-np.pi, np.pi))[0]
    coverage = float((hist > max(1, 0.002 * len(ang))).sum() / 36.0)

    return dict(axis=axis.tolist(), radius=r, length=float(t.ptp()),
                centre=centre.tolist(), residual=resid, coverage=coverage,
                cylindricity=float(s[2] / max(s[0], 1e-9)))


def infer_mates(parts, tol_ang=15.0, tol_dist=0.06):
    """Contact/mating reasoning over fitted primitives.

    Mirrors classical automatic-geometric-constraint practice: match functional
    surfaces (axes, planar faces) that are close and compatibly oriented.
    """
    mates = []
    for i, a in enumerate(parts):
        for j, b in enumerate(parts):
            if j <= i:
                continue
            ta, tb = a["type"], b["type"]
            if ta == "cylinder" and tb == "cylinder":
                ax_a = np.array(a["axis"]); ax_b = np.array(b["axis"])
                ang = np.degrees(np.arccos(min(1, abs(ax_a @ ax_b))))
                d = np.array(a["centre"]) - np.array(b["centre"])
                perp = np.linalg.norm(d - ax_a * (d @ ax_a))
                if ang < tol_ang and perp < tol_dist:
                    mates.append(dict(a=a["id"], b=b["id"], type="coaxial",
                                      axis_angle_deg=round(ang, 2),
                                      axis_offset=round(float(perp), 4)))
            elif {ta, tb} == {"plate", "cylinder"}:
                pl, cy = (a, b) if ta == "plate" else (b, a)
                n = np.array(pl["normal"]); ax = np.array(cy["axis"])
                d = np.array(cy["centre"]) - np.array(pl["centre"])
                if abs(n @ ax) > np.cos(np.radians(tol_ang)):
                    if abs(n @ d) < pl["thickness"] / 2 + cy["length"] / 2:
                        mates.append(dict(a=pl["id"], b=cy["id"], type="insertion",
                                          note="cylinder axis normal to plate face"))
                elif abs(n @ ax) < np.sin(np.radians(tol_ang)):
                    if np.linalg.norm(d) < tol_dist * 4:
                        mates.append(dict(a=pl["id"], b=cy["id"], type="tangent_contact"))
            else:  # plate / plate
                na = np.array(a["normal"]); nb = np.array(b["normal"])
                cosang = abs(na @ nb)
                d = np.array(a["centre"]) - np.array(b["centre"])
                gap = abs(na @ d)
                if cosang > np.cos(np.radians(tol_ang)):
                    if gap < (a["thickness"] + b["thickness"]) * 1.6 + tol_dist * 0.4:
                        mates.append(dict(a=a["id"], b=b["id"], type="planar_contact",
                                          gap=round(float(gap), 4)))
                elif cosang < np.sin(np.radians(tol_ang * 2)):
                    if np.linalg.norm(d) < tol_dist * 6:
                        mates.append(dict(a=a["id"], b=b["id"], type="perpendicular_joint",
                                          angle_deg=round(float(np.degrees(np.arccos(cosang))), 1)))
    return mates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grow-angle", type=float, default=20.0)
    ap.add_argument("--merge-angle", type=float, default=20.0)
    ap.add_argument("--merge-dist", type=float, default=0.045)
    ap.add_argument("--min-area-frac", type=float, default=0.0015)
    ap.add_argument("--max-thickness", type=float, default=0.12)
    ap.add_argument("--thickness-tol", type=float, default=0.45,
                    help="accept plate thickness within this relative band of the mode")
    ap.add_argument("--min-cyl-area-frac", type=float, default=0.006)
    ap.add_argument("--cyl-resid", type=float, default=0.09,
                    help="max mean radial error / radius")
    ap.add_argument("--cyl-coverage", type=float, default=0.55,
                    help="min fraction of the 360deg around the axis the normals span")
    ap.add_argument("--cyl-flatness", type=float, default=0.25,
                    help="max normal-set flatness; higher means the patch is really a plane")
    ap.add_argument("--scale", type=float, default=None,
                    help="mm per scene unit. REQUIRED for CAD export; omit to stay unitless")
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(args.mesh)
    m.remove_duplicated_vertices(); m.compute_triangle_normals()
    V = np.asarray(m.vertices); T = np.asarray(m.triangles); N = np.asarray(m.triangle_normals)
    area = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1) / 2
    total = area.sum()
    adj = face_adjacency(T)
    lab, n_reg = grow_regions(T, N, area, adj, args.grow_angle)

    groups = {}
    for i in range(n_reg):
        sel = lab == i
        a = area[sel].sum()
        if a < args.min_area_frac * total:
            continue
        pts = V[T[sel]].reshape(-1, 3)
        n, d = plane_of(pts, N[sel].mean(0))
        groups[i] = dict(n=n, d=d, centroid=pts.mean(0), area=float(a), faces=np.nonzero(sel)[0])
    merged = merge_coplanar(groups, args.merge_angle, args.merge_dist)
    surfaces = []
    for root, mem in merged.items():
        faces = np.concatenate([groups[k]["faces"] for k in mem])
        pts = V[T[faces]].reshape(-1, 3)
        n, d = plane_of(pts, np.vstack([groups[k]["n"] for k in mem]).mean(0))
        surfaces.append(dict(n=n, d=d, area=float(area[faces].sum()), faces=faces,
                             centroid=pts.mean(0)))
    surfaces.sort(key=lambda s: -s["area"])
    print(f"{len(surfaces)} merged surfaces from {n_reg:,} patches")

    # ---- plate pairing -------------------------------------------------------
    used, raw_plates = set(), []
    for i, a in enumerate(surfaces):
        if i in used:
            continue
        best = None
        for j, b in enumerate(surfaces):
            if j <= i or j in used or a["n"] @ b["n"] > -np.cos(np.radians(args.merge_angle * 2)):
                continue
            t = abs(a["n"] @ b["centroid"] + a["d"])
            if t > args.max_thickness:
                continue
            score = (a["area"] + b["area"]) / (t + 1e-6)
            if best is None or score > best[0]:
                best = (score, j, t)
        if best is None:
            continue
        _, j, t = best
        used.update({i, j})
        raw_plates.append((i, j, t, np.concatenate([a["faces"], surfaces[j]["faces"]])))

    ths = np.array([p[2] for p in raw_plates])
    if len(ths) == 0:
        raise SystemExit("no plane pairs found; loosen --merge-angle/--max-thickness")
    mode_t = float(np.median(ths[ths <= np.percentile(ths, 60)]))
    lo, hi = mode_t * (1 - args.thickness_tol), mode_t * (1 + args.thickness_tol)
    print(f"\nplane-pair thicknesses: {np.round(np.sort(ths),4).tolist()}")
    print(f"modal sheet thickness {mode_t:.4f} -> accepting [{lo:.4f}, {hi:.4f}]")

    parts, pid = [], 0
    claimed = set()
    for (i, j, t, faces) in raw_plates:
        if not (lo <= t <= hi):
            print(f"  reject plate (t={t:.4f} off-mode)")
            continue
        n = surfaces[i]["n"]
        u = np.array([1.0, 0, 0]); u = u - n * (u @ n)
        if np.linalg.norm(u) < 1e-6:
            u = np.array([0, 1.0, 0]) - n * (np.array([0, 1.0, 0]) @ n)
        u /= np.linalg.norm(u); v = np.cross(n, u)
        pts = V[T[faces]].reshape(-1, 3)
        ctr2, e1, e2, w, h = min_area_rect(pts @ np.c_[u, v])
        parts.append(dict(id=f"plate_{pid}", type="plate", normal=n.tolist(),
                          thickness=float(t), thickness_inferred=False,
                          length=float(w), width=float(h), centre=pts.mean(0).tolist(),
                          axis_u=(e1[0] * u + e1[1] * v).tolist(),
                          axis_v=(e2[0] * u + e2[1] * v).tolist(),
                          area=float(area[faces].sum())))
        claimed.update(faces.tolist()); pid += 1

    # ---- single-sided plates (occluded underside) ----------------------------
    for i, s in enumerate(surfaces):
        if i in used or s["area"] < 0.012 * total:
            continue
        n = s["n"]; faces = s["faces"]
        u = np.array([1.0, 0, 0]); u = u - n * (u @ n)
        if np.linalg.norm(u) < 1e-6:
            u = np.array([0, 1.0, 0]) - n * (np.array([0, 1.0, 0]) @ n)
        u /= np.linalg.norm(u); v = np.cross(n, u)
        pts = V[T[faces]].reshape(-1, 3)
        ctr2, e1, e2, w, h = min_area_rect(pts @ np.c_[u, v])
        parts.append(dict(id=f"plate_{pid}", type="plate", normal=n.tolist(),
                          thickness=mode_t, thickness_inferred=True,
                          length=float(w), width=float(h), centre=pts.mean(0).tolist(),
                          axis_u=(e1[0] * u + e1[1] * v).tolist(),
                          axis_v=(e2[0] * u + e2[1] * v).tolist(),
                          area=float(s["area"])))
        claimed.update(faces.tolist()); pid += 1
        print(f"  single-sided plate {parts[-1]['id']} "
              f"{w:.3f}x{h:.3f} (thickness inferred from mode)")

    # ---- cylinders on what is left ------------------------------------------
    leftover = np.setdiff1d(np.arange(len(T)), np.fromiter(claimed, int, len(claimed)))
    print(f"\nleftover faces for cylinder fitting: {len(leftover):,} "
          f"({100*len(leftover)/len(T):.0f}%)")
    sub_lab, sub_n = grow_regions(T[leftover], N[leftover], area[leftover],
                                  face_adjacency(T[leftover]), 45.0)
    cid = 0
    for k in range(sub_n):
        sel = leftover[sub_lab == k]
        a = area[sel].sum()
        if a < args.min_cyl_area_frac * total:
            continue
        pts = V[T[sel]].reshape(-1, 3)
        cyl = fit_cylinder(pts, N[sel])
        # A cylinder must be round (small radial residual), barrel-like (normals
        # wrapping the axis), and not a disguised plane (low cylindricity ratio).
        if cyl["radius"] < 1e-3 or cyl["residual"] / cyl["radius"] > args.cyl_resid:
            continue
        if cyl["coverage"] < args.cyl_coverage:
            continue
        if cyl["cylindricity"] > args.cyl_flatness:
            continue
        if cyl["length"] < cyl["radius"] * 0.35:
            continue
        cyl.update(id=f"cyl_{cid}", type="cylinder", area=float(a))
        parts.append(cyl); cid += 1
        print(f"  cylinder {cyl['id']}: r={cyl['radius']:.4f} L={cyl['length']:.3f} "
              f"axis={np.round(cyl['axis'],2)} resid/r={cyl['residual']/cyl['radius']:.3f} cov={cyl['coverage']:.2f}")

    mates = infer_mates(parts)
    print(f"\nfitted {len(parts)} parts, inferred {len(mates)} mates")
    for m_ in mates:
        print(f"  {m_['a']:<10} -- {m_['type']:<20} -- {m_['b']}")

    explained = sum(p["area"] for p in parts)
    print(f"\narea explained by primitives: {100*explained/total:.0f}%")

    out = dict(source_mesh=os.path.abspath(args.mesh), units="scene",
               mm_per_unit=args.scale, sheet_thickness_mode=mode_t,
               total_area=float(total), area_explained_frac=float(explained / total),
               parts=parts, mates=mates)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    if args.scale is None:
        print("NOTE: unitless. Supply --scale (mm per scene unit) before CAD export.")


if __name__ == "__main__":
    main()
