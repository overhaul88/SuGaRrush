"""Segment a scanned object mesh into primitive-coherent surface groups.

Stage 7a/7b groundwork. Pure geometry -- no learned model, no GPU -- because the
mesh is a noisy photogrammetric scan and the parts we care about (Meccano plates,
a pipe, wheels) are exactly the shapes normals describe well.

Three steps, each fixing a measured failure of the previous one:

  1. Region growing over face adjacency with a normal-deviation criterion.
     Position-only RANSAC is disqualified here: on object1 it fitted five planes
     with "thickness" 0.88-1.00, i.e. spanning the whole object, because
     open3d's segment_plane ignores normals and connectivity.
  2. Coplanar merging. Region growing alone shattered object1 into 8,380 patches
     whose top 10 covered only 27% of the area -- scan noise breaks surfaces into
     slivers that belong to one physical face.
  3. Plane-pair detection. A plate is not one surface, it is two antiparallel
     surfaces a thickness apart. Finding that pair is what makes the part a solid
     rather than a shell, and gives the thickness honestly (a tilted plate's
     axis-aligned bbox says nothing useful).

    python scripts/segment_parts.py --mesh object1_final.ply --out parts_raw.json \
        --color-out segmented.ply
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import open3d as o3d


def face_adjacency(T):
    edge = defaultdict(list)
    for f, (a, b, c) in enumerate(T):
        for e in ((a, b), (b, c), (c, a)):
            edge[(min(e), max(e))].append(f)
    adj = defaultdict(list)
    for fs in edge.values():
        if len(fs) == 2:
            adj[fs[0]].append(fs[1])
            adj[fs[1]].append(fs[0])
    return adj


def grow_regions(T, N, area, adj, angle_deg):
    """Flood-fill faces while they agree with the running mean normal."""
    thr = np.cos(np.radians(angle_deg))
    lab = -np.ones(len(T), int)
    cur = 0
    for seed in np.argsort(-area):
        if lab[seed] >= 0:
            continue
        lab[seed] = cur
        acc = N[seed].copy()
        stack = [seed]
        while stack:
            f = stack.pop()
            ref = acc / np.linalg.norm(acc)
            for g in adj[f]:
                if lab[g] < 0 and N[g] @ ref > thr:
                    lab[g] = cur
                    acc += N[g]
                    stack.append(g)
        cur += 1
    return lab, cur


def plane_of(pts, n_hint):
    c = pts.mean(0)
    # least-squares plane: smallest singular vector is the normal
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    if n @ n_hint < 0:
        n = -n
    return n, float(-n @ c)   # n . x + d = 0


def merge_coplanar(groups, angle_deg, dist_tol):
    """Union-find over groups that describe the same physical plane."""
    keys = list(groups)
    parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    thr = np.cos(np.radians(angle_deg))
    for i, a in enumerate(keys):
        na, da, ca = groups[a]["n"], groups[a]["d"], groups[a]["centroid"]
        for b in keys[i + 1:]:
            nb, db, cb = groups[b]["n"], groups[b]["d"], groups[b]["centroid"]
            if na @ nb < thr:
                continue
            # same orientation: also require each centroid to lie on the other plane
            if abs(na @ cb + da) < dist_tol and abs(nb @ ca + db) < dist_tol:
                parent[find(a)] = find(b)
    out = defaultdict(list)
    for k in keys:
        out[find(k)].append(k)
    return out


def min_area_rect(pts2d):
    """Rotating calipers on the convex hull -> (centre, axes, size, angle).

    Meccano plates are rectangular, so regularising the outline to a rectangle is
    a deliberate prior, not a shortcut: it removes scan noise from the silhouette
    and yields a profile CAD can extrude directly.
    """
    from scipy.spatial import ConvexHull
    h = pts2d[ConvexHull(pts2d).vertices]
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        L = np.linalg.norm(e)
        if L < 1e-12:
            continue
        u = e / L
        v = np.array([-u[1], u[0]])
        pu, pv = h @ u, h @ v
        w, ht = pu.ptp(), pv.ptp()
        if best is None or w * ht < best[0]:
            cu, cv = (pu.min() + pu.max()) / 2, (pv.min() + pv.max()) / 2
            best = (w * ht, u, v, w, ht, cu * u + cv * v)
    _, u, v, w, ht, ctr = best
    return ctr, u, v, float(w), float(ht)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True, help="parts JSON")
    ap.add_argument("--color-out", default=None, help="write a colour-coded mesh to eyeball")
    ap.add_argument("--grow-angle", type=float, default=15.0)
    ap.add_argument("--merge-angle", type=float, default=12.0)
    ap.add_argument("--merge-dist", type=float, default=0.02)
    ap.add_argument("--min-area-frac", type=float, default=0.004,
                    help="drop surface groups below this fraction of total area")
    ap.add_argument("--max-thickness", type=float, default=0.12,
                    help="plane pairs further apart than this are not one plate")
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(args.mesh)
    m.remove_duplicated_vertices()
    m.compute_triangle_normals()
    V = np.asarray(m.vertices)
    T = np.asarray(m.triangles)
    N = np.asarray(m.triangle_normals)
    area = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1) / 2
    total = area.sum()
    print(f"mesh {len(T):,} tris, area {total:.3f}")

    adj = face_adjacency(T)
    lab, n_reg = grow_regions(T, N, area, adj, args.grow_angle)
    print(f"region growing @{args.grow_angle}deg -> {n_reg:,} patches")

    groups = {}
    for i in range(n_reg):
        sel = lab == i
        a = area[sel].sum()
        if a < args.min_area_frac * total:
            continue
        pts = V[T[sel]].reshape(-1, 3)
        n, d = plane_of(pts, N[sel].mean(0))
        groups[i] = dict(n=n, d=d, centroid=pts.mean(0), area=float(a), faces=np.nonzero(sel)[0])
    print(f"patches above {100*args.min_area_frac:.1f}% area: {len(groups)}")

    merged = merge_coplanar(groups, args.merge_angle, args.merge_dist)
    surfaces = []
    for root, members in merged.items():
        faces = np.concatenate([groups[k]["faces"] for k in members])
        a = float(area[faces].sum())
        pts = V[T[faces]].reshape(-1, 3)
        n, d = plane_of(pts, np.vstack([groups[k]["n"] for k in members]).mean(0))
        surfaces.append(dict(n=n, d=d, area=a, faces=faces, centroid=pts.mean(0)))
    surfaces.sort(key=lambda s: -s["area"])
    print(f"after coplanar merge: {len(surfaces)} surfaces "
          f"(top covers {100*surfaces[0]['area']/total:.1f}% of area)")
    for i, s in enumerate(surfaces[:10]):
        print(f"  S{i:<2} area {s['area']:.3f} ({100*s['area']/total:4.1f}%) n={np.round(s['n'],2)}")

    # --- pair antiparallel surfaces into plates -------------------------------
    used, plates = set(), []
    for i, a in enumerate(surfaces):
        if i in used:
            continue
        best = None
        for j, b in enumerate(surfaces):
            if j <= i or j in used:
                continue
            if a["n"] @ b["n"] > -np.cos(np.radians(args.merge_angle * 2)):
                continue                      # not antiparallel
            t = abs(a["n"] @ b["centroid"] + a["d"])
            if t > args.max_thickness:
                continue
            # footprints must overlap when projected along the shared normal
            u = np.array([1.0, 0, 0])
            u = u - a["n"] * (u @ a["n"])
            if np.linalg.norm(u) < 1e-6:
                u = np.array([0, 1.0, 0]) - a["n"] * (np.array([0, 1.0, 0]) @ a["n"])
            u /= np.linalg.norm(u)
            v = np.cross(a["n"], u)
            pa = V[T[a["faces"]]].reshape(-1, 3) @ np.c_[u, v]
            pb = V[T[b["faces"]]].reshape(-1, 3) @ np.c_[u, v]
            ov = (min(pa[:, 0].max(), pb[:, 0].max()) - max(pa[:, 0].min(), pb[:, 0].min())) * \
                 (min(pa[:, 1].max(), pb[:, 1].max()) - max(pa[:, 1].min(), pb[:, 1].min()))
            if ov <= 0:
                continue
            score = ov / (t + 1e-6)
            if best is None or score > best[0]:
                best = (score, j, t, u, v)
        if best is None:
            continue
        _, j, t, u, v = best
        used.update({i, j})
        pts = np.vstack([V[T[a["faces"]]].reshape(-1, 3), V[T[surfaces[j]["faces"]]].reshape(-1, 3)])
        p2 = pts @ np.c_[u, v]
        ctr2, e1, e2, w, h = min_area_rect(p2)
        centre = ctr2[0] * u + ctr2[1] * v - a["n"] * ((a["n"] @ pts.mean(0)) * 0 + a["d"]) \
            + a["n"] * (a["n"] @ pts.mean(0) + a["d"]) * 0
        centre = pts.mean(0)
        plates.append(dict(
            type="plate",
            normal=a["n"].tolist(),
            thickness=float(t),
            length=float(w), width=float(h),
            centre=centre.tolist(),
            axis_u=(e1[0] * u + e1[1] * v).tolist(),
            axis_v=(e2[0] * u + e2[1] * v).tolist(),
            area=float(a["area"] + surfaces[j]["area"]),
            faces=np.concatenate([a["faces"], surfaces[j]["faces"]]).tolist()))

    plates.sort(key=lambda p: -p["area"])
    print(f"\nplate candidates (antiparallel surface pairs): {len(plates)}")
    for i, p in enumerate(plates):
        print(f"  P{i}: {p['length']:.3f} x {p['width']:.3f} x t={p['thickness']:.4f} "
              f"n={np.round(p['normal'],2)} area={p['area']:.3f}")

    assigned = set()
    for p in plates:
        assigned.update(p["faces"])
    leftover = np.setdiff1d(np.arange(len(T)), np.fromiter(assigned, int, len(assigned)))
    print(f"\nfaces not in any plate: {len(leftover):,} / {len(T):,} "
          f"({100*len(leftover)/len(T):.0f}%) -> candidates for cylinder/box fitting")

    with open(args.out, "w") as f:
        json.dump(dict(mesh=os.path.abspath(args.mesh), total_area=float(total),
                       plates=[{k: v for k, v in p.items() if k != "faces"} for p in plates],
                       n_surfaces=len(surfaces)), f, indent=2)
    print(f"wrote {args.out}")

    if args.color_out:
        rng = np.random.default_rng(0)
        col = np.tile(np.array([0.75, 0.75, 0.75]), (len(V), 1))
        for p in plates:
            c = rng.random(3) * 0.7 + 0.2
            col[T[np.array(p["faces"])].reshape(-1)] = c
        m.vertex_colors = o3d.utility.Vector3dVector(col)
        o3d.io.write_triangle_mesh(args.color_out, m)
        print(f"wrote {args.color_out}")


if __name__ == "__main__":
    main()
