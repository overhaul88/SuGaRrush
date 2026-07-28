"""Watertight safeguard: turn the isolated object surface into a closed manifold.

SuGaR meshes are open surfaces (the unobserved bottom of an object resting on a
surface is never reconstructed), and `clean_mesh_v2.py`'s `remove_non_manifold_edges`
can open further boundary loops. This step makes the shell a guaranteed watertight,
manifold solid so it can carry vertex colours, bake a texture, and -- above all --
feed CoACD for a valid collision asset with well-defined volume and inertia.

It uses **pymeshfix** (Attene's MeshFix): it removes non-manifold/degenerate geometry,
keeps the largest component, and fills every remaining hole -- including the sheared
bottom -- with a minimal-area triangulated patch. This is the same mathematically
grounded "drum-skin" closure Poisson would apply to the unobserved boundary; no
generative model and no invented detail, just the minimal surface across the gap.
Winding is then normalised so normals face consistently outward.

    python scripts/close_mesh.py --in clean.ply --out final.ply \
        --report-json final_watertight.json
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d
import trimesh
import pymeshfix


def boundary_edge_count(mesh):
    """Edges used by exactly one triangle -- i.e. the open borders of the shell."""
    T = np.asarray(mesh.triangles)
    if len(T) == 0:
        return 0
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum())


def enclosed_volume(mesh):
    """Divergence-theorem volume. Valid on any closed orientable mesh, unlike o3d's get_volume(),
    which refuses anything it calls non-watertight (including merely self-intersecting shells).
    This is the number that exposes a carved-out result: a shell encloses far less than the solid."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if len(F) == 0:
        return 0.0
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(abs(np.einsum("ij,ij->i", np.cross(a, b), c).sum()) / 6.0)


SELFX_FACE_LIMIT = 160_000


def stats(mesh, selfx_limit=SELFX_FACE_LIMIT):
    ext = (np.asarray(mesh.vertices).max(0) - np.asarray(mesh.vertices).min(0)
           if len(mesh.vertices) else np.zeros(3))
    # o3d's is_watertight() runs is_self_intersecting(), which is O(F^2) and stops being usable
    # somewhere above ~150k triangles -- it ran for over ten minutes on a 200k-face mesh. Above the
    # limit, report closedness from the combinatorial invariants (manifold + no boundary), which is
    # what the downstream collision asset actually needs, and say that self-intersection was skipped.
    big = len(mesh.triangles) > selfx_limit
    em = bool(mesh.is_edge_manifold(allow_boundary_edges=False))
    vm = bool(mesh.is_vertex_manifold())
    nb = boundary_edge_count(mesh)
    wt = (em and vm and nb == 0) if big else bool(mesh.is_watertight())
    return {
        "is_watertight": wt,
        "selfx_checked": not big,
        "is_edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "n_boundary_edges": boundary_edge_count(mesh),
        "euler_poincare": int(mesh.euler_poincare_characteristic()),
        "n_verts": len(mesh.vertices),
        "n_tris": len(mesh.triangles),
        "extent": [round(float(x), 4) for x in ext],
        "enclosed_volume": round(enclosed_volume(mesh), 6),
    }


def print_stats(label, s):
    print(f"  {label}: watertight={s['is_watertight']} "
          f"edge_manifold={s['is_edge_manifold']} "
          f"vertex_manifold={s['is_vertex_manifold']} "
          f"boundary_edges={s['n_boundary_edges']:,} "
          f"euler={s['euler_poincare']} "
          f"verts={s['n_verts']:,} tris={s['n_tris']:,} "
          f"extent={s['extent']} volume={s['enclosed_volume']:.4f}"
          + ("" if s.get("selfx_checked", True) else "  [self-intersection test skipped: too many faces]"))


def restore_colors(mesh, orig_verts, orig_cols):
    """Re-attach the input's vertex colors, remapping by nearest (verts change)."""
    if len(mesh.vertices) == 0:
        return
    if len(mesh.vertices) == len(orig_verts):
        mesh.vertex_colors = o3d.utility.Vector3dVector(orig_cols)
        return
    from scipy.spatial import cKDTree
    _, idx = cKDTree(orig_verts).query(np.asarray(mesh.vertices), k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(orig_cols[idx])


def _boundary_loops(F):
    """Trace boundary edges into ordered vertex rings."""
    from collections import defaultdict
    ec = defaultdict(int)
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ec[(a, b) if a < b else (b, a)] += 1
    bnd = {e for e, c in ec.items() if c == 1}
    adj = defaultdict(set)
    for a, b in bnd:
        adj[a].add(b); adj[b].add(a)
    unused, loops = set(bnd), []
    while unused:
        a0, b0 = next(iter(unused))
        unused.discard((a0, b0))
        ring, prev, cur = [a0, b0], a0, b0
        while True:
            nxt = None
            for cand in adj[cur]:
                if cand == prev:
                    continue
                key = (cur, cand) if cur < cand else (cand, cur)
                if key in unused:
                    nxt = cand; unused.discard(key); break
            if nxt is None or nxt == a0:
                break
            ring.append(nxt)
            prev, cur = cur, nxt
        if len(ring) >= 3:
            loops.append(ring)
    return loops


def planar_cap(mesh, min_loop_edges=40, max_planarity=0.02):
    """Close large boundary loops with a FLAT cap instead of a folded triangulation.

    Sec Q4: for a piecewise-planar object the TV-of-the-normal minimiser over the unobserved face is
    a flat face. pymeshfix triangulates a large, non-planar rim into a folded, pinched surface, and
    a local normal flow cannot descend out of a fold -- it is a geometric fold, not roughness. So
    construct the minimiser directly rather than iterating toward it.

    Observed vertices never move (Sec 7.4). We fit a plane to the rim, add a ring of NEW vertices at
    the rim's projection onto that plane, connect rim to projection with a short skirt, and fan the
    planar polygon from its own centroid -- which is star-shaped in-plane for a convex-ish rim, so
    the cap cannot self-intersect. Skirt height equals the rim's own deviation from planarity, so a
    flat rim gives a flat face and a ragged rim degrades gracefully.
    """
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()
    loops = _boundary_loops(F)
    big = [l for l in loops if len(l) >= min_loop_edges]
    if not big:
        return mesh, 0
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    new_v, new_f = [], []
    n_skipped = 0
    for ring in big:
        P = V[ring]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[-1] / np.linalg.norm(vt[-1])
        # A plane is only a model of this rim if the rim is PLANAR. On object4 the wound rim snaked
        # over 52% of the object diagonal (planarity deviation mean 0.138, max 0.466), and the
        # least-squares plane through its centroid passed straight through the solid: the "cap"
        # became a 327-triangle plate carrying 23.7% of the surface area, slicing a corner off the
        # cube. Refuse the model when its own residual says it does not fit, and leave the rim to
        # the general closure instead of inventing a plate.
        dev0 = np.abs((P - c) @ n)
        if dev0.mean() > max_planarity * diag:
            print(f"  planar cap: SKIPPED a rim of {len(ring)} edges -- planarity deviation "
                  f"mean {dev0.mean():.4f} > {max_planarity:g} x diagonal ({max_planarity*diag:.4f}); "
                  f"a plane does not model this rim")
            n_skipped += 1
            continue
        proj = P - np.outer((P - c) @ n, n)                     # rim projected onto the plane
        base = len(V) + len(new_v)
        new_v.extend(proj.tolist())
        apex = len(V) + len(new_v)
        new_v.append(proj.mean(0).tolist())
        k = len(ring)
        for i in range(k):
            a, b = ring[i], ring[(i + 1) % k]
            pa, pb = base + i, base + (i + 1) % k
            new_f.append([a, b, pb])                            # skirt
            new_f.append([a, pb, pa])
            new_f.append([pa, pb, apex])                        # planar fan
        dev = np.abs((P - c) @ n)
        print(f"  planar cap: rim of {k} edges, planarity deviation "
              f"mean {dev.mean():.4f} max {dev.max():.4f}")
    if not new_f:
        return mesh, 0
    V = np.vstack([V, np.array(new_v)])
    F = np.vstack([F, np.array(new_f, dtype=np.int32)])
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                    o3d.utility.Vector3iVector(F))
    out.remove_duplicated_vertices()
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    return out, len(big) - n_skipped


def poisson_refit(mesh, depth, n_samples, min_component_frac):
    """Re-fit the surface with screened Poisson so the result is closed by construction.

    Why this instead of handing the raw mesh to pymeshfix: MeshFix makes a mesh manifold by
    DELETING every configuration it cannot resolve. On a surface with hundreds of sub-millimetre
    tunnels (the --gaussian-prune path: genus 613, 1,726 boundary loops) that means hundreds of
    rounds of cutting, and on object4 it removed 8.5% of the object -- two axes shrank ~20% and the
    enclosed volume fell to 0.086 (14% of the bbox) when the true solid is ~0.28.

    Poisson instead solves for an implicit indicator function and extracts its isosurface, so small
    holes get the drum-skin fill we want for the unobserved bottom, tunnels below the octree
    resolution simply cannot be represented, and no input geometry is cut away. Measured on object4
    (depth 7): surface deletion drops 8.496% -> 0.091% of the object diagonal, and PCA returns to
    [1, 0.97, 0.93] against the pre-repair mesh's [1, 0.99, 0.93].

    Poisson needs consistently oriented normals. For an isolated, roughly star-shaped object,
    orienting every sample outward from the centroid is exact; the flipped fraction is printed so a
    non-star-shaped input is visible rather than silent.
    """
    lbl, cnt, _ = mesh.cluster_connected_triangles()
    lbl, cnt = np.asarray(lbl), np.asarray(cnt)
    keep_ids = {i for i, c in enumerate(cnt) if c >= min_component_frac * cnt.sum()}
    n_before = len(mesh.triangles)
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_triangles_by_mask(np.array([l not in keep_ids for l in lbl]))
    mesh.remove_unreferenced_vertices()
    print(f"  dropped {len(cnt) - len(keep_ids):,} noise components "
          f"({n_before:,} -> {len(mesh.triangles):,} triangles)")

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_samples, use_triangle_normal=False)
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    centre = pts.mean(0)
    flipped = (nrm * (pts - centre)).sum(1) < 0
    nrm[flipped] *= -1
    pcd.normals = o3d.utility.Vector3dVector(nrm)

    fitted, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    lbl, cnt, _ = fitted.cluster_connected_triangles()
    fitted.remove_triangles_by_mask(np.asarray(lbl) != int(np.argmax(np.asarray(cnt))))
    fitted.remove_unreferenced_vertices()
    print(f"  poisson re-fit depth={depth} on {n_samples:,} samples "
          f"({flipped.mean()*100:.0f}% normals flipped outward) -> {len(fitted.triangles):,} triangles")
    return fitted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="cleaned mesh .ply")
    ap.add_argument("--out", dest="out", required=True, help="final watertight .ply")
    ap.add_argument("--keep-components", action="store_true",
                    help="keep all connected components (default: keep only the largest solid)")
    ap.add_argument("--planar-cap", action="store_true",
                    help="close large boundary loops with a flat, plane-fitted cap before "
                         "pymeshfix. Use when the wound is one big face of a piecewise-planar "
                         "object: pymeshfix triangulates a large rim into a folded, pinched "
                         "surface, and no local flow can descend out of a fold.")
    ap.add_argument("--planar-cap-max-dev", type=float, default=0.02,
                    help="refuse to planar-cap a rim whose planarity deviation exceeds this "
                         "fraction of the object diagonal (a plane would not model it)")
    ap.add_argument("--planar-cap-min-edges", type=int, default=40,
                    help="only loops with at least this many boundary edges get a planar cap")
    ap.add_argument("--poisson-repair", action="store_true",
                    help="re-fit the surface with screened Poisson before pymeshfix. Use when the "
                         "input is topologically filthy (many tiny holes/tunnels), as the "
                         "--gaussian-prune path produces: pymeshfix alone resolves that by CUTTING, "
                         "which cost 8.5%% of the object on object4.")
    ap.add_argument("--poisson-depth", type=int, default=7,
                    help="octree depth for --poisson-repair (default 7). Lower = cleaner topology "
                         "and fewer triangles, higher = closer to the input surface.")
    ap.add_argument("--poisson-samples", type=int, default=800_000,
                    help="surface samples fed to the Poisson re-fit (default 800k)")
    ap.add_argument("--min-component-frac", type=float, default=0.001,
                    help="drop connected components below this fraction of all triangles before "
                         "the Poisson re-fit (default 0.001)")
    ap.add_argument("--crop-ply", default=None,
                    help="a Gaussian/point .ply whose bbox (padded) crops the mesh to the object "
                         "region before closing -- removes any residual drifted geometry")
    ap.add_argument("--crop-pad", type=float, default=0.10,
                    help="fractional padding added to the --crop-ply bbox (default 0.10)")
    ap.add_argument("--report-json", default=None,
                    help="optional path for a JSON before/after watertight report")
    # Accepted for backward compatibility; pymeshfix always fully closes the shell.
    ap.add_argument("--max-hole", type=float, default=0.05, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not os.path.isfile(args.inp):
        raise SystemExit(f"Input mesh not found: {args.inp}")

    mesh = o3d.io.read_triangle_mesh(args.inp)
    if len(mesh.triangles) == 0:
        raise SystemExit(f"Loaded no triangles from {args.inp}")

    # Optional: crop to the object region defined by a reference point cloud (the pruned object
    # Gaussians). Loss-masked SuGaR can leave a few redundant Gaussians drifting into the
    # (unpenalised) background; their surface is outside this bbox and is removed here.
    if args.crop_ply:
        from plyfile import PlyData
        v = PlyData.read(args.crop_ply)["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        lo, hi = pts.min(0), pts.max(0)
        pad = args.crop_pad * (hi - lo)
        aabb = o3d.geometry.AxisAlignedBoundingBox(lo - pad, hi + pad)
        n_before = len(mesh.triangles)
        mesh = mesh.crop(aabb)
        mesh.remove_unreferenced_vertices()
        print(f"  cropped to object bbox {np.round(lo - pad, 3)}..{np.round(hi + pad, 3)}: "
              f"{n_before:,} -> {len(mesh.triangles):,} triangles")
        if len(mesh.triangles) == 0:
            raise SystemExit("crop removed all geometry -- check --crop-ply matches the mesh frame")

    has_colors = mesh.has_vertex_colors()
    orig_verts = np.asarray(mesh.vertices).copy()
    orig_cols = np.asarray(mesh.vertex_colors).copy() if has_colors else None

    before = stats(mesh)
    area_before = mesh.get_surface_area()
    print_stats("before", before)

    # ---- optional flat cap over the big unobserved face ----
    if args.planar_cap:
        mesh, n_capped = planar_cap(mesh, args.planar_cap_min_edges, args.planar_cap_max_dev)
        if n_capped:
            print(f"  planar-capped {n_capped} large boundary loop(s)")

    # ---- optional Poisson re-fit so pymeshfix has a clean, low-genus surface to close ----
    if args.poisson_repair:
        mesh = poisson_refit(mesh, args.poisson_depth, args.poisson_samples,
                             args.min_component_frac)

    # ---- pymeshfix: guaranteed watertight manifold (fills the sheared bottom) ----
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    vc, fc = pymeshfix.clean_from_arrays(
        V, F, verbose=False, joincomp=False,
        remove_smallest_components=not args.keep_components)
    if len(fc) == 0:
        raise SystemExit("pymeshfix returned an empty mesh")

    # Normalise winding so normals face consistently outward (CoACD/inertia need this).
    tm = trimesh.Trimesh(vertices=vc, faces=fc, process=False)
    tm.fix_normals()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)))

    if has_colors:
        restore_colors(mesh, orig_verts, orig_cols)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    after = stats(mesh)
    area_after = mesh.get_surface_area()
    filled_area = area_after - area_before
    sealed = max(0, before["n_boundary_edges"] - after["n_boundary_edges"])
    print_stats("after ", after)
    print(f"  sealed {sealed:,} boundary edges, surface area {filled_area:+.4f}")

    # Distinguish the two failure modes. o3d's is_watertight() is edge-manifold AND vertex-manifold
    # AND *not self-intersecting*, so a perfectly closed shell still reports False if any pair of
    # triangles touches. Only an actually-open shell makes volume/inertia meaningless.
    closed = after["n_boundary_edges"] == 0 and after["is_edge_manifold"]
    if not after["is_watertight"]:
        if closed:
            print("  note: closed manifold (0 boundary edges, euler "
                  f"{after['euler_poincare']}) but o3d reports not-watertight, i.e. some triangles "
                  "self-intersect. Enclosed volume is still well defined.")
        else:
            print(f"  WARNING: mesh is not closed ({after['n_boundary_edges']:,} boundary edges). "
                  f"Downstream volume/inertia will be unreliable.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not o3d.io.write_triangle_mesh(args.out, mesh):
        raise SystemExit(f"Failed to write {args.out}")

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump({"input": args.inp, "output": args.out,
                       "method": ("poisson+pymeshfix" if args.poisson_repair else "pymeshfix"),
                       "poisson_depth": args.poisson_depth if args.poisson_repair else None,
                       "filled_area": filled_area, "sealed_boundary_edges": sealed,
                       "before": before, "after": after}, f, indent=2)
        print(f"  wrote report {args.report_json}")

    print(f"watertight: {before['is_watertight']} -> {after['is_watertight']}  |  "
          f"boundary_edges: {before['n_boundary_edges']:,} -> "
          f"{after['n_boundary_edges']:,}  |  wrote {args.out}")


if __name__ == "__main__":
    main()
