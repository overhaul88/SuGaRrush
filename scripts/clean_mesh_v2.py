"""Feature-preserving cleanup for thin-shell / perforated / mechanical objects.

Why a second version
--------------------
`clean_mesh.py` uses the standard point-cloud cleanup recipe (statistical outlier
removal, keep-largest-component, isotropic Taubin smoothing). Those are the right
defaults for a blobby scanned object, and the wrong ones for a Meccano-style model
made of thin perforated plates:

  * Statistical outlier removal assumes locally uniform point density. The rim of a
    thin plate, and the rim of every perforation, has genuinely lower neighbour
    density than the plate interior -- so the filter classifies real geometry as
    noise and erodes inward from every edge. Measured on object1: 3,583 of 54,420
    triangles removed, concentrated exactly on the features we care about.
  * keep-largest assumes the object is one connected shell. Poisson reconstruction
    of thin parts fragments naturally, so wheels and axles arrive as separate
    components and get discarded by construction.
  * Taubin smoothing is isotropic: it cannot tell a noise bump from a 90 degree
    mechanical crease or a perforation rim, and rounds all three.
  * An axis-aligned height cut cannot separate a flat support surface from wheels
    resting on it -- they occupy the same band of world Y.

This version replaces each of those with a topology- or geometry-aware equivalent.
Every stage is off by default and opt-in, so nothing erodes silently.

    python scripts/clean_mesh_v2.py --in refined.obj --out clean.ply \
        --roi-center "0.081,1.738,1.640" --roi-half "0.6,1.1,0.6" \
        --remove-support-plane --min-component-tris 150 \
        --smooth-iters 3 --preserve-angle 35
"""

import argparse
import os

import numpy as np
import open3d as o3d


def report(mesh, label):
    print(f"  {label:<28} {len(mesh.triangles):>8,} tris  {len(mesh.vertices):>8,} verts")


def parse_vec(s, flag):
    parts = s.replace("(", "").replace(")", "").split(",")
    if len(parts) != 3:
        raise SystemExit(f"{flag} expects 'x,y,z', got {s!r}")
    return np.array([float(p) for p in parts], float)


def boundary_vertices(mesh):
    """Vertices on an open edge -- i.e. perforation rims and plate borders.

    An edge shared by exactly one triangle bounds a hole. Those loops ARE the
    features on this object, so they must never be moved by smoothing.
    """
    T = np.asarray(mesh.triangles)
    e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    border = e[cnt[inv] == 1]
    out = np.zeros(len(mesh.vertices), bool)
    if len(border):
        out[np.unique(border)] = True
    return out


def crease_vertices(mesh, angle_deg):
    """Vertices touching an edge whose dihedral angle exceeds the threshold.

    These are the mechanical creases (plate folds, box corners). Freezing them is
    what keeps the model from melting into a blob under smoothing.
    """
    mesh.compute_triangle_normals()
    N = np.asarray(mesh.triangle_normals)
    T = np.asarray(mesh.triangles)
    e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    owner = np.tile(np.arange(len(T)), 3)
    order = np.lexsort((np.max(e, 1), np.min(e, 1)))
    e, owner = e[order], owner[order]
    key = np.sort(e, axis=1)
    same = np.all(key[1:] == key[:-1], axis=1)
    idx = np.nonzero(same)[0]
    out = np.zeros(len(mesh.vertices), bool)
    if len(idx) == 0:
        return out
    a, b = owner[idx], owner[idx + 1]
    cosang = np.clip(np.einsum("ij,ij->i", N[a], N[b]), -1, 1)
    sharp = np.degrees(np.arccos(cosang)) > angle_deg
    if sharp.any():
        out[np.unique(key[idx][sharp])] = True
    return out


def feature_preserving_smooth(mesh, iterations, angle_deg, lam=0.5, mu=-0.53):
    """Taubin smoothing that skips boundary and crease vertices.

    Taubin's alternating shrink/inflate avoids the volume loss of plain Laplacian
    smoothing, but it is still isotropic. Pinning rim and crease vertices makes it
    anisotropic in the only way that matters here: noise on flat spans gets
    averaged away, while holes and hard edges stay exactly where they are.
    """
    V = np.asarray(mesh.vertices).copy()
    T = np.asarray(mesh.triangles)
    frozen = boundary_vertices(mesh) | crease_vertices(mesh, angle_deg)
    print(f"  frozen {frozen.sum():,} of {len(V):,} verts "
          f"({100*frozen.mean():.1f}% rim/crease) -- these are not smoothed")

    e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(src, minlength=len(V)).astype(float)
    deg[deg == 0] = 1.0
    movable = ~frozen

    for _ in range(iterations):
        for factor in (lam, mu):
            acc = np.zeros_like(V)
            np.add.at(acc, src, V[dst])
            delta = acc / deg[:, None] - V
            V[movable] += factor * delta[movable]

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T))
    return out


def estimate_support_plane(mesh, roi_center, roi_half, expand, dist_thresh):
    """Fit the surface the object rests on, using the area AROUND the object.

    Fitting inside the ROI does not work: once the mat is cropped away the largest
    remaining plane belongs to the object itself (on object1 that was the mast
    plate, and removing it punched the mast full of holes). So sample an annulus --
    inside a widened box, outside the ROI -- which contains mat and no object, and
    fit there.
    """
    V = np.asarray(mesh.vertices)
    wide_lo, wide_hi = roi_center - roi_half * expand, roi_center + roi_half * expand
    in_wide = np.all((V >= wide_lo) & (V <= wide_hi), axis=1)
    in_roi = np.all((V >= roi_center - roi_half) & (V <= roi_center + roi_half), axis=1)
    ring = V[in_wide & ~in_roi]
    if len(ring) < 200:
        print(f"  only {len(ring)} verts around the object; skipping plane estimate")
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ring))
    plane, inl = pcd.segment_plane(distance_threshold=dist_thresh,
                                   ransac_n=3, num_iterations=4000)
    n = np.array(plane[:3], float)
    nrm = np.linalg.norm(n)
    plane = np.array(plane, float) / nrm
    print(f"  support plane from {len(ring):,} surrounding verts: "
          f"normal {np.round(plane[:3],3)} ({len(inl):,} inliers, "
          f"{100*len(inl)/len(ring):.0f}%)")
    # The object stands off its support; if the fitted plane slices through the
    # object centre it is not the support surface and must not be used.
    if abs(float(roi_center @ plane[:3]) + plane[3]) < dist_thresh * 2:
        print("  WARNING: fitted plane passes through the object centre; ignoring it")
        return None
    return plane


def remove_support_plane(mesh, plane, dist_thresh, normal_deg):
    """Delete the support surface without touching what rests on it.

    A height cut cannot do this: mat and wheels share a band of world Y. A plane
    can, because it discriminates on orientation as well as position -- a triangle
    goes only if it lies near the plane AND is parallel to it. Wheel triangles at
    the contact point survive because their normals turn away from the mat.
    """
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    n_hat = plane[:3]
    mesh.compute_triangle_normals()
    TN = np.asarray(mesh.triangle_normals)
    d = np.abs(V[T].mean(axis=1) @ n_hat + plane[3])
    parallel = np.abs(TN @ n_hat) > np.cos(np.radians(normal_deg))
    kill = (d < dist_thresh) & parallel
    print(f"  removing {kill.sum():,} planar triangles "
          f"(within {dist_thresh} of plane AND parallel to it)")
    mesh.remove_triangles_by_mask(kill)
    mesh.remove_unreferenced_vertices()
    return mesh


def filter_components(mesh, min_tris, attach_radius):
    """Keep every component that is either big or close to the main body.

    keep-largest would drop a detached wheel. Size alone would drop a small one.
    The union of the two tests keeps real parts and still discards speckle.
    """
    idx, counts, areas = mesh.cluster_connected_triangles()
    idx = np.asarray(idx)
    counts = np.asarray(counts)
    if len(counts) == 0:
        return mesh
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    main = int(np.argmax(counts))
    main_pts = V[np.unique(T[idx == main])]
    lo, hi = main_pts.min(0), main_pts.max(0)

    keep = np.zeros(len(counts), bool)
    for c in range(len(counts)):
        if counts[c] >= min_tris:
            keep[c] = True
            continue
        pts = V[np.unique(T[idx == c])]
        # distance from this component to the main body's bounding box
        outside = np.maximum(np.maximum(lo - pts, pts - hi), 0.0)
        if np.linalg.norm(outside, axis=1).min() <= attach_radius:
            keep[c] = True
    dropped = int((~keep[idx]).sum())
    print(f"  components: {len(counts):,} -> keeping {int(keep.sum()):,} "
          f"(dropped {dropped:,} tris of speckle)")
    mesh.remove_triangles_by_mask(~keep[idx])
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--roi-center", default=None, help="object centre 'x,y,z'")
    ap.add_argument("--roi-half", default=None, help="half-extent 'x,y,z'")
    ap.add_argument("--remove-support-plane", action="store_true",
                    help="delete the flat surface the object rests on")
    ap.add_argument("--plane-dist", type=float, default=0.02,
                    help="how close to the plane counts as on it")
    ap.add_argument("--plane-normal-deg", type=float, default=25.0,
                    help="max deviation from parallel to count as planar")
    ap.add_argument("--plane-expand", type=float, default=2.5,
                    help="how far beyond the ROI to look for the support surface")
    ap.add_argument("--roi-radius", type=float, default=0.0,
                    help="radial cut about the object's upright axis (0 = off); "
                         "separates an upright object from the surface it stands on")
    ap.add_argument("--roi-axis", default=None,
                    help="explicit upright axis 'x,y,z' if not using plane removal")
    ap.add_argument("--min-component-tris", type=int, default=150)
    ap.add_argument("--attach-radius", type=float, default=0.05,
                    help="small components within this distance of the body are kept")
    ap.add_argument("--smooth-iters", type=int, default=0)
    ap.add_argument("--preserve-angle", type=float, default=35.0,
                    help="dihedral angle above which an edge is a crease and is frozen")
    ap.add_argument("--decimate", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isfile(args.inp):
        raise SystemExit(f"Input mesh not found: {args.inp}")

    mesh = o3d.io.read_triangle_mesh(args.inp, enable_post_processing=True)
    if len(mesh.triangles) == 0:
        raise SystemExit(f"Loaded no triangles from {args.inp}")
    report(mesh, "loaded")

    n0 = len(mesh.vertices)
    mesh.remove_duplicated_vertices()
    if len(mesh.vertices) < n0:
        print(f"  welded {n0-len(mesh.vertices):,} split vertices (textured .obj)")
        report(mesh, "after weld")

    roi_center = parse_vec(args.roi_center, "--roi-center") if args.roi_center else None
    plane = None
    if roi_center is not None:
        if not args.roi_half:
            raise SystemExit("--roi-center requires --roi-half")
        half = parse_vec(args.roi_half, "--roi-half")
        # Fit the support plane BEFORE cropping, while the mat is still present.
        if args.remove_support_plane:
            plane = estimate_support_plane(mesh, roi_center, half,
                                           args.plane_expand, args.plane_dist)
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=roi_center - half, max_bound=roi_center + half))
        report(mesh, "after ROI crop")
        if len(mesh.triangles) == 0:
            raise SystemExit("ROI removed everything; widen --roi-half")
    elif args.remove_support_plane:
        raise SystemExit("--remove-support-plane requires --roi-center/--roi-half")

    if plane is not None:
        mesh = remove_support_plane(mesh, plane, args.plane_dist,
                                    args.plane_normal_deg)
        report(mesh, "after plane removal")

    if args.roi_radius > 0:
        # An upright object occupies a narrow column; the support surface spreads
        # outward from it. Poisson welds the two together, so no component filter
        # or height cut can separate them -- but a radial cut about the object's
        # own vertical axis can, because it discriminates on the axis the two
        # actually differ along.
        if plane is not None:
            axis = plane[:3]
        elif args.roi_axis:
            axis = parse_vec(args.roi_axis, "--roi-axis")
            axis = axis / np.linalg.norm(axis)
        else:
            raise SystemExit("--roi-radius needs --remove-support-plane or --roi-axis")
        V = np.asarray(mesh.vertices)
        T = np.asarray(mesh.triangles)
        rel = V[T].mean(axis=1) - roi_center
        radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
        kill = radial > args.roi_radius
        print(f"  radial cut about axis {np.round(axis,3)}: "
              f"removing {kill.sum():,} tris beyond r={args.roi_radius}")
        mesh.remove_triangles_by_mask(kill)
        mesh.remove_unreferenced_vertices()
        report(mesh, "after radial crop")

    if args.min_component_tris > 0:
        mesh = filter_components(mesh, args.min_component_tris, args.attach_radius)
        report(mesh, "after component filter")

    # Topology hygiene only -- nothing here erodes a surface.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    report(mesh, "after topology cleanup")

    if args.smooth_iters > 0:
        mesh = feature_preserving_smooth(mesh, args.smooth_iters, args.preserve_angle)
        report(mesh, "after smoothing")

    if args.decimate > 0 and len(mesh.triangles) > args.decimate:
        # Quadric decimation already respects creases; boundaries are what it
        # handles worst, so only use this when you actually need a lighter mesh.
        mesh = mesh.simplify_quadric_decimation(args.decimate)
        report(mesh, "after decimation")

    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not o3d.io.write_triangle_mesh(args.out, mesh):
        raise SystemExit(f"Failed to write {args.out}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
