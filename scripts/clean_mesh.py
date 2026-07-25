"""Post-process a SuGaR mesh down to a single clean object.

The SuGaR mesh is already Poisson-smooth, but for a single object you usually
still want to crop away the room, drop disconnected floaters, remove speckle,
and optionally decimate and lightly smooth.

    python scripts/clean_mesh.py \
      --in  output/refined_mesh/<scene>/<mesh>.obj \
      --out output/final/myobject_clean.ply \
      --keep-largest --outlier-nb 20 --outlier-std 2.0 \
      --decimate 300000 --smooth-iters 3

Operations run in a fixed order: crop -> cluster filter -> outlier removal ->
degenerate cleanup -> decimate -> smooth. Each stage prints its triangle count
so you can see which one removed what.
"""

import argparse
import os

import numpy as np
import open3d as o3d


def parse_xyz(s, flag):
    parts = s.replace("(", "").replace(")", "").split(",")
    if len(parts) != 3:
        raise SystemExit(f"{flag} expects 'x,y,z', got {s!r}")
    return np.array([float(p) for p in parts], dtype=float)


def report(mesh, label):
    print(f"  {label:<22} {len(mesh.triangles):>9,} tris  {len(mesh.vertices):>9,} verts")


def main():
    ap = argparse.ArgumentParser(description="Crop, denoise, decimate and smooth a mesh.")
    ap.add_argument("--in", dest="inp", required=True, help="input mesh (.obj/.ply)")
    ap.add_argument("--out", dest="out", required=True, help="output mesh (.ply/.obj)")
    ap.add_argument("--crop-min", default=None, help="axis-aligned crop min 'x,y,z'")
    ap.add_argument("--crop-max", default=None, help="axis-aligned crop max 'x,y,z'")
    ap.add_argument("--keep-largest", action="store_true", help="keep only the largest connected component")
    ap.add_argument("--min-cluster", type=int, default=0, help="drop connected components smaller than N triangles")
    ap.add_argument("--outlier-nb", type=int, default=0, help="statistical outlier removal: neighbour count (0 = off)")
    ap.add_argument("--outlier-std", type=float, default=2.0, help="statistical outlier removal: std-dev ratio")
    ap.add_argument("--decimate", type=int, default=0, help="target triangle count (0 = keep full resolution)")
    ap.add_argument("--smooth-iters", type=int, default=0, help="Taubin smoothing iterations (shape-preserving)")
    args = ap.parse_args()

    if not os.path.isfile(args.inp):
        raise SystemExit(f"Input mesh not found: {args.inp}")

    mesh = o3d.io.read_triangle_mesh(args.inp, enable_post_processing=True)
    if len(mesh.triangles) == 0:
        raise SystemExit(f"Loaded no triangles from {args.inp}")
    report(mesh, "loaded")

    # A UV-textured .obj stores one vertex per face-corner, so every triangle looks
    # like its own connected component. Merge coincident vertices before anything
    # topological runs, or --keep-largest keeps exactly one triangle.
    n_before = len(mesh.vertices)
    mesh.remove_duplicated_vertices()
    if len(mesh.vertices) < n_before:
        print(f"  merged {n_before - len(mesh.vertices):,} duplicated vertices "
              f"(textured .obj splits vertices per face)")
        report(mesh, "after vertex merge")

    if (args.crop_min is None) != (args.crop_max is None):
        raise SystemExit("--crop-min and --crop-max must be given together")
    if args.crop_min is not None:
        lo = parse_xyz(args.crop_min, "--crop-min")
        hi = parse_xyz(args.crop_max, "--crop-max")
        if np.any(lo >= hi):
            raise SystemExit(f"--crop-min must be strictly less than --crop-max (got {lo} / {hi})")
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=lo, max_bound=hi))
        report(mesh, "after crop")
        if len(mesh.triangles) == 0:
            raise SystemExit("Crop removed every triangle; widen the box.")

    if args.keep_largest or args.min_cluster > 0:
        idx, counts, _ = mesh.cluster_connected_triangles()
        idx = np.asarray(idx)
        counts = np.asarray(counts)
        if args.keep_largest:
            # Floaters are separate components; the object is the biggest one.
            remove = idx != int(counts.argmax())
        else:
            remove = counts[idx] < args.min_cluster
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
        label = "after keep-largest" if args.keep_largest else "after min-cluster"
        report(mesh, label)
        if len(mesh.triangles) == 0:
            raise SystemExit("Cluster filter removed every triangle; lower --min-cluster.")

    if args.outlier_nb > 0:
        # Vertex-level speckle removal: score vertices as a point cloud, then
        # drop the ones flagged as statistical outliers.
        pcd = o3d.geometry.PointCloud(mesh.vertices)
        _, keep = pcd.remove_statistical_outlier(nb_neighbors=args.outlier_nb,
                                                 std_ratio=args.outlier_std)
        mask = np.ones(len(mesh.vertices), dtype=bool)
        mask[np.asarray(keep)] = False
        mesh.remove_vertices_by_mask(mask)
        report(mesh, "after outlier removal")
        if len(mesh.triangles) == 0:
            raise SystemExit("Outlier removal emptied the mesh; raise --outlier-std.")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    report(mesh, "after cleanup")

    if args.decimate > 0 and len(mesh.triangles) > args.decimate:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.decimate)
        report(mesh, "after decimate")

    if args.smooth_iters > 0:
        # Taubin alternates shrink/inflate passes, so it smooths without the
        # volume loss plain Laplacian smoothing causes.
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iters)
        report(mesh, "after smoothing")

    mesh.compute_vertex_normals()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    if not o3d.io.write_triangle_mesh(args.out, mesh):
        raise SystemExit(f"Failed to write {args.out}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
