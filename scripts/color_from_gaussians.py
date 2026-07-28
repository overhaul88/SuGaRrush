"""Colour a final mesh from the surviving Gaussians' diffuse (SH DC) colour.

Meshing (Poisson + cleanup + hole closing) can drop, move or re-index vertices, so
the mesh no longer carries the per-vertex colour the extractor gave it. The pruned
Gaussian cloud, however, still holds the diffuse colour of every splat that survived
selection. Mesh and Gaussians live in the same COLMAP world frame, so a world-space
nearest-neighbour lookup transfers colour faithfully: each mesh vertex takes the
diffuse RGB of its closest Gaussian.

The SH DC term is converted with sugar's SH2RGB (sugar_utils/spherical_harmonics.py:
`rgb = f_dc * C0 + 0.5`, C0 = 0.28209479177387814), clamped to [0, 1].

    python scripts/color_from_gaussians.py --mesh final.ply \
        --gs-ply pruned/point_cloud.ply --out final_colored.ply --glb final_colored.glb
"""

import argparse
import os

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

C0 = 0.28209479177387814  # matches sugar_utils.spherical_harmonics.SH2RGB


def read_gaussians(gs_ply):
    """Return (centers Nx3, diffuse RGB Nx3 in [0,1]) from a 3DGS point_cloud.ply."""
    v = PlyData.read(gs_ply)["vertex"].data
    centers = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
    rgb = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)  # SH2RGB
    return centers, rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="final watertight .ply to colour")
    ap.add_argument("--gs-ply", dest="gs_ply", required=True,
                    help="pruned Gaussian point_cloud.ply (with f_dc_0/1/2)")
    ap.add_argument("--out", default=None, help="output .ply (default: colour in place)")
    ap.add_argument("--glb", default=None, help="also write a colored .glb via trimesh")
    args = ap.parse_args()

    if not os.path.isfile(args.mesh):
        raise SystemExit(f"Mesh not found: {args.mesh}")
    if not os.path.isfile(args.gs_ply):
        raise SystemExit(f"Gaussian ply not found: {args.gs_ply}")
    out = args.out or args.mesh

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.vertices) == 0:
        raise SystemExit(f"Loaded no vertices from {args.mesh}")

    centers, colors = read_gaussians(args.gs_ply)
    verts = np.asarray(mesh.vertices)

    # World-space nearest Gaussian per vertex (shared COLMAP frame).
    _, idx = cKDTree(centers).query(verts, k=1)
    vertex_colors = colors[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if not o3d.io.write_triangle_mesh(out, mesh):
        raise SystemExit(f"Failed to write {out}")

    if args.glb:
        import trimesh
        rgba = np.concatenate(
            [(vertex_colors * 255).round().astype(np.uint8),
             np.full((len(vertex_colors), 1), 255, np.uint8)], axis=1)
        tm = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.triangles),
                             vertex_colors=rgba, process=False)
        os.makedirs(os.path.dirname(os.path.abspath(args.glb)), exist_ok=True)
        tm.export(args.glb)
        print(f"  wrote glb {args.glb}")

    print(f"colored {len(verts)} vertices from {len(centers)} gaussians -> {out}")


if __name__ == "__main__":
    main()
