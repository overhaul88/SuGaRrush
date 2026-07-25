"""Isolate an object from a reconstructed scene mesh using multi-view masks.

Why this exists
---------------
Cropping a scene mesh with a hand-picked axis-aligned box does not work. On
object1 a box tuned by eye against a couple of renders silently discarded ~41%
of the object's neighbourhood -- including the top of the perforated mast --
because a box cannot express "the object" for anything that leans, spreads, or
is taller than it first appears.

The images already contain the answer. Segment the object in every frame, then
keep only the mesh triangles that project inside those masks across the views
that see them. This is space carving / the visual hull: a surface point of the
object projects into the object silhouette from *every* camera, while floor and
background project outside it from most. No geometric prior, no tuned box, and
nothing is eroded -- triangles are kept or dropped whole, so plate rims and
perforations survive exactly as reconstructed.

    python scripts/carve_mesh.py --mesh refined.obj --scene $SCENE \
        --out carved.ply --keep-ratio 0.85

Masks are produced separately (see scripts/gen_masks.py) because the
segmentation model lives in its own conda env.
"""

import argparse
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
            raise SystemExit(f"Camera model {cam.model} is not undistorted; "
                             "run convert.py's image_undistorter first")
        cams.append(dict(name=im.name, R=qvec2rotmat(im.qvec), t=np.asarray(im.tvec),
                         fx=fx, fy=fy, cx=cx, cy=cy, w=cam.width, h=cam.height))
    return cams


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--scene", required=True, help="COLMAP scene dir (has sparse/0 and masks/)")
    ap.add_argument("--masks", default=None, help="mask dir (default <scene>/masks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-ratio", type=float, default=0.85,
                    help="fraction of observing views that must see the vertex "
                         "inside the mask")
    ap.add_argument("--min-views", type=int, default=8,
                    help="vertices seen by fewer cameras than this are dropped")
    ap.add_argument("--tri-rule", choices=["all", "any", "majority"], default="majority",
                    help="how many of a triangle's vertices must survive")
    ap.add_argument("--dilate", type=int, default=2,
                    help="dilate masks by N px to tolerate slight pose error")
    args = ap.parse_args()

    masks_dir = args.masks or os.path.join(args.scene, "masks")
    mesh = o3d.io.read_triangle_mesh(args.mesh, enable_post_processing=True)
    if len(mesh.triangles) == 0:
        raise SystemExit("empty mesh")
    n0 = len(mesh.vertices)
    mesh.remove_duplicated_vertices()
    print(f"mesh: {len(mesh.triangles):,} tris, {len(mesh.vertices):,} verts "
          f"(welded {n0-len(mesh.vertices):,})")

    cams = load_cameras(args.scene)
    print(f"cameras: {len(cams)}")

    V = np.asarray(mesh.vertices)
    seen = np.zeros(len(V), np.int32)
    inside = np.zeros(len(V), np.int32)

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

        P = V @ c["R"].T + c["t"]
        z = P[:, 2]
        front = z > 1e-6
        u = np.full(len(V), -1.0); v = np.full(len(V), -1.0)
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
    print(f"used {used} masked views")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(seen > 0, inside / np.maximum(seen, 1), 0.0)
    keep_v = (seen >= args.min_views) & (ratio >= args.keep_ratio)
    print(f"vertices kept: {keep_v.sum():,} / {len(V):,} "
          f"({100*keep_v.mean():.1f}%)  [ratio>={args.keep_ratio}, views>={args.min_views}]")

    T = np.asarray(mesh.triangles)
    k = keep_v[T].sum(axis=1)
    if args.tri_rule == "all":
        keep_t = k == 3
    elif args.tri_rule == "any":
        keep_t = k >= 1
    else:
        keep_t = k >= 2
    print(f"triangles kept: {keep_t.sum():,} / {len(T):,} ({100*keep_t.mean():.1f}%)")
    if keep_t.sum() == 0:
        raise SystemExit("carve removed everything; lower --keep-ratio")

    mesh.remove_triangles_by_mask(~keep_t)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not o3d.io.write_triangle_mesh(args.out, mesh):
        raise SystemExit(f"failed to write {args.out}")
    print(f"wrote {args.out}: {len(mesh.triangles):,} tris, {len(mesh.vertices):,} verts")


if __name__ == "__main__":
    main()
