"""Open a .ply, picking the right viewer for what it actually contains.

SuGaR emits two very different things with the same extension:

  * a **Gaussian Splatting** .ply (output/refined_ply/...) — per-point opacity,
    anisotropic scales, rotations and spherical-harmonic colour. Opening this in a
    mesh viewer shows a meaningless grey blob, because none of that is geometry.
  * a **triangle mesh** .ply (output/refined_mesh/..., or clean_mesh.py output).

This script sniffs the header and tells you which you have, then renders it.

    python scripts/view_ply.py <file.ply>
    python scripts/view_ply.py <file.ply> --info     # report only, no window
"""

import argparse
import os
import sys


def read_header(path):
    fields, n_vertex, n_face = [], 0, 0
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise SystemExit(f"{path} is not a PLY file")
        section = None
        while True:
            line = f.readline()
            if not line:
                raise SystemExit("Malformed PLY: no end_header")
            s = line.decode("ascii", "replace").strip()
            if s.startswith("element vertex"):
                section, n_vertex = "vertex", int(s.split()[-1])
            elif s.startswith("element face"):
                section, n_face = "face", int(s.split()[-1])
            elif s.startswith("property") and section == "vertex":
                fields.append(s.split()[-1])
            elif s == "end_header":
                break
    return fields, n_vertex, n_face


def main():
    ap = argparse.ArgumentParser(description="View a SuGaR .ply with the right viewer.")
    ap.add_argument("path")
    ap.add_argument("--info", action="store_true", help="report the file type and exit")
    ap.add_argument("--point-size", type=float, default=2.0)
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        raise SystemExit(f"No such file: {args.path}")

    fields, n_vertex, n_face = read_header(args.path)
    # The 3DGS convention: DC spherical-harmonic terms plus per-point opacity.
    is_gaussian = any(f.startswith("f_dc_") for f in fields) and "opacity" in fields

    print(f"{args.path}")
    print(f"  vertices : {n_vertex:,}")
    print(f"  faces    : {n_face:,}")
    if is_gaussian:
        sh = sum(1 for f in fields if f.startswith("f_rest_"))
        print(f"  type     : 3D Gaussian Splatting cloud ({sh} SH coeffs/point)")
        print("  NOTE     : this is NOT a mesh. Use the SuGaR viewer for a correct render:")
        print(f"               python run_viewer.py -p {args.path}")
        print("             Or drag it onto https://superspl.at/editor")
        print("  Falling back to a plain point render (colour/opacity will look wrong).")
    else:
        print(f"  type     : triangle mesh" if n_face else "  type     : point cloud")

    if args.info:
        return
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit("No display available; re-run with --info, or use a GUI session.")

    import open3d as o3d
    if n_face and not is_gaussian:
        geom = o3d.io.read_triangle_mesh(args.path)
        geom.compute_vertex_normals()
    else:
        geom = o3d.io.read_point_cloud(args.path)
        if not geom.has_colors():
            geom.paint_uniform_color([0.7, 0.7, 0.7])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(args.path), width=1280, height=800)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
