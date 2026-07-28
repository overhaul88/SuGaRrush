"""Turn a watertight object shell into a physics collision asset + URDF.

Path B of the pipeline. A rigid-body engine cannot collide against an
arbitrary concave triangle soup, so we bound the *existing* watertight shell
into a small set of convex hulls with CoACD (Collision-Aware Convex
Decomposition). No geometry is invented -- CoACD strictly wraps the shell; the
union of hulls contains the mesh and stays close to it. We then emit inertia
from the same watertight solid and a URDF that references the hulls for
collision and the textured mesh for visuals.

    python scripts/make_collision.py \
        --mesh output/final/object4_final.ply \
        --out-dir output/final/object4/collision --name object4

CoACD's preprocess_mode=auto manifold-repairs a non-watertight input, so an
old scan mesh is a valid smoke test even though the real pipeline feeds a
watertight <name>_final.ply.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def load_mesh(path):
    """Load as a single Trimesh, no processing (keep verts/faces as authored)."""
    m = trimesh.load(path, process=False, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"{path} did not load as a single mesh (got {type(m)})")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def decompose(V, F, threshold, resolution, preprocess_mode):
    """Run CoACD; returns a list of (verts, faces) convex hulls."""
    import coacd
    try:
        coacd.set_log_level("error")
    except Exception:
        pass
    cmesh = coacd.Mesh(V, F)
    parts = coacd.run_coacd(
        cmesh,
        threshold=threshold,
        preprocess_mode=preprocess_mode,
        resolution=resolution,
    )
    # coacd 1.0.11 returns a list of [verts (n,3) float64, faces (m,3) int32].
    return [(np.asarray(pv, dtype=np.float64), np.asarray(pf, dtype=np.int64))
            for pv, pf in parts]


def write_combined_obj(path, parts):
    """One OBJ, each hull its own `o part_i` group with offset vertex indices."""
    with open(path, "w") as fh:
        fh.write("# collision hulls (CoACD) -- one group per convex part\n")
        voff = 0
        for i, (pv, pf) in enumerate(parts):
            fh.write(f"o part_{i}\n")
            for v in pv:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for f in pf:
                fh.write(f"f {f[0]+1+voff} {f[1]+1+voff} {f[2]+1+voff}\n")
            voff += len(pv)


def inertia_props(V, F, scale, density, mass_override):
    """Mass/COM/inertia from the watertight solid (scaled to metres).

    trimesh's mass properties are only trustworthy on a closed manifold. If the
    input is not watertight we fall back to the convex hull of the whole mesh
    (a valid solid) and flag it, rather than emit silently bogus numbers.
    """
    tm = trimesh.Trimesh(V * scale, F, process=False)
    used = "watertight"
    if not tm.is_watertight:
        print("WARN: mesh is not watertight; using its convex hull for inertia",
              flush=True)
        tm = tm.convex_hull
        used = "hull_fallback"
    if mass_override is not None:
        vol = float(tm.volume)
        if vol <= 0:
            raise SystemExit(f"non-positive volume ({vol}); cannot honour --mass")
        tm.density = mass_override / vol
    else:
        tm.density = density
    mass = float(tm.mass)
    com = np.asarray(tm.center_mass, dtype=np.float64)
    I = np.asarray(tm.moment_inertia, dtype=np.float64)  # 3x3 about COM
    return mass, com, I, used


def build_urdf(name, com, I, mass, n_parts, part_basename, visual_filename):
    robot = ET.Element("robot", name=name)
    link = ET.SubElement(robot, "link", name="base_link")

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin",
                  xyz=f"{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(inertial, "inertia",
                  ixx=f"{I[0, 0]:.9g}", iyy=f"{I[1, 1]:.9g}", izz=f"{I[2, 2]:.9g}",
                  ixy=f"{I[0, 1]:.9g}", ixz=f"{I[0, 2]:.9g}", iyz=f"{I[1, 2]:.9g}")

    for i in range(n_parts):
        col = ET.SubElement(link, "collision")
        geom = ET.SubElement(col, "geometry")
        # basename only: parts live beside the urdf, keeps the asset relocatable.
        ET.SubElement(geom, "mesh", filename=part_basename(i))

    vis = ET.SubElement(link, "visual")
    vgeom = ET.SubElement(vis, "geometry")
    ET.SubElement(vgeom, "mesh", filename=visual_filename)

    tree = ET.ElementTree(robot)
    try:
        ET.indent(tree, space="  ")  # py3.9+
    except AttributeError:
        pass
    return tree


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="watertight <name>_final.ply")
    ap.add_argument("--out-dir", required=True,
                    help="e.g. output/final/<name>/collision")
    ap.add_argument("--name", required=True, help="e.g. object4")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD concavity threshold")
    ap.add_argument("--resolution", type=int, default=2000,
                    help="CoACD voxel resolution")
    ap.add_argument("--preprocess", choices=["auto", "on", "off"], default="auto",
                    help="CoACD preprocess_mode")
    ap.add_argument("--visual", default=None,
                    help="relative path to textured visual mesh for the URDF "
                         "<visual> (default ../visual/<name>_textured_obj/<name>.obj)")
    ap.add_argument("--density", type=float, default=1000.0,
                    help="kg/m^3, used for inertia if --mass not given")
    ap.add_argument("--mass", type=float, default=None,
                    help="kg, overrides density-derived mass")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply mesh units -> metres before inertia")
    args = ap.parse_args()

    visual = args.visual or f"../visual/{args.name}_textured_obj/{args.name}.obj"
    os.makedirs(args.out_dir, exist_ok=True)

    V, F = load_mesh(args.mesh)
    print(f"loaded {args.mesh}: {len(V):,} verts, {len(F):,} faces", flush=True)

    parts = decompose(V, F, args.threshold, args.resolution, args.preprocess)
    if len(parts) == 0:
        raise SystemExit("CoACD returned no convex parts")
    print(f"CoACD -> {len(parts)} convex part(s)", flush=True)

    # Per-part OBJs.
    part_files = []
    part_stats = []
    for i, (pv, pf) in enumerate(parts):
        pfn = f"{args.name}_collision_part{i}.obj"
        trimesh.Trimesh(pv, pf, process=False).export(
            os.path.join(args.out_dir, pfn))
        part_files.append(pfn)
        part_stats.append({"part": i, "vertices": int(len(pv)),
                           "faces": int(len(pf))})

    # Combined OBJ (grouped).
    combined = os.path.join(args.out_dir, f"{args.name}_collision.obj")
    write_combined_obj(combined, parts)

    # Inertia from the watertight original (scaled).
    mass, com, I, inertia_src = inertia_props(V, F, args.scale, args.density,
                                              args.mass)

    # URDF.
    tree = build_urdf(args.name, com, I, mass, len(parts),
                      lambda i: part_files[i], visual)
    urdf_path = os.path.join(args.out_dir, f"{args.name}.urdf")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    # JSON summary.
    summary = {
        "name": args.name,
        "mesh": args.mesh,
        "parts": len(parts),
        "part_stats": part_stats,
        "mass": mass,
        "com": com.tolist(),
        "inertia": I.tolist(),
        "inertia_source": inertia_src,
        "threshold": args.threshold,
        "resolution": args.resolution,
        "preprocess": args.preprocess,
        "scale": args.scale,
        "density": args.density,
        "mass_override": args.mass,
        "visual": visual,
    }
    json_path = os.path.join(args.out_dir, f"{args.name}_collision.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"wrote {len(part_files)} part OBJ(s), {os.path.basename(combined)}, "
          f"{os.path.basename(urdf_path)}, {os.path.basename(json_path)} "
          f"-> {args.out_dir}", flush=True)
    print(f"{len(parts)} convex parts, mass={mass:.6g} kg, "
          f"com=({com[0]:.6g}, {com[1]:.6g}, {com[2]:.6g})", flush=True)


if __name__ == "__main__":
    main()
