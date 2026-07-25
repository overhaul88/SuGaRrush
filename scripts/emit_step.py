"""Emit a STEP assembly from fitted primitives (parts.json).

Stage 7e. Runs in the `cad` conda env (CadQuery / Open CASCADE), deliberately
separate from `sugar` so the OCCT stack cannot disturb the pinned torch build.
The interface between them is parts.json -- geometry in, B-rep out.

STEP rather than STL/OBJ because it is the only common format that carries
*assembly structure and named solids*, which is the entire point of this stage:
the operator wants editable components, not another triangle soup.

    conda activate cad
    python scripts/emit_step.py --parts parts.json --out assembly.step --scale 120

--scale is mm per scene unit and is REQUIRED. A CAD model with arbitrary units is
worse than no CAD model: it looks authoritative and silently mis-sizes every part.
"""

import argparse
import json

import cadquery as cq


def orthonormal(n, u):
    """Return a right-handed frame from a normal and a rough in-plane direction."""
    import math
    ln = math.sqrt(sum(c * c for c in n))
    n = [c / ln for c in n]
    d = sum(a * b for a, b in zip(u, n))
    u = [a - d * b for a, b in zip(u, n)]
    lu = math.sqrt(sum(c * c for c in u))
    if lu < 1e-9:
        u = [1.0, 0.0, 0.0]
        d = sum(a * b for a, b in zip(u, n))
        u = [a - d * b for a, b in zip(u, n)]
        lu = math.sqrt(sum(c * c for c in u))
    u = [c / lu for c in u]
    return n, u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, required=True,
                    help="mm per scene unit (REQUIRED - see module docstring)")
    ap.add_argument("--min-area-frac", type=float, default=0.0,
                    help="skip parts explaining less than this fraction of area")
    ap.add_argument("--holes", default=None,
                    help="holes.json from extract_holes.py; perforates its plate")
    ap.add_argument("--hole-mode", choices=["measured", "snapped"], default="snapped",
                    help="'measured' uses each detected radius; 'snapped' uses one "
                         "regularised diameter for all holes (real parts are uniform)")
    args = ap.parse_args()

    data = json.load(open(args.parts))
    holes = json.load(open(args.holes)) if args.holes else None
    s = args.scale
    asm = cq.Assembly(name="reconstructed_assembly")
    n_emit = 0

    for p in data["parts"]:
        if p.get("area", 0) < args.min_area_frac * data["total_area"]:
            continue
        if p["type"] == "plate":
            n, u = orthonormal(p["normal"], p["axis_u"])
            v = [n[1] * u[2] - n[2] * u[1],
                 n[2] * u[0] - n[0] * u[2],
                 n[0] * u[1] - n[1] * u[0]]
            L, W, Th = p["length"] * s, p["width"] * s, p["thickness"] * s
            solid = cq.Workplane("XY").box(L, W, Th)
            if holes and holes.get("part") == p["id"]:
                # Real perforations are one diameter on a regular grid; the scan
                # gives a spread. Snapping to the median regularises scan noise
                # the same way the rectangular outline does.
                pts = []
                rs = []
                for h in holes["holes"]:
                    pts.append((h["u"] * s, h["v"] * s))
                    rs.append((holes["hole_radius_scene"] if args.hole_mode == "snapped"
                               else h["r"]) * s)
                cut = 0
                for (hu, hv), hr in zip(pts, rs):
                    if abs(hu) > L / 2 - hr or abs(hv) > W / 2 - hr:
                        continue          # would breach the plate edge
                    try:
                        solid = (solid.faces(">Z").workplane(origin=(hu, hv, 0))
                                 .circle(hr).cutThruAll())
                        cut += 1
                    except Exception:
                        pass
                print(f"  {p['id']}: perforated with {cut}/{len(pts)} holes "
                      f"(d={2*rs[0]:.2f} mm, {args.hole_mode})")
            # place: columns of the rotation matrix are the part's local axes
            plane = cq.Plane(origin=cq.Vector(*[c * s for c in p["centre"]]),
                             xDir=cq.Vector(*u), normal=cq.Vector(*n))
            loc = cq.Location(plane)
            asm.add(solid, name=p["id"], loc=loc)
            n_emit += 1
        elif p["type"] == "cylinder":
            R, H = p["radius"] * s, p["length"] * s
            solid = cq.Workplane("XY").circle(R).extrude(H).translate((0, 0, -H / 2))
            n, u = orthonormal(p["axis"], [1.0, 0.0, 0.0])
            plane = cq.Plane(origin=cq.Vector(*[c * s for c in p["centre"]]),
                             xDir=cq.Vector(*u), normal=cq.Vector(*n))
            asm.add(solid, name=p["id"], loc=cq.Location(plane))
            n_emit += 1

    if n_emit == 0:
        raise SystemExit("no parts emitted")

    asm.save(args.out)
    print(f"wrote {args.out}: {n_emit} named solids, scale {s} mm/unit")
    for m in data.get("mates", []):
        print(f"  mate  {m['a']:<10} {m['type']:<20} {m['b']}")


if __name__ == "__main__":
    main()
