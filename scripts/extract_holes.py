"""Find the perforation grid on a fitted plate, and derive metric scale from it.

Stage 7c. Two outputs from one measurement:
  * the hole pattern, so the emitted CAD plate is perforated like the real part;
  * **absolute scale**, because Meccano holes sit on a fixed 1/2 inch (12.7 mm)
    pitch. Detecting the pitch turns a unitless photogrammetric reconstruction
    into millimetres with no operator measurement at all.

Why rasterise instead of tracing boundary loops: measured on object1, the mesh
has Euler characteristic -240 (genus ~121), i.e. the perforations are through
*tunnels* in a closed surface, not open boundary loops. Loop tracing finds
nothing. Projecting the plate's faces onto its own plane and looking for enclosed
empty regions is robust to either topology.

    python scripts/extract_holes.py --mesh object1_final.ply --parts parts.json \
        --part plate_0 --out holes.json
"""

import argparse
import json

import numpy as np
import open3d as o3d
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--part", default="plate_0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell", type=float, default=0.004, help="raster cell size, scene units")
    ap.add_argument("--samples", type=int, default=400000)
    ap.add_argument("--pitch-mm", type=float, default=12.7,
                    help="known hole pitch of the real part (Meccano = 12.7 mm)")
    ap.add_argument("--min-hole-cells", type=int, default=6)
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(args.mesh)
    m.remove_duplicated_vertices()
    V = np.asarray(m.vertices)
    T = np.asarray(m.triangles)

    data = json.load(open(args.parts))
    p = next(q for q in data["parts"] if q["id"] == args.part)
    n = np.array(p["normal"], float); n /= np.linalg.norm(n)
    u = np.array(p["axis_u"], float); u /= np.linalg.norm(u)
    v = np.array(p["axis_v"], float); v /= np.linalg.norm(v)
    c = np.array(p["centre"], float)
    L, W, Th = p["length"], p["width"], p["thickness"]
    print(f"{args.part}: {L:.3f} x {W:.3f} x t={Th:.4f}")

    C = V[T].mean(1) - c
    sel = ((np.abs(C @ n) < Th * 2.5) & (np.abs(C @ u) < L / 2 * 1.05)
           & (np.abs(C @ v) < W / 2 * 1.05))
    sub = T[sel]
    print(f"faces in plate slab: {sel.sum():,}")

    # Dense barycentric sampling of the slab's faces, projected to plate 2D.
    a, b, cc = V[sub[:, 0]], V[sub[:, 1]], V[sub[:, 2]]
    ar = np.linalg.norm(np.cross(b - a, cc - a), axis=1) / 2
    prob = ar / ar.sum()
    idx = np.random.default_rng(0).choice(len(sub), size=args.samples, p=prob)
    r1 = np.random.default_rng(1).random(args.samples)
    r2 = np.random.default_rng(2).random(args.samples)
    sq = np.sqrt(r1)
    P = (1 - sq)[:, None] * a[idx] + (sq * (1 - r2))[:, None] * b[idx] + (sq * r2)[:, None] * cc[idx]
    rel = P - c
    x, y = rel @ u, rel @ v

    nx = int(np.ceil(L / args.cell)) + 2
    ny = int(np.ceil(W / args.cell)) + 2
    ix = np.clip(((x + L / 2) / args.cell).astype(int) + 1, 0, nx - 1)
    iy = np.clip(((y + W / 2) / args.cell).astype(int) + 1, 0, ny - 1)
    occ = np.zeros((nx, ny), bool)
    occ[ix, iy] = True
    # close single-cell speckle so sampling noise is not read as holes
    occ = ndimage.binary_closing(occ, np.ones((3, 3)))
    print(f"raster {nx}x{ny} @ {args.cell}, material fill {100*occ.mean():.1f}%")

    empty = ~occ
    lab, nlab = ndimage.label(empty)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    holes = []
    for k in range(1, nlab + 1):
        if k in border:
            continue
        cells = np.argwhere(lab == k)
        if len(cells) < args.min_hole_cells:
            continue
        cy_, cx_ = cells[:, 0].mean(), cells[:, 1].mean()
        rad = np.sqrt(len(cells) / np.pi) * args.cell
        # roundness: a real hole is compact, a crack is not
        ext = cells.max(0) - cells.min(0) + 1
        rounds = min(ext) / max(ext)
        holes.append(dict(u=float((cy_ - 1) * args.cell - L / 2),
                          v=float((cx_ - 1) * args.cell - W / 2),
                          r=float(rad), cells=int(len(cells)), roundness=float(rounds)))
    holes = [h for h in holes if h["roundness"] > 0.45]
    print(f"enclosed empty regions accepted as holes: {len(holes)}")
    if len(holes) < 4:
        raise SystemExit("too few holes detected; lower --cell or --min-hole-cells")

    rr = np.array([h["r"] for h in holes])
    print(f"hole radius: median {np.median(rr):.4f}  p10 {np.percentile(rr,10):.4f}  "
          f"p90 {np.percentile(rr,90):.4f}")

    # Nearest-neighbour spacing -> grid pitch
    Q = np.array([[h["u"], h["v"]] for h in holes])
    d = np.linalg.norm(Q[:, None] - Q[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    # modal pitch via a histogram peak, robust to a few doubled spacings
    hist, edges = np.histogram(nn, bins=40)
    pitch = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
    med = float(np.median(nn))
    print(f"nearest-neighbour spacing: median {med:.4f}  modal {pitch:.4f}")

    scale = args.pitch_mm / pitch
    print(f"\nSCALE from {args.pitch_mm} mm pitch: {scale:.1f} mm per scene unit")
    print(f"  -> plate becomes {L*scale:.1f} x {W*scale:.1f} x {Th*scale:.2f} mm")
    print(f"  -> hole diameter {2*np.median(rr)*scale:.2f} mm")

    out = dict(part=args.part, pitch_scene=pitch, pitch_mm=args.pitch_mm,
               mm_per_unit=scale, hole_radius_scene=float(np.median(rr)),
               n_holes=len(holes), holes=holes)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
