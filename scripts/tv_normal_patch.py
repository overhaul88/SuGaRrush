"""Flatten the completed patch by minimising the total variation of the normal field.

Implements Sec 7.2 of morphogenetic_completion.md, with the quiescence rule of Sec 8 (R1/R5:
"refinement only in the wound; observed tissue is quiescent").

The problem this solves
-----------------------
Once non-evidence is removed, the unobserved region is an honest hole and something must span it.
Every smooth-energy closure -- screened Poisson, Liepa fill + fairing, L2 bending -- minimises a
quadratic and therefore returns a DOME. That is not a tuning artefact; a dome is the correct
minimiser of the energy being posed. Sec Q4 of the document states the fix exactly: regularise with

    TV(n) = sum_e  len_e * angle(n_f, n_f')

an l1 penalty on dihedral angle. l1 penalties have sparse minimisers, and sparsity of a normal
field means the normal is piecewise CONSTANT -- i.e. flat faces meeting at sharp creases. The cube
face is then the minimiser, not something proposed by a prior. Same mechanism as ROF
total-variation denoising producing piecewise-constant images.

How it is realised
------------------
Two-step normal filtering + vertex fitting (Taubin; Sun et al.), with the key detail that the
normal filter is a **median**. The median is the l1 minimiser exactly as the mean is the l2
minimiser, so median-filtering the normal field is a descent step on TV(n) rather than on the
Dirichlet energy -- which is precisely the difference between a flat face and a dome.

    1. n_f  <-  normalise( median over the edge-neighbours of f )
    2. move each FREE vertex to best fit its incident faces' filtered normals

Sec 7.4's guarantee is respected in the strongest possible form: observed vertices never move at
all. The flow can only reshape the wound, so it is structurally incapable of altering measured
geometry, and the rim acts as Dirichlet data that the patch must meet.

Closedness is an invariant, not a goal (Sec 8): this only moves vertices, never adds or removes
them, so a closed input stays closed and manifoldness is untouched.
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d


def face_adjacency(T, n_faces):
    """Faces sharing an edge, as a padded index array."""
    edges = {}
    for fi, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (a, b) if a < b else (b, a)
            edges.setdefault(key, []).append(fi)
    nbr = [[] for _ in range(n_faces)]
    for fs in edges.values():
        if len(fs) == 2:
            nbr[fs[0]].append(fs[1])
            nbr[fs[1]].append(fs[0])
    width = max((len(x) for x in nbr), default=0)
    out = np.full((n_faces, width), -1, dtype=np.int64)
    for i, x in enumerate(nbr):
        out[i, :len(x)] = x
    return out


def median_filter_normals(FN, nbr, free_face):
    """Component-wise median of each face's own normal with its edge neighbours.

    Median (l1) rather than mean (l2): this is what makes the step descend TV(n) and hence produce
    piecewise-constant normals instead of a smoothly varying field.
    """
    out = FN.copy()
    valid = nbr >= 0
    idx = np.where(valid, nbr, 0)
    stack = np.concatenate([FN[idx], FN[:, None, :]], axis=1)          # (F, k+1, 3)
    mask = np.concatenate([valid, np.ones((len(FN), 1), bool)], axis=1)
    big = np.where(mask[..., None], stack, np.nan)
    med = np.nanmedian(big, axis=1)
    nrm = np.linalg.norm(med, axis=1, keepdims=True)
    good = (nrm[:, 0] > 1e-12) & free_face
    out[good] = med[good] / nrm[good]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="closed mesh")
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--wound-npz", default=None,
                    help="npz from observation_confidence.py --save-npz, computed on THIS mesh")
    ap.add_argument("--observed-mesh", default=None,
                    help="the PRE-closure, rho-filtered surface. The wound is then every face of "
                         "the input that lies farther than --provenance-tol from it, i.e. the "
                         "geometry the closure invented. Prefer this over --wound-npz: rho "
                         "recomputed AFTER closure certifies the patch as observed, because the "
                         "patch is the outermost surface and therefore visible to every camera. "
                         "Provenance must be carried through the closure, not re-derived from it.")
    ap.add_argument("--provenance-tol", type=float, default=0.004,
                    help="distance (fraction of object diagonal) beyond which a face counts as "
                         "invented rather than measured")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--vertex-steps", type=int, default=12,
                    help="vertex-fitting sweeps per normal filtering step")
    ap.add_argument("--step", type=float, default=0.6, help="vertex update relaxation")
    ap.add_argument("--dilate", type=int, default=1,
                    help="rings of observed faces adjacent to the wound also allowed to move, so "
                         "the crease can form at the rim instead of being pinned just inside it")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.inp)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    V = np.asarray(mesh.vertices).copy()
    T = np.asarray(mesh.triangles)
    if args.observed_mesh:
        ref = o3d.io.read_triangle_mesh(args.observed_mesh, enable_post_processing=True)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ref))
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        cen = V[T].mean(axis=1).astype(np.float32)
        dist = rc.compute_distance(o3d.core.Tensor(cen)).numpy()
        wound_face = dist > args.provenance_tol * diag
        print(f"provenance: {wound_face.sum():,} of {len(T):,} faces "
              f"({wound_face.mean()*100:.1f}%) are invented (>{args.provenance_tol:g} x diag "
              f"from the measured surface)")
    elif args.wound_npz:
        d = np.load(args.wound_npz)
        wound_face = np.asarray(d["unobserved"]).astype(bool)
        if len(wound_face) != len(T):
            raise SystemExit(f"wound mask has {len(wound_face)} faces but mesh has {len(T)}; "
                             "recompute observation_confidence.py on THIS mesh")
    else:
        raise SystemExit("need --observed-mesh (preferred) or --wound-npz")

    nbr = face_adjacency(T, len(T))
    # grow the movable set by a ring or two so the crease can land on the rim
    free_face = wound_face.copy()
    for _ in range(args.dilate):
        grown = free_face.copy()
        nb = nbr[free_face]
        grown[nb[nb >= 0]] = True
        free_face = grown

    free_vert = np.zeros(len(V), bool)
    free_vert[T[free_face].ravel()] = True
    # a vertex touching any OBSERVED face is measured geometry and must not move (Sec 7.4)
    observed_vert = np.zeros(len(V), bool)
    observed_vert[T[~wound_face].ravel()] = True
    movable = free_vert & ~observed_vert
    print(f"faces {len(T):,}  wound {wound_face.sum():,} ({wound_face.mean()*100:.1f}%)  "
          f"movable vertices {movable.sum():,} of {len(V):,}")
    if movable.sum() == 0:
        print("nothing movable; writing input unchanged")
        o3d.io.write_triangle_mesh(args.out, mesh)
        return

    # vertex -> incident faces
    vf = [[] for _ in range(len(V))]
    for fi, t in enumerate(T):
        for v in t:
            vf[v].append(fi)
    move_idx = np.nonzero(movable)[0]
    inc = [np.array(vf[v], dtype=np.int64) for v in move_idx]

    def tv_energy(verts):
        m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                       o3d.utility.Vector3iVector(T))
        m2.compute_triangle_normals()
        fn = np.asarray(m2.triangle_normals)
        valid = nbr >= 0
        dots = np.einsum("ij,ikj->ik", fn, fn[np.where(valid, nbr, 0)])
        ang = np.arccos(np.clip(dots, -1, 1))
        return float(ang[valid].sum() / 2)

    e0 = tv_energy(V)
    for it in range(args.iters):
        m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                       o3d.utility.Vector3iVector(T))
        m2.compute_triangle_normals()
        FN = np.asarray(m2.triangle_normals)
        FN = median_filter_normals(FN, nbr, free_face)

        centroids = V[T].mean(axis=1)
        for _ in range(args.vertex_steps):
            disp = np.zeros((len(move_idx), 3))
            for k, v in enumerate(move_idx):
                f = inc[k]
                n = FN[f]
                dv = centroids[f] - V[v]
                disp[k] = (n * np.einsum("ij,ij->i", n, dv)[:, None]).mean(0)
            V[move_idx] += args.step * disp
            centroids = V[T].mean(axis=1)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>3}: TV(n) = {tv_energy(V):.1f}")

    e1 = tv_energy(V)
    print(f"TV(n): {e0:.1f} -> {e1:.1f}  ({(e1-e0)/max(e0,1e-9)*100:+.1f}%)")

    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.compute_vertex_normals()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"wrote {args.out}")

    if args.report_json:
        with open(args.report_json, "w") as fh:
            json.dump({"input": args.inp, "output": args.out, "tv_before": e0, "tv_after": e1,
                       "wound_faces": int(wound_face.sum()), "movable_vertices": int(movable.sum())},
                      fh, indent=2)


if __name__ == "__main__":
    main()
