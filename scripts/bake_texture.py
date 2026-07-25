"""Stage 8 -- bake an observation-only texture onto the final cleaned mesh.

Why this exists
---------------
The pipeline ends at a geometry-only deliverable (output/final/<name>_final.ply):
carved + cleaned, with vertex normals but no UVs and no color. SuGaR *does* emit
a textured OBJ mid-pipeline, but it bakes from *Gaussian renders averaged with a
mean* onto the 500k-face pre-carve surface -- a mean smears a specular highlight
(seen in only a handful of the ~50 views that observe a texel) into a ghost that
rides across the whole surface.

This stage instead projects the *source frames* onto the final mesh and blends
with a **weighted median**, which rejects the specular outlier instead of
averaging it in. That is the load-bearing choice here, and it gives a partial
de-lighting for free. It matches the pipeline's observation-only stance: color
comes only from what a camera actually saw, never from the Gaussians.

Method (project-and-blend)
--------------------------
  1. xatlas unwrap the final mesh                         -> per-vertex UVs
  2. rasterize the atlas in UV space (nvdiffrast, ortho)  -> per-texel world
     position + normal + owning triangle id. No camera conventions involved.
  3. for each registered camera:
       - render the mesh's depth from that camera (nvdiffrast) for a z-test
       - project each texel's world position into the image
       - reject occluded texels (z-test), texels outside the U2Net mask, and
         grazing views (|dot(n, view)| < 0.2)
       - sample the GT color (bilinear); weight = |dot(n, view)| (obliquity)
       - keep, per texel, only the K most head-on observations (bounded memory,
         and the K best views are the ones worth blending anyway)
  4. per texel: weighted median of the surviving samples (not a mean)
  5. inpaint in-chart holes + dilate chart borders into the gutter so bilinear
     / mipmapping does not bleed background across UV seams
  6. report the unobserved fraction (coverage) as a first-class number
  7. export OBJ+MTL+PNG and a self-contained GLB

The camera math (COLMAP intrinsics/extrinsics + the pinhole projection) is the
exact, already-verified projection from carve_mesh.py -- reused so visibility
inherits a projection we know is correct. The z-test isolates the front visible
surface, so grazing only needs the *magnitude* of dot(n, view): that makes the
stage immune to whether the cleaned mesh's normals point in or out.

    python scripts/bake_texture.py --mesh output/final/object4_final.ply \
        --scene scenes/object4 --out-dir output/final --atlas 2048 --validate 8
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the verified COLMAP camera loader + pinhole conventions from the carve stage.
from carve_mesh import load_cameras  # noqa: E402

import nvdiffrast.torch as dr  # noqa: E402
import xatlas  # noqa: E402

DEV = "cuda"
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# ----------------------------------------------------------------------------- utils
def mult8(x):
    """nvdiffrast requires the render resolution to be a multiple of 8."""
    return int(math.ceil(x / 8.0) * 8)


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=DEV)


def project_clip(Pworld_t, cam):
    """World points -> nvdiffrast clip coords, using carve_mesh's exact pinhole.

    The projection u=fx*X/Z+cx, v=fy*Y/Z+cy is the one carving used, so it is
    known-correct for these scenes. We only re-express it as clip coords with w=z
    so that after perspective divide, x,y land at (u,v)/(W,H) and z is a monotone
    NDC depth -- visibility then inherits carve's correct projection exactly.
    """
    R = to_t(cam["R"]); t = to_t(cam["t"])
    W, H = float(cam["w"]), float(cam["h"])
    Pc = Pworld_t @ R.T + t                 # camera space (COLMAP: +Z fwd, y down)
    z = Pc[:, 2]
    zc = z.clamp_min(1e-8)
    u = cam["fx"] * Pc[:, 0] / zc + cam["cx"]
    v = cam["fy"] * Pc[:, 1] / zc + cam["cy"]
    zf = z[z > 1e-6]
    if zf.numel() == 0:
        znear, zfar = 1e-3, 1.0
    else:
        znear = float(zf.min().clamp_min(1e-4)); zfar = float(zf.max())
        if zfar <= znear:
            zfar = znear + 1e-3
    zndc = (z - znear) / (zfar - znear) * 2.0 - 1.0
    clip = torch.stack([(2.0 * u / W - 1.0) * z,
                        (2.0 * v / H - 1.0) * z,
                        zndc * z, z], dim=1)
    return clip[None].contiguous(), z


def render_depth_and_faces(glctx, Vworld_t, faces_i, cam, Wr, Hr):
    """Rasterize the mesh from a camera; return (pix_to_face 0-based, cam-depth)."""
    clip, zvert = project_clip(Vworld_t, cam)
    rast, _ = dr.rasterize(glctx, clip.float(), faces_i, resolution=[Hr, Wr])
    ptf = rast[0, ..., 3].int() - 1                          # -1 = empty, else face idx
    depth, _ = dr.interpolate(zvert[None, :, None].contiguous(), rast, faces_i)
    return ptf, depth[0, ..., 0]                             # (Hr,Wr), (Hr,Wr)


# ------------------------------------------------------------------------ blend (top-K)
def topk_weighted_median(acc_rgb, acc_w, acc_cnt, chunk=200_000):
    """Per-texel weighted median RGB from a fixed (T,K) top-weight sample buffer.

    Within each texel, sort its (<=K) samples by luminance and take the sample
    whose cumulative weight first crosses half the total. That returns an *actual
    observed color*, discarding bright specular outliers (which sort to the top
    and only win if they dominate the weight). Fully vectorized in (T,K) chunks.
    """
    T, K = acc_w.shape
    out = np.zeros((T, 3), np.uint8)
    kidx = np.arange(K)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        cnt = acc_cnt[s:e]
        sel = cnt > 0
        if not sel.any():
            continue
        rgb = acc_rgb[s:e].astype(np.float32)                # (c,K,3)
        w = acc_w[s:e].astype(np.float64).copy()             # (c,K)
        valid = kidx[None, :] < cnt[:, None]
        w *= valid
        lum = rgb @ LUMA
        lum[~valid] = np.inf                                 # invalid slots sort last
        order = np.argsort(lum, axis=1)
        cw = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1)
        target = 0.5 * np.maximum(cw[:, -1:], 1e-12)
        first = np.argmax(cw >= target, axis=1)              # first crossing per row
        pick = np.take_along_axis(order, first[:, None], axis=1)[:, 0]
        chosen = rgb[np.arange(e - s), pick].astype(np.uint8)
        blk = out[s:e]; blk[sel] = chosen[sel]; out[s:e] = blk
    return out, (acc_cnt > 0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="final cleaned mesh (.ply)")
    ap.add_argument("--scene", required=True, help="COLMAP scene dir (images/, masks/, sparse/0)")
    ap.add_argument("--masks", default=None, help="mask dir (default <scene>/masks)")
    ap.add_argument("--images", default=None, help="GT image dir (default <scene>/images)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None, help="output basename (default from --mesh)")
    ap.add_argument("--atlas", type=int, default=2048, help="texture atlas side (mult of 8)")
    ap.add_argument("--views-per-texel", type=int, default=24,
                    help="K: keep the K most head-on observations per texel (bounds memory)")
    ap.add_argument("--grazing", type=float, default=0.2, help="reject views with |dot(n,v)| below this")
    ap.add_argument("--pad", type=int, default=6, help="dilate charts N px into the gutter")
    ap.add_argument("--mask-erode", type=int, default=2, help="erode masks N px to avoid silhouette bleed")
    ap.add_argument("--depth-tol", type=float, default=0.005, help="z-test tolerance, fraction of bbox diag")
    ap.add_argument("--validate", type=int, default=0, help="render K views from the baked mesh, report PSNR")
    args = ap.parse_args()

    atlas = mult8(args.atlas)
    K = max(1, args.views_per_texel)
    masks_dir = args.masks or os.path.join(args.scene, "masks")
    images_dir = args.images or os.path.join(args.scene, "images")
    name = args.name or os.path.basename(args.mesh).replace("_final", "").rsplit(".", 1)[0]
    os.makedirs(args.out_dir, exist_ok=True)

    # -- 1. load mesh ---------------------------------------------------------------
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(args.mesh)
    m.remove_duplicated_vertices(); m.remove_unreferenced_vertices()
    m.remove_degenerate_triangles(); m.compute_vertex_normals()
    V = np.asarray(m.vertices, np.float32)
    Fc = np.asarray(m.triangles, np.int32)
    VN = np.asarray(m.vertex_normals, np.float32)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    depth_tol = args.depth_tol * diag
    print(f"mesh: {len(Fc):,} tris, {len(V):,} verts, bbox diag {diag:.3f}")

    # -- 2. UV unwrap ---------------------------------------------------------------
    vmapping, uv_faces, uvs = xatlas.parametrize(V, Fc)
    uv_faces = uv_faces.astype(np.int32)
    assert len(uv_faces) == len(Fc), "xatlas dropped/reordered faces; tri-id mapping unsafe"
    Vuv = V[vmapping]; Nuv = VN[vmapping]
    print(f"xatlas: {len(uvs):,} uv-verts ({len(uvs)-len(V):,} seam splits), atlas {atlas}^2, K={K}")

    glctx = dr.RasterizeCudaContext()

    # -- 3. rasterize the atlas in UV space (orthographic; no camera conventions) ---
    uv_t = to_t(uvs)
    clip_uv = torch.cat([uv_t * 2 - 1, torch.zeros_like(uv_t[:, :1]),
                         torch.ones_like(uv_t[:, :1])], dim=1)[None].contiguous()
    tri_uv = to_t(uv_faces, torch.int32)
    rast_uv, _ = dr.rasterize(glctx, clip_uv.float(), tri_uv, resolution=[atlas, atlas])
    attr = to_t(np.concatenate([Vuv, Nuv], axis=1))[None].contiguous()
    interp, _ = dr.interpolate(attr, rast_uv, tri_uv)
    P_tex = interp[0, ..., :3].reshape(-1, 3)
    N_tex = interp[0, ..., 3:6].reshape(-1, 3)
    N_tex = N_tex / N_tex.norm(dim=1, keepdim=True).clamp_min(1e-8)
    tri_tex = (rast_uv[0, ..., 3].int() - 1).reshape(-1)
    chart = tri_tex >= 0
    tex_idx = torch.nonzero(chart, as_tuple=False).squeeze(1)   # flat ids of chart texels
    Pv = P_tex[tex_idx]; Nv = N_tex[tex_idx]
    T = int(tex_idx.numel())
    n_texels = atlas * atlas
    flat_ids = tex_idx.cpu().numpy()
    print(f"atlas: {T:,} chart texels of {n_texels:,} ({100*T/n_texels:.1f}%)")

    cams = load_cameras(args.scene)
    print(f"cameras: {len(cams)}")
    Vt = to_t(V); Ft = to_t(Fc, torch.int32)

    # bounded per-texel accumulator (compact texel space 0..T-1)
    acc_w = np.zeros((T, K), np.float16)
    acc_rgb = np.zeros((T, K, 3), np.uint8)
    acc_cnt = np.zeros(T, np.int32)
    raw_cnt = np.zeros(T, np.int32)             # honest observation count (uncapped)
    used = 0

    # -- 4. per-camera gather -------------------------------------------------------
    for ci, c in enumerate(cams):
        mp = os.path.join(masks_dir, os.path.splitext(c["name"])[0] + ".png")
        ip = os.path.join(images_dir, c["name"])
        if not (os.path.isfile(mp) and os.path.isfile(ip)):
            continue
        W, H = int(c["w"]), int(c["h"]); Wr, Hr = mult8(W), mult8(H)

        ptf, depth = render_depth_and_faces(glctx, Vt, Ft, c, Wr, Hr)

        R = to_t(c["R"]); tt = to_t(c["t"])
        Pc = Pv @ R.T + tt
        z = Pc[:, 2]; zc = z.clamp_min(1e-8)
        u = c["fx"] * Pc[:, 0] / zc + c["cx"]
        v = c["fy"] * Pc[:, 1] / zc + c["cy"]
        front = z > 1e-6
        col = (u * Wr / W).round().long(); row = (v * Hr / H).round().long()
        inb = front & (col >= 0) & (col < Wr) & (row >= 0) & (row < Hr)
        ri = row.clamp(0, Hr - 1); cj = col.clamp(0, Wr - 1)
        # occlusion z-test: texel visible iff its depth matches the front render
        vis = inb & (ptf[ri, cj] >= 0) & ((z - depth[ri, cj]) <= depth_tol)
        # obliquity (orientation-free): magnitude of dot(n, view)
        cam_center = -(R.T @ tt)
        view = cam_center[None] - Pv
        view = view / view.norm(dim=1, keepdim=True).clamp_min(1e-8)
        wmag = (Nv * view).sum(1).abs()
        graz_ok = wmag >= args.grazing
        # mask (downscaled -> rescale like carve), eroded to avoid silhouette bleed
        mask = np.array(Image.open(mp).convert("L")) > 127
        if args.mask_erode > 0:
            mask = ndimage.binary_erosion(mask, iterations=args.mask_erode)
        mh, mw = mask.shape
        mask_t = torch.from_numpy(mask).to(DEV)
        mcol = (u * (mw / W)).round().long().clamp(0, mw - 1)
        mrow = (v * (mh / H)).round().long().clamp(0, mh - 1)
        in_mask = mask_t[mrow, mcol]

        keep = vis & graz_ok & in_mask
        ki = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if ki.numel() == 0:
            continue

        # bilinear GT color at the kept texels
        img = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
        img_t = torch.from_numpy(img).to(DEV).permute(2, 0, 1)[None]
        gx = 2.0 * u[ki] / W - 1.0; gy = 2.0 * v[ki] / H - 1.0
        grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2)
        samp = F.grid_sample(img_t, grid, mode="bilinear",
                             align_corners=False, padding_mode="border")
        rgb = (samp[0, :, :, 0].T.clamp(0, 1) * 255).to(torch.uint8)

        # --- insert into the bounded top-K accumulator (ki unique within a camera) ---
        idc = ki.cpu().numpy()
        wv = wmag[ki].cpu().numpy().astype(np.float16)
        cv = rgb.cpu().numpy()
        raw_cnt[idc] += 1
        cnt = acc_cnt[idc]
        nf = cnt < K
        r1 = idc[nf]; s1 = cnt[nf]
        acc_w[r1, s1] = wv[nf]; acc_rgb[r1, s1] = cv[nf]; acc_cnt[r1] = s1 + 1
        r2 = idc[~nf]
        if r2.size:
            sub = acc_w[r2]
            mn = sub.argmin(1); mnw = sub[np.arange(r2.size), mn]
            better = wv[~nf] > mnw
            rr = r2[better]; cc = mn[better]
            acc_w[rr, cc] = wv[~nf][better]; acc_rgb[rr, cc] = cv[~nf][better]
        used += 1
        if ci % 50 == 0:
            print(f"  view {ci}/{len(cams)}  observed {int((acc_cnt>0).sum()):,}/{T:,}", flush=True)

    if used == 0 or (acc_cnt > 0).sum() == 0:
        raise SystemExit("no texels observed by any view -- check masks/poses")

    # -- 5. weighted-median blend ---------------------------------------------------
    out_c, obs_c = topk_weighted_median(acc_rgb, acc_w, acc_cnt)
    tex_flat = np.zeros((n_texels, 3), np.uint8)
    observed = np.zeros(n_texels, bool)
    tex_flat[flat_ids] = out_c
    observed[flat_ids] = obs_c

    # -- 6. inpaint in-chart holes + dilate charts into the gutter ------------------
    tex2d = tex_flat.reshape(atlas, atlas, 3)
    obs2d = observed.reshape(atlas, atlas)
    chart2d = chart.reshape(atlas, atlas).cpu().numpy()
    dist, ind = ndimage.distance_transform_edt(~obs2d, return_distances=True,
                                               return_indices=True)
    nearest = tex2d[tuple(ind)]
    fill_here = (~obs2d) & (chart2d | (dist <= args.pad))
    out2d = tex2d.copy(); out2d[fill_here] = nearest[fill_here]

    # -- 7. coverage report ---------------------------------------------------------
    coverage = int(obs_c.sum()) / max(T, 1)
    seen = raw_cnt[raw_cnt > 0]
    stats = dict(
        name=name, atlas=atlas, K=K, cameras=len(cams), used_views=used,
        chart_texels=T, observed_texels=int(obs_c.sum()),
        coverage_pct=round(100 * coverage, 2),
        unobserved_pct=round(100 * (1 - coverage), 2),
        mean_views_per_texel=round(float(seen.mean()) if seen.size else 0, 2),
        seam_pad_px=args.pad, grazing=args.grazing,
    )
    stats_path = os.path.join(args.out_dir, f"{name}_texbake.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"COVERAGE {stats['coverage_pct']}%  (unobserved {stats['unobserved_pct']}%, "
          f"mean {stats['mean_views_per_texel']} views/texel)  -> {stats_path}")

    cov = np.zeros((atlas, atlas), np.uint8)
    cov[chart2d] = 90; cov[obs2d & chart2d] = 255
    Image.fromarray(np.flipud(cov)).save(os.path.join(args.out_dir, f"{name}_coverage.png"))

    # -- 8. export OBJ+MTL+PNG and GLB (texture flipped to top-down PNG convention) --
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    tex_png = Image.fromarray(np.flipud(out2d))       # bottom-up raster -> top-down file
    png_path = os.path.join(args.out_dir, f"{name}_texture.png")
    tex_png.save(png_path)
    mat = PBRMaterial(name=name, baseColorTexture=tex_png, metallicFactor=0.0, roughnessFactor=1.0)
    visual = TextureVisuals(uv=uvs, image=tex_png, material=mat)
    tm = trimesh.Trimesh(vertices=Vuv, faces=uv_faces, visual=visual, process=False)
    glb_path = os.path.join(args.out_dir, f"{name}_textured.glb")
    tm.export(glb_path)
    # OBJ writes a generic material.mtl/material_0.png; isolate it per-object so
    # baking several meshes into one out-dir does not clobber each other's texture.
    obj_dir = os.path.join(args.out_dir, f"{name}_textured_obj")
    os.makedirs(obj_dir, exist_ok=True)
    obj_path = os.path.join(obj_dir, f"{name}.obj")
    tm.export(obj_path)
    print(f"wrote {glb_path}\n      {obj_path} (+ .mtl/.png)\n      {png_path}")

    # -- 9. optional validation: render the baked mesh, PSNR vs GT inside the mask --
    if args.validate > 0:
        idxs = np.linspace(0, len(cams) - 1, args.validate).astype(int)
        Vuv_t = to_t(Vuv); tex_gpu = to_t(np.flipud(out2d).copy()) / 255.0
        uvattr = torch.cat([uv_t, torch.zeros_like(uv_t[:, :1])], dim=1)[None].contiguous()
        psnrs = []
        for k in idxs:
            c = cams[k]
            mp = os.path.join(masks_dir, os.path.splitext(c["name"])[0] + ".png")
            ip = os.path.join(images_dir, c["name"])
            if not (os.path.isfile(mp) and os.path.isfile(ip)):
                continue
            W, H = int(c["w"]), int(c["h"]); Wr, Hr = mult8(W), mult8(H)
            clip, _ = project_clip(Vuv_t, c)
            rast, _ = dr.rasterize(glctx, clip.float(), tri_uv, resolution=[Hr, Wr])
            uvpix, _ = dr.interpolate(uvattr, rast, tri_uv)
            g = uvpix[0, ..., :2] * 2 - 1
            g[..., 1] = -g[..., 1]                              # texture PNG is top-down
            ren = F.grid_sample(tex_gpu.permute(2, 0, 1)[None], g[None],
                                mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            face = rast[0, ..., 3] > 0
            gt = np.asarray(Image.open(ip).convert("RGB").resize((Wr, Hr)), np.float32) / 255.0
            mask = np.array(Image.open(mp).convert("L").resize((Wr, Hr))) > 127
            sel = face & to_t(mask, torch.bool)
            if sel.sum() < 100:
                continue
            mse = ((ren[sel] - to_t(gt)[sel]) ** 2).mean().item()
            psnrs.append(10 * math.log10(1.0 / max(mse, 1e-8)))
        if psnrs:
            print(f"VALIDATION: mean PSNR {np.mean(psnrs):.2f} dB over {len(psnrs)} held views "
                  f"(inside mask). Low PSNR => UV/flip convention wrong.")
            stats["validation_psnr_db"] = round(float(np.mean(psnrs)), 2)
            with open(stats_path, "w") as fh:
                json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
