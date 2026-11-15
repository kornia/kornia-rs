#!/usr/bin/env python
"""Shared mesh accuracy/completeness scorer (NICE-SLAM style) for the TSDF baselines.

Run:  /mnt/data/venvs/tsdf-bench/bin/python -I score_mesh.py MESH --gt GT.ply [--cull-to-frustum --traj traj.txt] ...
Prints exactly ONE JSON line on stdout (logs go to stderr). Exit 0 on success or on an
empty (0-triangle, 0-vertex) prediction; exit 2 on a hard input error (JSON line still printed).

Metric (metres internally, reported in cm):
  - N points sampled uniformly by AREA on pred and on GT (numpy rng, distinct fixed seeds).
  - accuracy_cm      = mean dist  pred sample -> nearest GT sample
  - completeness_cm  = mean dist  GT sample   -> nearest pred sample
  - completion_ratio_pct = % GT samples with dist to pred < thresh (5 cm)
  - precision_pct        = % pred samples with dist to GT < thresh
  - f_score_pct          = harmonic mean of the two
  Extras (point-to-SURFACE via Open3D RaycastingScene, removes the sample-spacing floor):
  accuracy_surf_cm, completeness_surf_cm, completion_ratio_surf_pct, precision_surf_pct.
"""
import argparse, hashlib, json, os, sys, time

import numpy as np

SCRIPT = os.path.abspath(__file__)
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(SCRIPT)), "cache")
# Replica (frozen DECISIONS.md): 1200x680, fx=fy=600, cx=599.5, cy=339.5, depth scale 6553.5
DEF_INTR = dict(w=1200, h=680, fx=600.0, fy=600.0, cx=599.5, cy=339.5, depth_scale=6553.5)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_PLY_T = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2",
          "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4",
          "double": "f8", "float64": "f8"}


def _fan(rows):
    """rows: list of 1-D index arrays (polygons) -> (T (m,3), n_polygons_with_>3_verts)."""
    tri, split = [], 0
    for r in rows:
        split += int(len(r) > 3)
        tri.extend((r[0], r[i], r[i + 1]) for i in range(1, len(r) - 1))
    return np.array(tri, dtype=np.int64).reshape(-1, 3), split


def read_ply(path):
    """Own PLY reader. Open3D's reader SILENTLY keeps only the triangles of a mesh that contains quads --
    Replica GT meshes are all quads and Open3D returns 16516 of 953647 faces (with only a warning).
    Supports ascii / binary LE / binary BE, vertex then face element, polygons fan-triangulated.
    Returns (V float64 (n,3), T int64 (m,3), n_polygons_split)."""
    with open(path, "rb") as f:
        buf = f.read()
    e = buf.index(b"\n", buf.index(b"end_header")) + 1
    hdr = buf[:e].decode("latin-1").splitlines()
    fmt = next(l.split()[1] for l in hdr if l.startswith("format"))
    elems = []
    for l in hdr:
        w = l.split()
        if w and w[0] == "element":
            elems.append(dict(name=w[1], n=int(w[2]), props=[]))
        elif w and w[0] == "property":
            elems[-1]["props"].append(w[1:])
    if not elems or elems[0]["name"] != "vertex":
        raise ValueError("PLY: first element must be 'vertex'")
    end = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
    ascii_ = fmt == "ascii"
    if not ascii_ and end is None:
        raise ValueError(f"PLY format {fmt} unsupported")
    off, li = e, 0
    lines = buf[e:].split(b"\n") if ascii_ else None
    V, T, split = None, np.zeros((0, 3), np.int64), 0
    for el in elems:
        n, props = el["n"], el["props"]
        if el["name"] == "vertex":
            if any(p[0] == "list" for p in props):
                raise ValueError("PLY: list property in vertex element unsupported")
            if ascii_ and n == 0:
                V = np.zeros((0, 3))
            elif ascii_:
                a = np.fromstring(b" ".join(lines[li:li + n]).decode(), sep=" ", dtype=np.float64).reshape(n, -1)
                li += n
                cols = [p[1] for p in props]
                V = a[:, [cols.index("x"), cols.index("y"), cols.index("z")]]
            else:
                dt = np.dtype([(p[1], end + _PLY_T[p[0]]) for p in props])
                a = np.frombuffer(buf, dtype=dt, count=n, offset=off)
                off += n * dt.itemsize
                V = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
        elif el["name"] == "face":
            if len(props) != 1 or props[0][0] != "list":
                raise ValueError("PLY: face element must be a single list property")
            _, ct, it, _nm = props[0]
            if ascii_:
                rows = [np.fromstring(l, sep=" ", dtype=np.int64)[1:] for l in lines[li:li + n]]
                T, split = _fan(rows)
            elif n:
                cdt, idt = np.dtype(end + _PLY_T[ct]), np.dtype(end + _PLY_T[it])
                raw = np.frombuffer(buf, dtype=np.uint8, offset=off)
                k = int(np.frombuffer(raw[:cdt.itemsize], dtype=cdt)[0])
                rec = cdt.itemsize + k * idt.itemsize
                fast = rec * n <= len(raw)
                if fast:
                    m = raw[:rec * n].reshape(n, rec)
                    cnt = np.frombuffer(np.ascontiguousarray(m[:, :cdt.itemsize]).tobytes(), dtype=cdt)
                    fast = bool((cnt == k).all())
                if fast:                          # uniform polygon size: vectorised
                    F = np.frombuffer(np.ascontiguousarray(m[:, cdt.itemsize:]).tobytes(), dtype=idt).reshape(n, k).astype(np.int64)
                    T = np.concatenate([F[:, [0, i, i + 1]] for i in range(1, k - 1)]) if k >= 3 else np.zeros((0, 3), np.int64)
                    split = n if k > 3 else 0
                else:                             # mixed polygon sizes: walk
                    rows, p = [], off
                    for _ in range(n):
                        c = int(np.frombuffer(buf, dtype=cdt, count=1, offset=p)[0]); p += cdt.itemsize
                        rows.append(np.frombuffer(buf, dtype=idt, count=c, offset=p).astype(np.int64)); p += c * idt.itemsize
                    T, split = _fan(rows)
        else:
            break
    if T.size and (T.min() < 0 or T.max() >= len(V)):
        raise ValueError("PLY: face index out of range")
    return V, T, split


def load_mesh(path):
    """-> (legacy o3d mesh, V, T, n_polygons_split). PLY via read_ply; stl/obj/off via Open3D."""
    import open3d as o3d
    if path.lower().endswith(".ply"):
        v, t, split = read_ply(path)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t.astype(np.int32)))
        return m, v, t, split
    m = o3d.io.read_triangle_mesh(path)
    return m, np.asarray(m.vertices, dtype=np.float64), np.asarray(m.triangles, dtype=np.int64), 0


def sample_surface(v, t, n, seed):
    """n points uniform by area (triangle ~ area, sqrt barycentric). Deterministic in seed."""
    tri = v[t]                                    # (T,3,3)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    tot = area.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(t), size=n, p=area / tot)
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    a, b, c = 1 - r1, r1 * (1 - r2), r1 * r2
    p = a[:, None] * tri[idx, 0] + b[:, None] * tri[idx, 1] + c[:, None] * tri[idx, 2]
    return p, float(tot)


def load_traj(path):
    a = np.loadtxt(path)
    return a.reshape(-1, 4, 4)


def c2w_opencv(T, convention):
    """Return camera-to-world with OpenCV axes (x right, y down, z forward).
    'opengl' (NICE-SLAM Replica traj.txt): camera looks -z, y up -> flip y and z columns."""
    T = T.copy()
    if convention == "opengl":
        T[:, :3, 1] *= -1
        T[:, :3, 2] *= -1
    return T


def parse_frames(spec, n_total):
    s = [int(x) if x else None for x in (spec.split(":") + ["", ""])[:3]]
    sl = slice(s[0], s[1], s[2])
    return list(range(n_total))[sl]


def read_depth(path):
    import open3d as o3d
    return np.asarray(o3d.io.read_image(path))


def visible_mask(points, traj_cv, frame_ids, depth_dir, intr, dmin, dmax, margin, depth_name="depth{:06d}.png"):
    """Point is 'seen' if, in >=1 frame: z in [dmin,dmax), projects inside image, sensor depth at the
    pixel is valid (>0, < dmax) and z <= depth + margin (i.e. not behind the observed surface)."""
    seen = np.zeros(len(points), bool)
    w, h = int(intr["w"]), int(intr["h"])
    P = points
    for k, fi in enumerate(frame_ids):
        T = traj_cv[fi]
        R, t = T[:3, :3], T[:3, 3]
        pc = (P - t) @ R                          # R^T (p - t)
        z = pc[:, 2]
        ok = (z >= dmin) & (z < dmax)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            continue
        zz = z[idx]
        u = np.rint(intr["fx"] * pc[idx, 0] / zz + intr["cx"]).astype(np.int64)
        vv = np.rint(intr["fy"] * pc[idx, 1] / zz + intr["cy"]).astype(np.int64)
        inb = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        idx, zz, u, vv = idx[inb], zz[inb], u[inb], vv[inb]
        if idx.size == 0:
            continue
        d = read_depth(os.path.join(depth_dir, depth_name.format(fi))).astype(np.float64) / intr["depth_scale"]
        dp = d[vv, u]
        vis = (dp > 0) & (dp < dmax) & (zz <= dp + margin)
        seen[idx[vis]] = True
        if (k + 1) % 250 == 0:
            log(f"  visibility {k+1}/{len(frame_ids)} seen={seen.mean()*100:.1f}%")
    return seen


def nn_dist(src, dst, workers):
    from scipy.spatial import cKDTree
    return cKDTree(dst).query(src, workers=workers)[0]


def surf_dist(mesh_legacy, pts):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy))
    return sc.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy().astype(np.float64)


def stats(d, thr):
    return float(d.mean()), float((d < thr).mean() * 100.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", help="predicted mesh (ply/stl/obj), world frame = traj.txt world")
    ap.add_argument("--gt", required=True)
    ap.add_argument("-n", "--num-points", type=int, default=200000)
    ap.add_argument("--seed-pred", type=int, default=0)
    ap.add_argument("--seed-gt", type=int, default=1)
    ap.add_argument("--thresh-cm", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=3, help="KD-tree threads")
    ap.add_argument("--cull-to-frustum", action="store_true",
                    help="drop GT samples never seen (frustum + depth-test) in the integrated frames")
    ap.add_argument("--cull-pred", action="store_true", help="also cull pred samples (default off: floaters must hurt)")
    ap.add_argument("--traj", help="traj.txt (4x4 c2w row-major, 1 matrix per line)")
    ap.add_argument("--frames", default="0:2000:1", help="python slice START:STOP:STEP of frame ids integrated")
    ap.add_argument("--depth-dir", help="dir with depthNNNNNN.png (default: <traj dir>/results)")
    ap.add_argument("--cam-convention", choices=["opengl", "opencv"], default="opencv",
                    help="axes of traj.txt c2w. Replica traj.txt here = opencv (x right, y down, z fwd; NO flip), "
                         "verified against sensor depth by validate_scorer.py (3 mm median residual; opengl flip = 1 m garbage)")
    ap.add_argument("--cull-dmin", type=float, default=0.1)
    ap.add_argument("--cull-dmax", type=float, default=7.0, help="sensor range (frozen Row M), NOT a library depth_max")
    ap.add_argument("--cull-margin", type=float, default=0.05, help="m behind observed depth still counted as seen")
    ap.add_argument("--intr", default="1200,680,600,600,599.5,339.5,6553.5", help="w,h,fx,fy,cx,cy,depth_scale")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--label", default="", help="free-text tag echoed into the JSON")
    a = ap.parse_args(argv)

    import open3d as o3d
    t0 = time.time()
    out = dict(mesh=os.path.abspath(a.mesh), gt=os.path.abspath(a.gt), label=a.label, n_points=a.num_points,
               seed_pred=a.seed_pred, seed_gt=a.seed_gt, thresh_cm=a.thresh_cm, open3d=o3d.__version__,
               numpy=np.__version__, scorer_sha256=sha256_file(SCRIPT), cull_to_frustum=bool(a.cull_to_frustum),
               cull_pred=bool(a.cull_pred))

    def fail(msg, code=2):
        out["error"] = msg
        print(json.dumps(out)); sys.exit(code)

    for p in (a.mesh, a.gt):
        if not os.path.isfile(p):
            fail(f"file not found: {p}")
    try:
        gm, gv, gt_, gsplit = load_mesh(a.gt)
    except Exception as ex:
        fail(f"cannot read GT: {ex!r}")
    if len(gt_) == 0:
        fail("GT has no triangles")
    try:
        pm, pv, pt, psplit = load_mesh(a.mesh)
    except Exception as ex:
        fail(f"cannot read pred mesh: {ex!r}")
    out.update(pred_vertices=int(len(pv)), pred_triangles=int(len(pt)), gt_vertices=int(len(gv)), gt_triangles=int(len(gt_)),
               gt_polygons_split=int(gsplit), pred_polygons_split=int(psplit))
    if len(pt) == 0 and len(pv) > 0:
        fail("pred has vertices but no triangles (point cloud?): not a mesh, refusing to score")
    if len(pt) == 0:
        out.update(empty=True, accuracy_cm=None, completeness_cm=None, completion_ratio_pct=None, precision_pct=None,
                   f_score_pct=None, accuracy_surf_cm=None, completeness_surf_cm=None,
                   completion_ratio_surf_pct=None, precision_surf_pct=None, wall_s=round(time.time() - t0, 2))
        print(json.dumps(out)); return 0
    out["empty"] = False

    thr = a.thresh_cm / 100.0
    pp, parea = sample_surface(pv, pt, a.num_points, a.seed_pred)
    if pp is None:
        fail("pred has zero total triangle area")
    gp, garea = sample_surface(gv, gt_, a.num_points, a.seed_gt)
    out.update(pred_area_m2=round(parea, 4), gt_area_m2=round(garea, 4))

    vis_args = None
    if a.cull_to_frustum or a.cull_pred:
        if not a.traj:
            fail("--cull-to-frustum needs --traj")
        intr = dict(zip(["w", "h", "fx", "fy", "cx", "cy", "depth_scale"], map(float, a.intr.split(","))))
        traj = load_traj(a.traj)
        fids = parse_frames(a.frames, len(traj))
        if not fids:
            fail("--frames selects no frames")
        depth_dir = a.depth_dir or os.path.join(os.path.dirname(os.path.abspath(a.traj)), "results")
        tcv = c2w_opencv(traj, a.cam_convention)
        vis_args = (tcv, fids, depth_dir, intr, a.cull_dmin, a.cull_dmax, a.cull_margin)
        out.update(cull=dict(frames=a.frames, n_frames=len(fids), convention=a.cam_convention, dmin=a.cull_dmin,
                             dmax=a.cull_dmax, margin_m=a.cull_margin, depth_dir=depth_dir, traj=os.path.abspath(a.traj),
                             intr=intr))

    out["gt_points_total"] = int(len(gp)); out["pred_points_total"] = int(len(pp))
    if a.cull_to_frustum:
        key = hashlib.sha256(json.dumps([sha256_file(a.gt), a.frames, a.seed_gt, a.num_points, a.cull_dmin, a.cull_dmax,
                                         a.cull_margin, a.cam_convention, a.intr, sha256_file(a.traj)]).encode()).hexdigest()[:16]
        os.makedirs(a.cache_dir, exist_ok=True)
        cp = os.path.join(a.cache_dir, f"gtmask_{os.path.basename(a.gt)}_{key}.npz")
        if os.path.isfile(cp):
            gmask = np.load(cp)["mask"]; out["gt_mask_cache"] = "hit"
        else:
            log("computing GT visibility mask ...")
            gmask = visible_mask(gp, *vis_args); np.savez_compressed(cp, mask=gmask); out["gt_mask_cache"] = "miss"
        out["gt_mask_file"] = cp
        gp = gp[gmask]
        if len(gp) == 0:
            fail("GT cull removed every point")
    if a.cull_pred:
        pp = pp[visible_mask(pp, *vis_args)]
    out["gt_points_kept"] = int(len(gp)); out["pred_points_kept"] = int(len(pp))
    out["gt_kept_frac"] = round(len(gp) / out["gt_points_total"], 4)
    if len(pp) == 0:
        fail("pred cull removed every point")

    d_pg = nn_dist(pp, gp, a.workers)             # accuracy direction
    d_gp = nn_dist(gp, pp, a.workers)             # completeness direction
    acc, prec = stats(d_pg, thr); comp, cr = stats(d_gp, thr)
    out.update(accuracy_cm=round(acc * 100, 4), completeness_cm=round(comp * 100, 4),
               completion_ratio_pct=round(cr, 3), precision_pct=round(prec, 3),
               f_score_pct=round(2 * cr * prec / (cr + prec), 3) if cr + prec > 0 else 0.0)
    s_pg = surf_dist(gm, pp)                      # pred sample -> GT surface
    s_gp = surf_dist(pm, gp)                      # GT sample   -> pred surface
    sa, sp = stats(s_pg, thr); sc_, scr = stats(s_gp, thr)
    out.update(accuracy_surf_cm=round(sa * 100, 4), completeness_surf_cm=round(sc_ * 100, 4),
               completion_ratio_surf_pct=round(scr, 3), precision_surf_pct=round(sp, 3),
               accuracy_median_cm=round(float(np.median(d_pg)) * 100, 4),
               completeness_median_cm=round(float(np.median(d_gp)) * 100, 4),
               wall_s=round(time.time() - t0, 2))
    print(json.dumps(out)); return 0


if __name__ == "__main__":
    sys.exit(main())
