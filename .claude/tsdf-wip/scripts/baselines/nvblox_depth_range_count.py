"""Count Replica depth pixels outside the Row M range [0.1, 7.0) m (nvblox has no min-depth param)."""
import json, sys
import numpy as np
import open3d as o3d
root = "/mnt/data/datasets/replica/Replica"
out = {}
for scene in sys.argv[1:]:
    n = lt01 = ge7 = zero = valid = 0
    dmin, dmax = np.inf, 0.0
    for i in range(2000):
        d = np.asarray(o3d.io.read_image(f"{root}/{scene}/results/depth{i:06d}.png")).astype(np.float64) / 6553.5
        n += d.size; zero += int((d == 0).sum()); v = d[d > 0]; valid += v.size
        lt01 += int((v < 0.1).sum()); ge7 += int((v >= 7.0).sum())
        if v.size: dmin = min(dmin, v.min()); dmax = max(dmax, v.max())
    out[scene] = dict(pixels=n, zero=zero, valid=valid, valid_lt_0p1=lt01, valid_ge_7=ge7, dmin_m=dmin, dmax_m=dmax)
print(json.dumps(out))
