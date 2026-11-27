# Replica <scene> -> InfiniTAM CLI inputs: rgb %04i.ppm, depth %04i.pgm (16-bit P5, big-endian), calib.txt, poses/%04i.txt.
# Usage: replica_to_infinitam.py <N> [scene=room0]
# InfiniTAM reads depth as SIGNED short, so raw uint16 at 6553.5/m would wrap beyond 5 m.
# We store raw = round(depth_m * 3276.75) (max 10 m fits int16) and set calib affine a = 1/3276.75 (0.305 mm step).
# Poses: traj.txt line i (c2w, row-major, OpenCV axes, used AS-IS) -> poses/%04i.txt, 16 numbers in the same row-major
# order; ITMFileBasedTracker reads m00 m10 m20 m30 ... = row 0 first (ORUtils mXY = column X, row Y) and calls SetInvM (c2w).
import sys, os, numpy as np, cv2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SCENE = sys.argv[2] if len(sys.argv) > 2 else "room0"
R = f"/mnt/data/datasets/replica/Replica/{SCENE}"
O = f"/mnt/data/datasets/replica/infinitam/{SCENE}"
os.makedirs(O + "/poses", exist_ok=True)
SC = 3276.75
T = np.loadtxt(f"{R}/traj.txt").reshape(-1, 4, 4)
assert len(T) >= N, (len(T), N)
for i in range(N):
    d = cv2.imread(f"{R}/results/depth{i:06d}.png", cv2.IMREAD_UNCHANGED).astype(np.float64) / 6553.5
    raw = np.clip(np.round(d * SC), 0, 32767).astype(">u2")
    with open(f"{O}/{i:04d}.pgm", "wb") as f:
        f.write(b"P5\n%d %d\n65535\n" % (raw.shape[1], raw.shape[0])); f.write(raw.tobytes())
    c = cv2.imread(f"{R}/results/frame{i:06d}.jpg")[:, :, ::-1]
    with open(f"{O}/{i:04d}.ppm", "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (c.shape[1], c.shape[0])); f.write(np.ascontiguousarray(c).tobytes())
    with open(f"{O}/poses/{i:04d}.txt", "w") as f:
        f.write(" ".join("%.17g" % v for v in T[i].reshape(-1)) + "\n")
with open(f"{O}/calib.txt", "w") as f:
    f.write("1200 680\n600 600\n599.5 339.5\n\n1200 680\n600 600\n599.5 339.5\n\n"
            "1 0 0 0\n0 1 0 0\n0 0 1 0\n\naffine %.10g 0\n" % (1.0 / SC))
print("wrote", N, "frames to", O)
