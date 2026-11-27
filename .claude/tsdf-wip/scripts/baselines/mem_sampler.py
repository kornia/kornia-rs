"""Sidecar memory sampler (not timed itself). Polls every 50 ms until the target process exits.
Columns: unix_t, MemTotal-MemAvailable (kB, system-wide; on Orin unified memory this includes
cudaMalloc/nvmap pages), proc VmRSS kB, proc VmHWM kB. Target = first process whose comm matches argv[2]."""
import os, sys, time
out, comm = sys.argv[1], sys.argv[2]
period = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1); d[k] = int(v.split()[0])
    return d["MemTotal"] - d["MemAvailable"]
def find():
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/comm") as f:
                    if f.read().strip() == comm: return p
            except OSError: pass
    return None
with open(out, "w") as fo:
    fo.write("t,sys_used_kb,vmrss_kb,vmhwm_kb\n")
    base_end = time.time() + 0.5
    while time.time() < base_end:  # pre-run baseline rows, pid = none
        fo.write(f"{time.time():.3f},{meminfo()},-1,-1\n"); time.sleep(period)
    pid, t0 = None, time.time()
    done = os.path.join(os.path.dirname(out), "done")
    while pid is None and time.time() - t0 < 120 and not os.path.exists(done):
        pid = find(); fo.write(f"{time.time():.3f},{meminfo()},-1,-1\n"); time.sleep(period)
    while pid is not None:
        try:
            rss = hwm = -1
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"): rss = int(line.split()[1])
                    elif line.startswith("VmHWM:"): hwm = int(line.split()[1])
        except OSError:
            break
        fo.write(f"{time.time():.3f},{meminfo()},{rss},{hwm}\n"); fo.flush(); time.sleep(period)
    for _ in range(10):
        fo.write(f"{time.time():.3f},{meminfo()},-1,-1\n"); time.sleep(period)
