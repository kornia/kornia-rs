"""Attribute nvblox device allocations (nsys --cuda-memory-usage sqlite) to the enclosing NVTX range; prints live MB
before the first frame, at peak, and the sum of allocations by enclosing range (CPU only)."""
import sqlite3, sys, collections, numpy as np
c = sqlite3.connect(sys.argv[1])
rng = c.execute("select e.start, e.end, coalesce(e.text,s.value) from NVTX_EVENTS e left join StringIds s on e.textId=s.id where e.eventType=60 and e.end is not null order by e.start").fetchall()
ev = c.execute("select start, bytes, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start").fetchall()
first = min(r[0] for r in rng if r[2] == "fuser/file_loading")
live = np.cumsum([b if op == 0 else -b for _, b, op in ev]); t = np.array([e[0] for e in ev])
i0 = np.searchsorted(t, first) - 1
print(f"live before first frame: {live[i0]/2**20:.1f} MB; peak {live.max()/2**20:.1f} MB; final {live[-1]/2**20:.1f} MB")
starts = np.array([r[0] for r in rng])
by = collections.Counter(); byn = collections.Counter()
for ts, b, op in ev:
    if op != 0 or b < 1 << 20: continue
    j = np.searchsorted(starts, ts) - 1; ctx = "pre-loop/no NVTX range"
    while j >= 0:
        if rng[j][0] <= ts <= rng[j][1]: ctx = rng[j][2]; break
        j -= 1
    if ts < first: ctx = "pre-loop (mapper/layer construction): " + ctx
    by[ctx] += b; byn[ctx] += 1
for k, v in by.most_common(12): print(f"{v/2**20:9.1f} MB in {byn[k]:5d} allocs >=1MB  ctx={k}")
# pre-loop allocations grouped by size (block memory pools pre-expand one block per allocation)
pre = collections.Counter(); pren = collections.Counter()
for ts, b, op in ev:
    if op == 0 and ts < first: pre[b] += b; pren[b] += 1
print("pre-loop allocs by size (top):")
for b, v in pre.most_common(8): print(f"  size {b:>10d} B x {pren[b]:5d} = {v/2**20:8.1f} MB")
