"""Box-wide device allocation ceiling probe via ctypes -> libcudart (same cudaMallocAsync/cudaMalloc nvblox uses).
Allocates CHUNK_MB pieces until failure or LIMIT_MB; reports what one process can hold. Not timed; run alone."""
import ctypes, json, sys
rt = ctypes.CDLL("/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12")
fn = sys.argv[1]; chunk = int(sys.argv[2]) << 20; limit = int(sys.argv[3]) << 20
free_b, tot_b = ctypes.c_size_t(), ctypes.c_size_t()
rt.cudaFree(ctypes.c_void_p(0)); rt.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(tot_b))
out = dict(api=fn, chunk_mb=chunk >> 20, memgetinfo_free_mb_start=free_b.value >> 20, memgetinfo_total_mb=tot_b.value >> 20, steps=[])
ptrs = []; held = 0
while held < limit:
    p = ctypes.c_void_p()
    if fn == "async":
        err = rt.cudaMallocAsync(ctypes.byref(p), ctypes.c_size_t(chunk), ctypes.c_void_p(0)); err2 = rt.cudaStreamSynchronize(ctypes.c_void_p(0))
    else:
        err = rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(chunk)); err2 = 0
    if err != 0 or err2 != 0:
        out["first_failure"] = dict(held_mb=held >> 20, err=err, err_sync=err2); break
    # touch it so pages are really committed
    rt.cudaMemset(p, ctypes.c_int(1), ctypes.c_size_t(chunk)); rt.cudaDeviceSynchronize()
    ptrs.append(p); held += chunk
    rt.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(tot_b)); out["steps"].append((held >> 20, free_b.value >> 20))
out["max_held_mb"] = held >> 20
print(json.dumps(out))
