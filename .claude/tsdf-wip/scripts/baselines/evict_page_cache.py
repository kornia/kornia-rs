"""Drop clean page-cache pages of the given dirs' files via posix_fadvise(DONTNEED) (no root needed).
Used before nvblox 2 cm runs: nvmap/cudaMallocAsync on this Orin fails with OOM while RAM sits in page cache."""
import os, sys
n = 0
for root in sys.argv[1:]:
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                fd = os.open(os.path.join(dp, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError:
                pass
print("fadvised", n, "files")
