"""Read per-grid stats metadata (file_voxel_count = active voxels, file_mem_bytes = in-memory tree bytes) that OpenVDB
writes into a .vdb on save. Parses the metadata records: int32-len name, int32-len type name, int32-len value."""
import sys, struct, json
b = open(sys.argv[1], "rb").read()
out = []
for key in (b"file_voxel_count", b"file_mem_bytes"):
    pos = 0
    while True:
        k = b.find(struct.pack("<I", len(key)) + key, pos)
        if k < 0: break
        p = k + 4 + len(key)
        (tl,) = struct.unpack_from("<I", b, p); tn = b[p + 4:p + 4 + tl].decode(); p += 4 + tl
        (vl,) = struct.unpack_from("<I", b, p); p += 4
        val = struct.unpack_from("<q", b, p)[0] if tn == "int64" and vl == 8 else None
        out.append(dict(key=key.decode(), type=tn, value=val, offset=k)); pos = k + 1
names = [n for n in (b"D(x): signed distance grid", b"W(x): weights grid") if n in b]
print(json.dumps(dict(file=sys.argv[1], grid_names_in_file=[n.decode() for n in names], records=out)))
