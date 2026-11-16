"""Print VBG method signatures (docstring first lines) for the installed Open3D."""
import open3d as o3d
V = o3d.t.geometry.VoxelBlockGrid
for m in ("__init__", "compute_unique_block_coordinates", "integrate", "extract_triangle_mesh", "ray_cast"):
    print(m, "::", getattr(V, m).__doc__.strip()[:1400].replace("\n", " | "))
hm = o3d.core.HashMap
print("hashmap methods:", [x for x in dir(hm) if not x.startswith("_")])
