import json
G = "/mnt/data/kr-tsdf-spikes/results/baselines/nvblox"
inc_room = ("not run: nvblox v0.0.10 incremental colour mesh (--mesh_frame_subsampling=1, the fuser default) dies with 'CUDA error = 2 ... cudaMallocAsync ... out of memory' "
            "at frame 4 in BOTH rows: MeshIntegrator<ColorVoxel>::meshBlocksGPU -> unified_vector<PerVoxelMarchingCubesResults>::reserveAsync asks for 484,847,616 B "
            "(about 7k blocks in view x 512 voxels x ~136 B marching-cubes scratch, re-allocated every frame) while ~1.43 GB is already live and ~3 GB RAM is free. "
            f"Evidence: {G}/gate/oom_gdb/gdb.log (gdb backtrace + request sizes), {G}/runs/room0_v0.02_M_inc_cuda_n100 (nsys trace of the failure), "
            f"{G}/runs/room0_v0.02_M_inc_plain_n100 (repeated after evicting page cache, free 2.2->3.3 GB: same OOM), {G}/runs/room0_v0.02_R_inc_plain_n100 (Row R: same OOM at frame 4).")
full_room = ("not run: after all 2000 frames integrate fine, the single post-loop full Mapper::updateColorMesh over all blocks dies with the same cudaMallocAsync OOM "
             f"in the marching-cubes scratch (run.log: 'Ran out of data at frame: 1999' -> 'Generating the mesh.' -> CUDA error 2). Logs: {G}/runs/room0_v0.02_M_full_nvtx_n2000/run.log, "
             f"{G}/runs/room0_v0.02_R_full_nvtx_n2000/run.log. Block/voxel counts and memory therefore come from the no-mesh runs (nomesh_plain / nomesh_cuda).")
noscore = "not run: nvblox produced no mesh for this config (incremental and full meshing both OOM, see mesh_incremental / full_mesh)"
inc_off = ("not run to 2000 frames: the incremental-mesh run dies with cudaMallocAsync OOM after 1256 frames (Row M and Row R alike) in the same marching-cubes scratch path "
           "(MeshIntegrator<ColorVoxel>::meshBlocksGPU). Partial stats over its completed frames are in mesh_incremental_partial. Logs: "
           f"{G}/runs/office0_v0.02_M_inc_nvtx_n2000/run.log, {G}/runs/office0_v0.02_R_inc_nvtx_n2000/run.log")
notes = {}
for r in "MR":
    notes[f"room0_v0.02_{r}"] = dict(mesh_incremental=inc_room, full_mesh=full_room, score_full_mesh=noscore, score_incremental_mesh=noscore)
    notes[f"office0_v0.02_{r}"] = dict(mesh_incremental=inc_off, score_incremental_mesh="not run: the incremental-mesh run did not reach frame 2000 (OOM at frame 1256); the 2000-frame full mesh is scored instead (score_full_mesh)")
json.dump(notes, open(f"{G}/not_run_notes.json", "w"), indent=1)
