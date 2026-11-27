# Learning from Open3D VBG / nvblox v0.0.10 / InfiniTAM — what to take, what to avoid
Read-only study 2026-11-15 (agent baseline-source-study, saved by main loop). Ours = kornia-rs feat/tsdf-submaps @ 6d4a793d. Nothing built/run; baseline numbers from results/baselines/.
PROVISIONAL: our CUDA path never timed (WF-2 pending) — ranking is structural; re-rank after our own nsys.

## 1. Where their per-frame time goes (room0, 1200x680, nsys sqlite)

### Open3D VBG CUDA, 5 cm, B=16, row M (runs/room0_v0.05_M_cuda_time_nsys_n2000/prof.sqlite; headline = plain run 2.82 ms e2e; nsys inflates API, mean 3.24 ms → proportions only)
| per frame | time | note |
|---|---|---|
| IntegrateCUDA | 391 us | 104 frustum blocks x 4096 voxels = 426k voxels; 27 regs; 20 B/voxel r+w each way |
| DepthTouchCUDA (+ compaction) | 183 + 14 us | 1/16 of pixels, 4 ray samples each |
| STDGPUInsert (frustum map + main map) | 148 + 14 us | |
| stdgpu clear/fill/iota/tabulate, find, copy | ~55 us | |
| **GPU kernels total** | **~0.80 ms** | |
| H2D colour 2.45 MB + depth 1.63 MB, pageable | 0.30 ms GPU, **0.58 ms API** | per-copy API 296/281 us vs GPU 185/115 us |
| small sync memcpys (Item() reads) | 13 x 39 us = 0.52 ms | |
| syncs | 7 cudaDeviceSynchronize + 11 cudaStreamSynchronize | |
| launches | ~12 x 29.5 us API | |
→ >half of Open3D's frame is host API/sync/pageable-copy, not kernels.

### nvblox, 5 cm, B=8, incremental colour mesh (runs/room0_v0.05_M_inc_cuda_n2000 + timing.txt)
- TSDF/frame: tsdf integrate 166 us; combinedBlockIndicesInImage 164 us; colour integrate 127 us; checkBlocksInTruncationBand 71 us; colour sphereTracing 647 us.
- Mesh/frame: meshBlocksCalculateTableIndices 5.7 ms; flatMarchingCubes 4.2 ms; weldVertices 1.7 ms (213 regs); CalculateVertices 0.6 ms.
- Host dominates: 61584 cudaStreamSynchronize = 31/frame, 15.5 ms/frame waiting; 36297 cudaMallocAsync = 18/frame; 34528 memsets = 17/frame; 4092 stream create/destroy.
- 2 cm (nomesh_cuda): tsdf/integrate 7.8 ms wall = allocate_blocks 2.6 + update_blocks 2.6 + transfer_blocks 1.3 + get_blocks_in_view 1.2.
- Colour at 2 cm 23.2 ms wall, reduce_to_blocks_in_band 12.7 ms wall vs 0.9 ms GPU kernel.

### InfiniTAM
No trace. ITMSceneReconstructionEngine_CUDA.tcu:130-190: 5 launches + sync cudaMemcpy per frame.

## 2. Ranked techniques

### T1. Input frames read in place on unified DRAM (zero-copy staging)
- Theirs: all three upload pageable; nvblox f32 depth doubles bytes.
- Ours: dev.htod from caller &[u16]/&[u8] (cuda.rs:866, 873, 883) — pageable = CPU bounce + DMA, effectively sync.
- Orin: one DRAM; DMA is DRAM→DRAM of memory the SMs could read directly.
- Change: workspace ring of K pinned frame buffers (cuMemHostAlloc DEVICEMAP or cuMemHostRegister once, cold) exposed as `&mut [u16]`/`&mut [u8]`; decoder writes in, kernels read device alias. Keep copy path for arbitrary slices. Optional double-buffer.
- Payoff: ~4 MB copy + bounce/frame removed (Open3D pays 0.58 ms API).
- Contract: safe (§14 placement may differ; §17 ok, registration cold).
- Verify: GPU caching of pinned host mem on Orin (I/O coherence); if poor, pinned + async copy still cuts API cost.

### T2. Touch: skip duplicate boxes on GPU (CPU already does)
- Theirs: Open3D stride-4 + 4 ray samples (VoxelBlockGrid.cpp:223-241, VoxelBlockGridCUDA.cu:106-180), lossy; InfiniTAM idempotent plain stores per ray segment (ITMSceneReconstructionEngine_Shared.h:184-268); nvblox plain stores to dense bitmap (view_calculator.cu:198-255). None does a contended atomic per (pixel, block).
- Ours: grid-stride over every pixel (tsdf.rs:292-303), loop over box (tsdf.rs:201-220), each block = hash_find + atomic swap on mark[pool] + first-touch list fetch_add (tsdf.rs:146-160). B=8, trunc 4 → box ALWAYS 2 blocks/axis: 8 probes + 8 swaps/pixel ≈ 6.5M/frame onto ~1k mark addresses. B=16 ≈ 3.4/pixel. CPU skip of same box (alloc.rs:135-139, §9 conforming) never fires on GPU (consecutive iterations gstride apart).
- Change: per-lane box, `shfl_up` its 6 i32, skip if == lane-1 (lane 0 vs previous round's lane 31); lanes = consecutive pixels of a row. Optional match_any on packed lo key (fix-queue item 7). Count valid/out_of_range per pixel as now.
- Payoff: >10x fewer probes/atomics expected; likely largest kernel-side item.
- Contract: safe (touched set §9, list order free §14, stats unchanged).

### T3. Dense frustum-AABB block bitmap (nvblox), optional after T2
- Change: AABB of step points on device (or frustum + range_max), byte-store 1 per box cell; compaction scans x-major = ascending key order (§5.3 lexicographic x,y,z), one probe per set cell. Hits → visible list ascending; misses already sorted → pool = len + exclusive-scan rank. Removes miss CAS set, rank_misses, miss memset. Compaction clears cells it reads.
- Caveat: bound AABB (2 cm B=8 7 m ≈ 90^3 = 730 KB bytes / 91 KB bits); LiDAR/oversized → hash path. Workspace buffer, cold growth.
- Contract: safe.

### T4. O(m) miss-set reset
- Ours: memset miss_keys over miss_slots*8 B each step (cuda.rs:984); miss_slots = hash_slots(stride) (cuda.rs:641) grows with map: 512 KB/step at ~30k blocks.
- Change: rank/assign writes EMPTY_KEY back to the m slots in miss_list. Full memset only after abort/resize.
- Contract: safe.

### T5. Cold-frame miss ranking
- Ours: rank_misses tiled O(m^2) (tsdf.rs:446-472); quadratic on first frames/spawns → p95/max (m≈65k ≈ 4e9 compares).
- Change: m > ~4-8k → radix sort path (§9 lists voxel/cuda.rs radix+unique as conforming). T3 removes it for pinhole.
- Contract: safe.

### T6. MC: emit only non-empty blocks
- Theirs: nvblox isBlockMeshableKernel; Open3D 27-neighbour precompute (VoxelBlockGrid.cpp:48) — ours has it in smem (tsdf.rs:1111-1115, 1217-1220).
- Ours: mc_emit over all n listed blocks (cuda.rs:1471-1478) even when nt==0; host already has host_nt (cuda.rs:1404-1405).
- Change: host compacts (pool, v_off, t_off) for nt>0, emit over that list. Byte-identical.
- Second step, measure first: (B+1)^3 smem tile (2.9 KB B=8, 19.6 KB B=16) instead of 8 global corner reads/cube.

### T7. Integrate occupancy (measure first)
- Theirs: nvblox 1 thread/voxel 8x8x8, 32 regs (projective_integrator_impl.cuh:40-48, 59-114); InfiniTAM same (ITMSceneReconstructionEngine_CUDA.tcu:333-364); Open3D 27 regs, int64 div/mod per voxel (VoxelBlockGridImpl.h:219-303).
- Ours: 66 regs (pre byval fix; now 36-40 per fix queue) → was 3 blocks/SM; Pass 12 x 64-bit ptrs; colour branch in-kernel; colour 3 byte loads/voxel (tsdf.rs:683-685).
- Change: monomorphise colour on/off; u32 offsets; optional RGBX u32 colour staging (same bits).
- Payoff: moderate; DRAM-bound, we move 10 B/voxel vs Open3D 20 (§5.6). Verify ncu occupancy + DRAM throughput.
- Contract: safe (§14).

### T8. Raycast
- Block cache: Open3D MiniVecCache (VoxelBlockGridImpl.h:557-576, 784-812), InfiniTAM IndexCache (ITMRepresentationAccess.h:22-43, 70-93). Queue item 7; exact (volume immutable during raycast).
- Range map: Open3D EstimateRange (VoxelBlockGridImpl.h:311-555) BREAKS §13 (march starts at depth_min, z sequential f32 accumulation). Safe A: conservative per-tile coverage → exact miss for pixels no block projects onto (z<=0 straddlers mark all). Safe B: skip lookups but replay `z += block_step` to first covered z (mind max_steps, fine_until).
- Outputs ~20 MB dtoh'd pageable (cuda.rs:1320-1325) → mapped pinned, or device-resident for GPU ICP.

### T9. Host launch overhead (low; measure first)
- Ours per step: 3 memsets (cuda.rs:982-984), touch, rank, assign, merge, integrate, 1 dtoh, 1 sync (cuda.rs:1153-1155). Already leaner than Open3D (18 syncs), nvblox (31). T4 removes a memset; counters/list_counts memsets foldable into prior kernel. CUDA graphs need per-frame param updates → unlikely to pay; skip unless nsys shows gaps.

## 3. What they do WORSE (must not copy)
**Open3D:** 1/16-pixel 4-sample touch (lossy, violates §9; why its frustum set of 104 blocks is smaller than ours — remember when comparing ms); silent clamp to frustum-map capacity (VoxelBlockGridCUDA.cu:200); per-op cudaDeviceSynchronize (VoxelBlockGridImpl.h:306, 544, 1118, 1379) + Item() reads = 18 syncs/frame; 20 B/voxel; int64 div/mod; atomic-counter output order (nondeterministic).
**nvblox:** CPU block hash, std::vector blocks-in-view, host allocation, per-frame pointer transfer (projective_integrator_impl.cuh:237-262) = 5 of 7.8 ms at 2 cm; 31 syncs + 18 cudaMallocAsync + 17 memsets/frame + stream create/destroy; colour sphere-trace + band re-read (projective_appearance_integrator.cu:378-410, 459) 12.7 ms wall at 2 cm; per-voxel MC scratch + per-frame re-reserve (mesh_integrator.cu:333-420, 639-643) = the 485 MB OOM; weldVertices 213 regs (we do no weld, §12.4); f32 depth upload.
**InfiniTAM:** racy non-atomic alloc writes, dropped+retried collisions (ITMSceneReconstructionEngine_Shared.h:257-263); atomicSub free-list pool ids (ITMSceneReconstructionEngine_CUDA.tcu:396-445) break §5.8; buildVisibleList O(table)/frame; stopMaxW skip (.tcu:350) changes bits vs our running average at cap (§8.2); fixed bucket + excess list can overflow.

## 4. What we already do better (keep)
1 sync/step; byval params; device-resident counts drive grid-stride; 10 B/voxel SoA; lookup-first alloc with deterministic ranks; colour fused into depth pass; MC conf in smem, 27 neighbours once per block; 0 allocs/frame (§17); CPU same-box skip (alloc.rs:135). CPU touch-par 6t 2 cm 11.6 ms vs nvblox GPU 2 cm 30.8 ms.

## 5. Next measurement (WF-2)
nsys ours room0 5 cm + 2 cm, B=8 + B=16, per kernel: touch time + atomics/frame; htod API vs GPU; integrate DRAM throughput (ncu); share of mc_emit blocks with nt==0. Then re-rank T1-T9.
