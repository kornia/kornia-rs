#!/bin/bash
# List heavy jobs by process NAME (comm), so shells whose command line merely mentions a pattern never match.
# Heavy = compilers/builders, nvblox fuser, nsys, the Open3D runner (comm o3d_bench), python running a scorer/benchmark/nvblox script.
ps -eo pid=,comm=,args= | awk -v me=$$ '
  $1 == me { next }
  $2 ~ /^(cargo|cc1plus|cc1|rustc|nvcc|ninja|make|fuse_replica|nsys|o3d_bench|vdb_bench|cicc|ptxas)$/ { print; next }
  $2 ~ /^python/ && $0 ~ /(score_mesh|validate_scorer|bench|nvblox_|o3d_vbg|o3d_side)/ && $0 !~ /mem_sampler/ { print }'
