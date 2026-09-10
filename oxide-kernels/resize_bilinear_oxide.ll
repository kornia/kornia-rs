; ModuleID = 'builtin.module'
source_filename = "resize_bilinear_oxide"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.fptoui.sat.i32.f32(float)

define ptx_kernel void @resize_bilinear_oxide_3c(ptr %v0, ptr %v1, i32 %v2, i32 %v3, i32 %v4, i32 %v5, float %v6, float %v7, float %v8, float %v9) {
entry:
  br label %bb0
bb0:
  %v10 = phi ptr [ %v0, %entry ]
  %v11 = phi ptr [ %v1, %entry ]
  %v12 = phi i32 [ %v2, %entry ]
  %v13 = phi i32 [ %v3, %entry ]
  %v14 = phi i32 [ %v4, %entry ]
  %v15 = phi i32 [ %v5, %entry ]
  %v16 = phi float [ %v6, %entry ]
  %v17 = phi float [ %v7, %entry ]
  %v18 = phi float [ %v8, %entry ]
  %v19 = phi float [ %v9, %entry ]
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb1
bb1:
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v22 = mul i32 %v20, %v21
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb3
bb3:
  %v24 = add i32 %v22, %v23
  %v25 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v26 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  br label %bb5
bb5:
  %v27 = mul i32 %v25, %v26
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb6
bb6:
  %v29 = add i32 %v27, %v28
  %v30 = icmp uge i32 %v24, %v14
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb7, label %bb8
bb7:
  %v32 = icmp uge i32 %v29, %v15
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb9, label %bb8
bb8:
  br label %bb28
bb9:
  %v34 = sub i32 %v12, 1
  %v35 = uitofp i32 %v34 to float
  %v36 = sub i32 %v13, 1
  %v37 = uitofp i32 %v36 to float
  %v38 = uitofp i32 %v24 to float
  %v39 = fmul float %v16, %v38
  %v40 = fadd float %v39, %v17
  %v41 = uitofp i32 %v29 to float
  %v42 = fmul float %v18, %v41
  %v43 = fadd float %v42, %v19
  %v44 = fcmp olt float %v40, %v35
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb11, label %bb10
bb10:
  br label %bb12
bb11:
  br label %bb12
bb12:
  %v46 = phi float [ %v40, %bb10 ], [ %v35, %bb11 ]
  %v47 = fcmp olt float %v43, %v37
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb14, label %bb13
bb13:
  br label %bb15
bb14:
  br label %bb15
bb15:
  %v49 = phi float [ %v43, %bb13 ], [ %v37, %bb14 ]
  %v50 = fcmp ogt float %v46, 0.0
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb17, label %bb16
bb16:
  br label %bb18
bb17:
  br label %bb18
bb18:
  %v52 = phi float [ %v46, %bb16 ], [ 0.0, %bb17 ]
  %v53 = fcmp ogt float %v49, 0.0
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb20, label %bb19
bb19:
  br label %bb21
bb20:
  br label %bb21
bb21:
  %v55 = phi float [ %v49, %bb19 ], [ 0.0, %bb20 ]
  %v56 = call i32 @llvm.fptoui.sat.i32.f32(float %v52)
  %v57 = call i32 @llvm.fptoui.sat.i32.f32(float %v55)
  %v58 = add i32 %v56, 1
  %v59 = icmp ult i32 %v58, %v34
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb23, label %bb22
bb22:
  br label %bb24
bb23:
  br label %bb24
bb24:
  %v61 = phi i32 [ %v58, %bb22 ], [ %v34, %bb23 ]
  %v62 = add i32 %v57, 1
  %v63 = icmp ult i32 %v62, %v36
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb26, label %bb25
bb25:
  br label %bb27
bb26:
  br label %bb27
bb27:
  %v65 = phi i32 [ %v62, %bb25 ], [ %v36, %bb26 ]
  %v66 = uitofp i32 %v56 to float
  %v67 = fsub float %v52, %v66
  %v68 = uitofp i32 %v57 to float
  %v69 = fsub float %v55, %v68
  %v70 = fsub float 1.0, %v69
  %v71 = fsub float 1.0, %v67
  %v72 = fmul float %v70, %v71
  %v73 = fmul float %v70, %v67
  %v74 = fmul float %v69, %v71
  %v75 = fmul float %v69, %v67
  %v76 = mul i32 %v57, %v12
  %v77 = add i32 %v76, %v56
  %v78 = mul i32 %v77, 3
  %v79 = zext i32 %v78 to i64
  %v80 = add i32 %v76, %v61
  %v81 = mul i32 %v80, 3
  %v82 = zext i32 %v81 to i64
  %v83 = mul i32 %v65, %v12
  %v84 = add i32 %v83, %v56
  %v85 = mul i32 %v84, 3
  %v86 = zext i32 %v85 to i64
  %v87 = mul i32 %v65, %v12
  %v88 = add i32 %v87, %v61
  %v89 = mul i32 %v88, 3
  %v90 = zext i32 %v89 to i64
  %v91 = mul i32 %v29, %v14
  %v92 = add i32 %v91, %v24
  %v93 = mul i32 %v92, 3
  %v94 = zext i32 %v93 to i64
  %v95 = getelementptr inbounds float, ptr %v10, i64 %v79
  %v96 = load float, ptr %v95
  %v97 = getelementptr inbounds float, ptr %v10, i64 %v82
  %v98 = load float, ptr %v97
  %v99 = getelementptr inbounds float, ptr %v10, i64 %v86
  %v100 = load float, ptr %v99
  %v101 = getelementptr inbounds float, ptr %v10, i64 %v90
  %v102 = load float, ptr %v101
  %v103 = add i64 %v79, 1
  %v104 = getelementptr inbounds float, ptr %v10, i64 %v103
  %v105 = load float, ptr %v104
  %v106 = add i64 %v82, 1
  %v107 = getelementptr inbounds float, ptr %v10, i64 %v106
  %v108 = load float, ptr %v107
  %v109 = add i64 %v86, 1
  %v110 = getelementptr inbounds float, ptr %v10, i64 %v109
  %v111 = load float, ptr %v110
  %v112 = add i64 %v90, 1
  %v113 = getelementptr inbounds float, ptr %v10, i64 %v112
  %v114 = load float, ptr %v113
  %v115 = add i64 %v79, 2
  %v116 = getelementptr inbounds float, ptr %v10, i64 %v115
  %v117 = load float, ptr %v116
  %v118 = add i64 %v82, 2
  %v119 = getelementptr inbounds float, ptr %v10, i64 %v118
  %v120 = load float, ptr %v119
  %v121 = add i64 %v86, 2
  %v122 = getelementptr inbounds float, ptr %v10, i64 %v121
  %v123 = load float, ptr %v122
  %v124 = add i64 %v90, 2
  %v125 = getelementptr inbounds float, ptr %v10, i64 %v124
  %v126 = load float, ptr %v125
  %v127 = fmul float %v72, %v96
  %v128 = fmul float %v73, %v98
  %v129 = fadd float %v127, %v128
  %v130 = fmul float %v74, %v100
  %v131 = fadd float %v129, %v130
  %v132 = fmul float %v75, %v102
  %v133 = getelementptr inbounds float, ptr %v11, i64 %v94
  %v134 = fadd float %v131, %v132
  store float %v134, ptr %v133
  %v135 = fmul float %v72, %v105
  %v136 = fmul float %v73, %v108
  %v137 = fadd float %v135, %v136
  %v138 = fmul float %v74, %v111
  %v139 = fadd float %v137, %v138
  %v140 = fmul float %v75, %v114
  %v141 = add i64 %v94, 1
  %v142 = getelementptr inbounds float, ptr %v11, i64 %v141
  %v143 = fadd float %v139, %v140
  store float %v143, ptr %v142
  %v144 = fmul float %v72, %v117
  %v145 = fmul float %v73, %v120
  %v146 = fadd float %v144, %v145
  %v147 = fmul float %v74, %v123
  %v148 = fadd float %v146, %v147
  %v149 = fmul float %v75, %v126
  %v150 = add i64 %v94, 2
  %v151 = getelementptr inbounds float, ptr %v11, i64 %v150
  %v152 = fadd float %v148, %v149
  store float %v152, ptr %v151
  br label %bb28
bb28:
  ret void
}

define ptx_kernel void @resize_bilinear_oxide_slice_3c(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, float %v8, float %v9, float %v10, float %v11) {
entry:
  %v12 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v1, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v3, 1
  br label %bb0
bb0:
  %v16 = phi { ptr, i64 } [ %v13, %entry ]
  %v17 = phi { ptr, i64 } [ %v15, %entry ]
  %v18 = phi i32 [ %v4, %entry ]
  %v19 = phi i32 [ %v5, %entry ]
  %v20 = phi i32 [ %v6, %entry ]
  %v21 = phi i32 [ %v7, %entry ]
  %v22 = phi float [ %v8, %entry ]
  %v23 = phi float [ %v9, %entry ]
  %v24 = phi float [ %v10, %entry ]
  %v25 = phi float [ %v11, %entry ]
  %v26 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb1
bb1:
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb2:
  %v28 = mul i32 %v26, %v27
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb3
bb3:
  %v30 = add i32 %v28, %v29
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb4
bb4:
  %v32 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  br label %bb5
bb5:
  %v33 = mul i32 %v31, %v32
  %v34 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  br label %bb6
bb6:
  %v35 = add i32 %v33, %v34
  %v36 = icmp uge i32 %v30, %v20
  %v37 = xor i1 %v36, 1
  br i1 %v37, label %bb7, label %bb8
bb7:
  %v38 = icmp uge i32 %v35, %v21
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb9, label %bb8
bb8:
  br label %bb40
bb9:
  %v40 = sub i32 %v18, 1
  %v41 = uitofp i32 %v40 to float
  %v42 = sub i32 %v19, 1
  %v43 = uitofp i32 %v42 to float
  %v44 = uitofp i32 %v30 to float
  %v45 = fmul float %v22, %v44
  %v46 = fadd float %v45, %v23
  %v47 = uitofp i32 %v35 to float
  %v48 = fmul float %v24, %v47
  %v49 = fadd float %v48, %v25
  %v50 = fcmp olt float %v46, %v41
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb11, label %bb10
bb10:
  br label %bb12
bb11:
  br label %bb12
bb12:
  %v52 = phi float [ %v46, %bb10 ], [ %v41, %bb11 ]
  %v53 = fcmp olt float %v49, %v43
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb14, label %bb13
bb13:
  br label %bb15
bb14:
  br label %bb15
bb15:
  %v55 = phi float [ %v49, %bb13 ], [ %v43, %bb14 ]
  %v56 = fcmp ogt float %v52, 0.0
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb17, label %bb16
bb16:
  br label %bb18
bb17:
  br label %bb18
bb18:
  %v58 = phi float [ %v52, %bb16 ], [ 0.0, %bb17 ]
  %v59 = fcmp ogt float %v55, 0.0
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb20, label %bb19
bb19:
  br label %bb21
bb20:
  br label %bb21
bb21:
  %v61 = phi float [ %v55, %bb19 ], [ 0.0, %bb20 ]
  %v62 = call i32 @llvm.fptoui.sat.i32.f32(float %v58)
  %v63 = call i32 @llvm.fptoui.sat.i32.f32(float %v61)
  %v64 = add i32 %v62, 1
  %v65 = icmp ult i32 %v64, %v40
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb23, label %bb22
bb22:
  br label %bb24
bb23:
  br label %bb24
bb24:
  %v67 = phi i32 [ %v64, %bb22 ], [ %v40, %bb23 ]
  %v68 = add i32 %v63, 1
  %v69 = icmp ult i32 %v68, %v42
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb26, label %bb25
bb25:
  br label %bb27
bb26:
  br label %bb27
bb27:
  %v71 = phi i32 [ %v68, %bb25 ], [ %v42, %bb26 ]
  %v72 = uitofp i32 %v62 to float
  %v73 = fsub float %v58, %v72
  %v74 = uitofp i32 %v63 to float
  %v75 = fsub float %v61, %v74
  %v76 = fsub float 1.0, %v75
  %v77 = fsub float 1.0, %v73
  %v78 = fmul float %v76, %v77
  %v79 = fmul float %v76, %v73
  %v80 = fmul float %v75, %v77
  %v81 = fmul float %v75, %v73
  %v82 = mul i32 %v63, %v18
  %v83 = add i32 %v82, %v62
  %v84 = mul i32 %v83, 3
  %v85 = zext i32 %v84 to i64
  %v86 = add i32 %v82, %v67
  %v87 = mul i32 %v86, 3
  %v88 = zext i32 %v87 to i64
  %v89 = mul i32 %v71, %v18
  %v90 = add i32 %v89, %v62
  %v91 = mul i32 %v90, 3
  %v92 = zext i32 %v91 to i64
  %v93 = mul i32 %v71, %v18
  %v94 = add i32 %v93, %v67
  %v95 = mul i32 %v94, 3
  %v96 = zext i32 %v95 to i64
  %v97 = mul i32 %v35, %v20
  %v98 = add i32 %v97, %v30
  %v99 = mul i32 %v98, 3
  %v100 = zext i32 %v99 to i64
  %v101 = extractvalue { ptr, i64 } %v16, 1
  %v102 = icmp ult i64 %v85, %v101
  br i1 %v102, label %bb28, label %bb41
bb28:
  %v103 = extractvalue { ptr, i64 } %v16, 0
  %v104 = getelementptr inbounds float, ptr %v103, i64 %v85
  %v105 = load float, ptr %v104
  %v106 = fmul float %v78, %v105
  %v107 = icmp ult i64 %v88, %v101
  br i1 %v107, label %bb29, label %bb42
bb29:
  %v108 = extractvalue { ptr, i64 } %v16, 0
  %v109 = getelementptr inbounds float, ptr %v108, i64 %v88
  %v110 = load float, ptr %v109
  %v111 = fmul float %v79, %v110
  %v112 = fadd float %v106, %v111
  %v113 = icmp ult i64 %v92, %v101
  br i1 %v113, label %bb30, label %bb43
bb30:
  %v114 = extractvalue { ptr, i64 } %v16, 0
  %v115 = getelementptr inbounds float, ptr %v114, i64 %v92
  %v116 = load float, ptr %v115
  %v117 = fmul float %v80, %v116
  %v118 = fadd float %v112, %v117
  %v119 = icmp ult i64 %v96, %v101
  br i1 %v119, label %bb31, label %bb44
bb31:
  %v120 = extractvalue { ptr, i64 } %v16, 0
  %v121 = getelementptr inbounds float, ptr %v120, i64 %v96
  %v122 = load float, ptr %v121
  %v123 = fmul float %v81, %v122
  %v124 = fadd float %v118, %v123
  %v125 = add i64 %v85, 1
  %v126 = icmp ult i64 %v125, %v101
  br i1 %v126, label %bb32, label %bb45
bb32:
  %v127 = extractvalue { ptr, i64 } %v16, 0
  %v128 = getelementptr inbounds float, ptr %v127, i64 %v125
  %v129 = load float, ptr %v128
  %v130 = fmul float %v78, %v129
  %v131 = add i64 %v88, 1
  %v132 = icmp ult i64 %v131, %v101
  br i1 %v132, label %bb33, label %bb46
bb33:
  %v133 = extractvalue { ptr, i64 } %v16, 0
  %v134 = getelementptr inbounds float, ptr %v133, i64 %v131
  %v135 = load float, ptr %v134
  %v136 = fmul float %v79, %v135
  %v137 = fadd float %v130, %v136
  %v138 = add i64 %v92, 1
  %v139 = icmp ult i64 %v138, %v101
  br i1 %v139, label %bb34, label %bb47
bb34:
  %v140 = extractvalue { ptr, i64 } %v16, 0
  %v141 = getelementptr inbounds float, ptr %v140, i64 %v138
  %v142 = load float, ptr %v141
  %v143 = fmul float %v80, %v142
  %v144 = fadd float %v137, %v143
  %v145 = add i64 %v96, 1
  %v146 = icmp ult i64 %v145, %v101
  br i1 %v146, label %bb35, label %bb48
bb35:
  %v147 = extractvalue { ptr, i64 } %v16, 0
  %v148 = getelementptr inbounds float, ptr %v147, i64 %v145
  %v149 = load float, ptr %v148
  %v150 = fmul float %v81, %v149
  %v151 = fadd float %v144, %v150
  %v152 = add i64 %v85, 2
  %v153 = icmp ult i64 %v152, %v101
  br i1 %v153, label %bb36, label %bb49
bb36:
  %v154 = extractvalue { ptr, i64 } %v16, 0
  %v155 = getelementptr inbounds float, ptr %v154, i64 %v152
  %v156 = load float, ptr %v155
  %v157 = fmul float %v78, %v156
  %v158 = add i64 %v88, 2
  %v159 = icmp ult i64 %v158, %v101
  br i1 %v159, label %bb37, label %bb50
bb37:
  %v160 = extractvalue { ptr, i64 } %v16, 0
  %v161 = getelementptr inbounds float, ptr %v160, i64 %v158
  %v162 = load float, ptr %v161
  %v163 = fmul float %v79, %v162
  %v164 = fadd float %v157, %v163
  %v165 = add i64 %v92, 2
  %v166 = icmp ult i64 %v165, %v101
  br i1 %v166, label %bb38, label %bb51
bb38:
  %v167 = extractvalue { ptr, i64 } %v16, 0
  %v168 = getelementptr inbounds float, ptr %v167, i64 %v165
  %v169 = load float, ptr %v168
  %v170 = fmul float %v80, %v169
  %v171 = fadd float %v164, %v170
  %v172 = add i64 %v96, 2
  %v173 = icmp ult i64 %v172, %v101
  br i1 %v173, label %bb39, label %bb52
bb39:
  %v174 = extractvalue { ptr, i64 } %v16, 0
  %v175 = getelementptr inbounds float, ptr %v174, i64 %v172
  %v176 = load float, ptr %v175
  %v177 = fmul float %v81, %v176
  %v178 = fadd float %v171, %v177
  %v179 = extractvalue { ptr, i64 } %v17, 0
  %v180 = getelementptr inbounds float, ptr %v179, i64 %v100
  store float %v124, ptr %v180
  %v181 = add i64 %v100, 1
  %v182 = getelementptr inbounds float, ptr %v179, i64 %v181
  store float %v151, ptr %v182
  %v183 = add i64 %v100, 2
  %v184 = getelementptr inbounds float, ptr %v179, i64 %v183
  store float %v178, ptr %v184
  br label %bb40
bb40:
  ret void
bb41:
  unreachable
bb42:
  unreachable
bb43:
  unreachable
bb44:
  unreachable
bb45:
  unreachable
bb46:
  unreachable
bb47:
  unreachable
bb48:
  unreachable
bb49:
  unreachable
bb50:
  unreachable
bb51:
  unreachable
bb52:
  unreachable
}

