#!/bin/bash
# Record clock + load state (no sudo).
G=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
echo "date: $(date --iso-8601=ns)"
echo "nvpmodel: $(nvpmodel -q 2>&1 | tr '\n' ' ')"
echo "gpu_freq cur/min/max: $(cat $G/cur_freq) $(cat $G/min_freq) $(cat $G/max_freq) gov=$(cat $G/governor 2>/dev/null)"
echo "cpu_freq: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tr '\n' ' ')"
echo "emc: $(cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || echo n/a-no-root)"
echo "tegrastats: $(timeout 2 tegrastats --interval 500 2>/dev/null | head -1)"
echo "thermal: $(for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/temp 2>/dev/null) && echo -n "$(cat $z/type)=$t "; done)"
echo "loadavg: $(cat /proc/loadavg)"
echo "free_m: $(free -m | sed -n 2p)"
echo "heavy_procs: $(pgrep -a -f 'cargo|cc1plus|rustc|nvcc|ninja|fuse_replica|nsys|score_mesh|bench|open3d|o3d' | grep -v -e clock_state -e mem_sampler -e pgrep | tr '\n' ';')"
