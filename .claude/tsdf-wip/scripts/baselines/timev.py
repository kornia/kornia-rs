"""Stand-in for /usr/bin/time -v (not installed on this box, no apt allowed).
Usage: timev.py OUT.json -- CMD...  Runs CMD, writes wall_s, rc and getrusage(RUSAGE_CHILDREN)
(ru_maxrss kB = peak RSS of the largest waited-for descendant, as GNU time reports)."""
import json, resource, subprocess, sys, time
out = sys.argv[1]; cmd = sys.argv[sys.argv.index("--") + 1:]
t0 = time.monotonic(); rc = subprocess.call(cmd); wall = time.monotonic() - t0
r = resource.getrusage(resource.RUSAGE_CHILDREN)
json.dump(dict(cmd=cmd, rc=rc, wall_s=wall, max_rss_kb=r.ru_maxrss, user_s=r.ru_utime, sys_s=r.ru_stime,
               minflt=r.ru_minflt, majflt=r.ru_majflt, nvcsw=r.ru_nvcsw, nivcsw=r.ru_nivcsw), open(out, "w"), indent=1)
sys.exit(rc)
