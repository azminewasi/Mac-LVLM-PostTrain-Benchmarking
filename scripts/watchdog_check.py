"""Feasibility check for a full-fine-tuning RL run. Usage: python scripts/watchdog_check.py <run_dir>
Exit 3 (and print the reason) if, after >= 6 optimizer steps, the median of the last 6 step times exceeds 60 s
(= more than ~10 h for 625 steps: swap thrash) or peak MLX memory exceeds 15.5 GB. Otherwise exit 0."""
import json, sys, statistics as st
rows = []
try:
    rows = [json.loads(l) for l in open(f"{sys.argv[1]}/metrics.jsonl")]
except Exception: sys.exit(0)
tr = [x for x in rows if "time/step_s" in x and "reward/mean" in x]
if len(tr) >= 6:
    med = st.median(x["time/step_s"] for x in tr[-6:]); peak = max(x.get("mem/peak_gb", 0) for x in tr); swap = max(x.get("sys/swap_gb_max", 0) for x in tr)
    if med > 60 or peak > 15.5:
        print(f"INFEASIBLE: median step {med:.0f}s (last 6), peak MLX {peak:.1f} GB, swap {swap:.1f} GB after {len(tr)} steps"); sys.exit(3)
sys.exit(0)
