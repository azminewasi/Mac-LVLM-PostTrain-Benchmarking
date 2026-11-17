"""Regenerates results/report.html (blog-style report) from runs/. Kept as the entry point the queue calls at the end."""
import runpy; runpy.run_path("analysis/make_report.py", run_name="__main__")
