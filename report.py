"""Regenerates the HTML reports from the local run folders: results/report.html (GSM8K) and results/report_mathstory.html (math-story-problems)."""
import runpy
runpy.run_path("analysis/make_report.py", run_name="__main__")
runpy.run_path("analysis/make_report_mathstory.py", run_name="__main__")
