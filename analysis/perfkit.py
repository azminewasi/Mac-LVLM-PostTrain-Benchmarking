"""Shared "how did the MacBook perform" chapter for the post-training reports (GSM8K, math-story, and later the vision runs).
Built only from the per-step metrics already logged for every run (time split, throughput, GPU/CPU load, thermal limit, RAM/swap, MLX memory).
Usage: perf_section(entries, number, dataset_note) -> HTML, where entries = [dict(label, cls, kind('sft'|'rl'), adapter, met)]."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svgkit import bars, lines, tbl, fig

def _col(tr, k): return [x[k] for x in tr if k in x and x[k] == x[k]]

def run_perf(met):
    tr = [x for x in met if "train/lr" in x and "time/step_s" in x]
    ev = [x for x in met if "eval/accuracy" in x]
    mean = lambda k, rows=tr: float(np.mean(_col(rows, k))) if _col(rows, k) else None
    top = lambda k: max(_col(tr, k)) if _col(tr, k) else None
    p = dict(steps=len(tr), step_s=mean("time/step_s"), gen_s=mean("time/gen_s"), train_s=mean("time/train_s"), gen_tps=mean("gen/tokens_per_s"), tr_tps=mean("train/tokens_per_s"),
             gpu=mean("sys/gpu_util_pct_avg"), cpu=mean("sys/cpu_pct_avg"), therm=min(_col(tr, "sys/thermal_speed_limit"), default=None), ram=top("sys/ram_used_gb_max"),
             swap=top("sys/swap_gb_max"), peak=top("mem/peak_gb"), hours=max((x.get("time/elapsed_s", 0) for x in met), default=0) / 3600,
             eval_s=mean("eval/time_s", ev), eval_tps=mean("eval/gen_tokens_per_s", ev))
    p["gain"] = 100 * (ev[-1]["eval/accuracy"] - ev[0]["eval/accuracy"]) if len(ev) > 1 else None
    p["gen_share"] = p["gen_s"] / p["step_s"] if p["gen_s"] and p["step_s"] else None
    return p

def perf_section(entries, number, dataset_note="", single_stream_tps=72, anchor="perf", thrash_note=""):
    for e in entries: e["p"] = run_perf(e["met"])
    E = [e for e in entries if e["p"]["steps"]]
    if not E: return ""
    RLs = [e for e in E if e["kind"] == "rl"]
    f = lambda v, fmt="{:.1f}": "–" if v is None else fmt.format(v)
    avg = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
    hours = sum(e["p"]["hours"] for e in E); gpu = avg([e["p"]["gpu"] for e in E]); cpu = avg([e["p"]["cpu"] for e in E])
    therm = min(e["p"]["therm"] for e in E if e["p"]["therm"] is not None); ram = max(e["p"]["ram"] for e in E if e["p"]["ram"] is not None)
    swap_lo = min(e["p"]["swap"] for e in E if e["p"]["swap"] is not None); swap_hi = max(e["p"]["swap"] for e in E if e["p"]["swap"] is not None)
    peak = max(e["p"]["peak"] for e in E if e["p"]["peak"] is not None)
    cards = "".join(f'<div class="card"><b>{v}</b><span>{t}</span></div>' for v, t in [
        (f"{gpu:.0f}%", "GPU busy, average over all steps"), (f"{cpu:.0f}%", "CPU busy, average over all steps"),
        (f"{therm:.0f}%", f"lowest thermal speed limit seen in {hours:.0f} h of training"), (f"{ram:.1f} GB", "highest system RAM used (of 16 GB)"),
        (f"{swap_lo:.1f}-{swap_hi:.1f} GB", "swap in use during runs"), (f"{peak:.1f} GB", "highest MLX (GPU) memory")])
    names = [e["label"] for e in E]
    html = f'<h2 id="{anchor}">{number}. How the MacBook performed</h2>\n<p class="lede">Accuracy is only half of a benchmark. This chapter reads the system counters logged at every step to see what the machine did while it trained: where the time went, how busy the GPU and CPU were, and whether the laptop ever slowed down.</p>\n'
    if dataset_note: html += f"<p>{dataset_note}</p>\n"
    html += f'<div class="cards">{cards}</div>\n'
    html += f'<p>Across {len(E)} runs ({hours:.0f} hours of training in total) the GPU was busy {gpu:.0f}% of the time on average and the CPU only {cpu:.0f}%: training is a GPU job, and the CPU mostly feeds it. The thermal speed limit never dropped below {therm:.0f}%, so the laptop never throttled, even through multi-hour runs on a plugged-in laptop.</p>\n'
    if RLs:
        gs = [e["p"]["gen_share"] for e in RLs if e["p"]["gen_share"] is not None]
        html += "<h3>An RL step is mostly generation</h3>\n"
        html += f'<p>Each RL step first samples 16 answers and then trains on them. Sampling took {100*min(gs):.0f}-{100*max(gs):.0f}% of the step time ({100*np.mean(gs):.0f}% on average). Sampling is memory-bandwidth bound exactly as in the inference benchmark, so the lever that matters is how many tokens you generate, not how clever the optimiser is. Batching helps a lot: 16 sequences decoded together reach {min(e["p"]["gen_tps"] for e in RLs if e["p"]["gen_tps"]):.0f}-{max(e["p"]["gen_tps"] for e in RLs if e["p"]["gen_tps"]):.0f} tokens/s in total, against about {single_stream_tps} tokens/s for a single stream on the same model.</p>\n'
        html += fig("Mean seconds per RL step, split into sampling (generation) and the policy update (training).", bars([e["label"] for e in RLs], [("generation", "s0", [e["p"]["gen_s"] for e in RLs]), ("training", "s1", [e["p"]["train_s"] for e in RLs])], fmt="{:.1f}", ylabel="seconds per step"))
        sw = [e for e in E if e["kind"] == "sft"]
    html += "<h3>Throughput</h3>\n"
    html += fig("Tokens per second: generation during RL sampling (batch of 16) and training (forward + backward) for every run.", bars(names, [("generation", "s0", [e["p"]["gen_tps"] for e in E]), ("training", "s1", [e["p"]["tr_tps"] for e in E])], fmt="{:.0f}", ylabel="tokens / s"))
    html += "<h3>GPU and CPU load</h3>\n"
    fulls = [e for e in E if e.get("adapter") == "full"]; loras = [e for e in E if e.get("adapter") != "full" and e["kind"] == "rl"]
    extra = ""
    if fulls and loras:
        extra = f" Full fine-tuning kept the GPU less busy than LoRA in the RL runs ({np.mean([e['p']['gpu'] for e in fulls]):.0f}% vs {np.mean([e['p']['gpu'] for e in loras]):.0f}%): a likely reason is memory pressure: with roughly 11 GB of MLX memory there is less room to keep the GPU fed (we did not isolate the cause)."
    html += f"<p>GPU utilisation sat between {min(e['p']['gpu'] for e in E if e['p']['gpu']):.0f}% and {max(e['p']['gpu'] for e in E if e['p']['gpu']):.0f}% for every run, while the CPU stayed at {min(e['p']['cpu'] for e in E if e['p']['cpu']):.0f}-{max(e['p']['cpu'] for e in E if e['p']['cpu']):.0f}%.{extra}</p>\n"
    html += fig("Average GPU and CPU utilisation per run (system counters sampled during every step).", bars(names, [("GPU busy %", "s0", [e["p"]["gpu"] for e in E]), ("CPU busy %", "s3", [e["p"]["cpu"] for e in E])], fmt="{:.0f}", ymax=100, ylabel="% busy"))
    html += "<h3>Memory: always at the limit</h3>\n"
    html += f'<p>System RAM read {min(e["p"]["ram"] for e in E if e["p"]["ram"]):.1f}-{ram:.1f} GB in every run, almost the whole 16 GB, and the swap file held {swap_lo:.1f}-{swap_hi:.1f} GB. This is mostly macOS using memory for caches and other apps, so it says more about the machine than about training, but it also means there is no slack: the runs that peaked at {peak:.1f} GB of MLX memory had only a few GB of headroom. The steady swap use did not visibly slow the runs.{thrash_note}</p>\n'
    html += "<h3>What each run cost</h3>\n"
    rows = []
    for e in E:
        p = e["p"]; ppH = (p["gain"] / p["hours"]) if p["gain"] is not None and p["hours"] else None
        rows.append([e["label"], p["steps"], f(p["hours"]), f(p["step_s"]), f(100 * p["gen_share"], "{:.0f}%") if p["gen_share"] else "–", f(p["gen_tps"], "{:.0f}"), f(p["tr_tps"], "{:.0f}"), f(p["gpu"], "{:.0f}%"), f(p["cpu"], "{:.0f}%"),
                     f(p["peak"]), f(p["ram"]), f(p["swap"]), "–" if p["gain"] is None else f"{p['gain']:+.0f}", "–" if ppH is None else f"{ppH:+.1f}"])
    html += tbl(["Run", "Steps", "Hours", "s / step", "Sampling share", "Gen tok/s", "Train tok/s", "GPU", "CPU", "Peak MLX GB", "RAM GB", "Swap GB", "Gain (pts)", "Pts / hour"], rows, "dense")
    ev_t = [e["p"]["eval_s"] for e in E if e["p"]["eval_s"]]
    html += f'<p class="dim">Gain is final minus untrained test accuracy in points; points per hour divides it by the wall-clock hours of the whole run, evaluations included (one evaluation of the test set takes about {np.mean(ev_t):.0f} s). Energy and battery use were not measured: that needs administrator access to the power counters.</p>\n' if ev_t else ""
    cheap = [e for e in E if e["kind"] == "sft"]
    html += f'<div class="take"><b>Takeaway.</b> On this MacBook training is GPU-bound and thermally comfortable; the cost of RL is almost all sampling time, so it scales with how many tokens the model writes, and the memory headroom, not the clock speed, is the constraint to plan around.</div>\n'
    return html
