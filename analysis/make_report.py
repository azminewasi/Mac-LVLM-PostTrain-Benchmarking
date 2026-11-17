"""Builds results/report.html from the local run folders (runs/<model>__<method>__<adapter>/): self-contained HTML with
inline-SVG charts, no matplotlib / W&B needed, and it works on partial results (runs still in progress are shown as such).
Usage (from the project root):  python analysis/make_report.py     (report.py is a thin wrapper)"""
import json, os, glob, html, datetime
import numpy as np

RUNS = "runs"; OUT = "results"
METHODS = ["sft", "grpo", "rloo", "gspo"]
MCOL = {"sft": "s4", "grpo": "s0", "rloo": "s1", "gspo": "s2"}
NAMES = {"sft": "SFT", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
WEIGHT_GB = {"0.8B": 1.59, "2B": 4.12, "4B": 8.46}      # bf16 weights as loaded by MLX (measured in the inference project)
PARAMS_B = {"0.8B": 0.76, "2B": 2.0, "4B": 4.2}
esc = lambda s: html.escape(str(s))

def read_jsonl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
runs = {}
for d in sorted(glob.glob(f"{RUNS}/*__*__*")):
    if not os.path.exists(f"{d}/metrics.jsonl") or not os.path.exists(f"{d}/config.json"): continue
    name = os.path.basename(d); model, method, adapter = name.split("__")
    cfg = json.load(open(f"{d}/config.json")); met = read_jsonl(f"{d}/metrics.jsonl")
    ev = [(x["epoch"], x["eval/accuracy"], x.get("eval/truncated_frac")) for x in met if "eval/accuracy" in x]
    tr = [x for x in met if "time/step_s" in x and "train/lr" in x or "reward/mean" in x]
    runs[name] = dict(name=name, model=model, method=method, adapter=adapter, cfg=cfg, met=met, ev=ev, tr=tr, done=os.path.exists(f"{d}/DONE"))

def col(r, k): return [x[k] for x in r["tr"] if k in x and x[k] == x[k]]
def smooth(v, w=25): return [float(np.mean(v[max(0, i - w + 1): i + 1])) for i in range(len(v))]
def last_step(r): return max((x.get("global_step", 0) for x in r["met"]), default=0)

# ------------------------------------------------------------------ svg helpers
def nice(mx, n=4):
    c = [a * 10.0 ** e for e in range(-3, 6) for a in (1, 2, 2.5, 5)]
    return min(c, key=lambda s: abs(mx / s - n))
def lines(series, xlabel="", ylabel="", ymax=None, fmt="{:g}", height=300, xmax=None):
    """series: [(name, cls, dashed, [(x, y)])]"""
    W, H, L, Rr, T, B = 760, height, 52, 12, 14, 40
    pts = [p for _, _, _, ps in series for p in ps]
    if not pts: return '<p class="dim">no data yet</p>'
    xm = xmax or max(p[0] for p in pts) or 1; ym = ymax or max(p[1] for p in pts) * 1.1 or 1
    st = nice(ym); ym = np.ceil(ym / st) * st
    X = lambda x: L + (W - L - Rr) * x / xm; Y = lambda y: T + (H - T - B) * (1 - y / ym)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for t in np.arange(0, ym + st / 2, st): o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L-6}" y="{Y(t)+3:.1f}" class="tick" text-anchor="end">{esc(fmt.format(t))}</text>')
    xs = nice(xm, 5)
    for t in np.arange(0, xm + xs / 2, xs): o.append(f'<text x="{X(t):.1f}" y="{H-B+15}" class="tick" text-anchor="middle">{t:g}</text>')
    for name, cls, dashed, ps in series:
        if not ps: continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f} {Y(y):.1f}" for i, (x, y) in enumerate(ps))
        dash = 'stroke-dasharray="6 4"' if dashed else ''
        o.append(f'<path d="{d}" class="ln {cls}" {dash}><title>{esc(name)}</title></path>')
        if len(ps) <= 12: o += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" class="dot {cls}"><title>{esc(name)}: {esc(fmt.format(y))} at {x:g}</title></circle>' for x, y in ps]
    o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{H-B}" y2="{H-B}" class="axis"/>')
    if xlabel: o.append(f'<text x="{(W+L)/2:.0f}" y="{H-6}" class="tick" text-anchor="middle">{esc(xlabel)}</text>')
    if ylabel: o.append(f'<text x="11" y="{(H-B)/2:.0f}" class="tick" text-anchor="middle" transform="rotate(-90 11 {(H-B)/2:.0f})">{esc(ylabel)}</text>')
    o.append("</svg>")
    leg = '<div class="legend">' + "".join(f'<span><i class="sw {c} {"dash" if d else ""}"></i>{esc(n)}</span>' for n, c, d, _ in series) + "</div>"
    return leg + "".join(o)
def bars(groups, series, fmt="{:.1f}", ymax=None, ylabel="", yref=None, height=290):
    """series: [(name, cls, [values])]"""
    W, H, L, Rr, T, B = 760, height, 52, 12, 16, 62
    vals = [v for _, _, vs in series for v in vs if v is not None]
    if not vals: return '<p class="dim">no data yet</p>'
    mx = ymax or max(vals) * 1.15; st = nice(mx); mx = np.ceil(mx / st) * st
    Y = lambda v: T + (H - T - B) * (1 - v / mx); pw = W - L - Rr; gw = pw / len(groups); bw = min(34, gw * .8 / len(series))
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for t in np.arange(0, mx + st / 2, st): o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L-6}" y="{Y(t)+3:.1f}" class="tick" text-anchor="end">{esc(fmt.format(t))}</text>')
    if yref: o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{Y(yref[0]):.1f}" y2="{Y(yref[0]):.1f}" class="ref"/><text x="{W-Rr-2}" y="{Y(yref[0])-4:.1f}" class="tick reflab" text-anchor="end">{esc(yref[1])}</text>')
    for gi, g in enumerate(groups):
        gx = L + gw * gi + gw / 2
        o.append(f'<text x="{gx:.1f}" y="{H-B+14}" class="glab" text-anchor="end" transform="rotate(-28 {gx:.1f} {H-B+14})">{esc(g)}</text>')
        for si, (n, c, vs) in enumerate(series):
            v = vs[gi]
            if v is None: continue
            x = gx - len(series) * bw / 2 + si * bw
            o.append(f'<rect x="{x+1:.1f}" y="{Y(v):.1f}" width="{bw-2:.1f}" height="{max(0, H-B-Y(v)):.1f}" class="{c}" rx="2"><title>{esc(n)} · {esc(g)}: {esc(fmt.format(v))}</title></rect><text x="{x+bw/2:.1f}" y="{Y(v)-4:.1f}" class="val" text-anchor="middle">{esc(fmt.format(v))}</text>')
    o.append(f'<line x1="{L}" x2="{W-Rr}" y1="{H-B}" y2="{H-B}" class="axis"/>')
    if ylabel: o.append(f'<text x="11" y="{(H-B)/2:.0f}" class="tick" text-anchor="middle" transform="rotate(-90 11 {(H-B)/2:.0f})">{esc(ylabel)}</text>')
    o.append("</svg>")
    return '<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{esc(n)}</span>' for n, c, _ in series) + "</div>" + "".join(o)
def tbl(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
def fig(cap, body): return f'<div class="fig"><div class="cap">{cap}</div>{body}</div>'

CSS = """
:root{--bg:#fbfbfa;--fg:#1d2024;--dim:#6b7078;--card:#fff;--line:#e3e4e0;--grid:#ececE8;--s0:#2f6fdc;--s1:#e07b00;--s2:#139b6c;--s3:#a23cc4;--s4:#7a7f87;--accent:#2f6fdc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--accent:#6aa3ff}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--accent:#6aa3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:21px;margin:44px 0 8px;border-top:1px solid var(--line);padding-top:26px}h3{font-size:16px;margin:26px 0 6px}
p{margin:8px 0}.sub{color:var(--dim);margin:0 0 18px}.dim{color:var(--dim)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px;letter-spacing:-.02em}.card span{color:var(--dim);font-size:13px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 6px;margin:12px 0}.fig .cap{color:var(--dim);font-size:13px;margin:2px 4px 6px}.note{border-left:3px solid var(--accent);background:var(--card);padding:8px 12px;margin:12px 0;border-radius:0 8px 8px 0;font-size:14px}
.chart{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--dim);stroke-width:1}.ref{stroke:var(--dim);stroke-width:1;stroke-dasharray:4 3}
.tick,.glab{fill:var(--dim);font-size:11px}.glab{font-size:11.5px;fill:var(--fg)}.val{fill:var(--fg);font-size:10px}.reflab{paint-order:stroke;stroke:var(--card);stroke-width:3px}
.s0{fill:var(--s0)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}path.ln{fill:none;stroke-width:2.2}path.ln.s0{stroke:var(--s0)}path.ln.s1{stroke:var(--s1)}path.ln.s2{stroke:var(--s2)}path.ln.s3{stroke:var(--s3)}path.ln.s4{stroke:var(--s4)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin:0 4px 6px}.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}.sw.dash{outline:1px dashed var(--fg);outline-offset:1px}
i.s0{background:var(--s0)}i.s1{background:var(--s1)}i.s2{background:var(--s2)}i.s3{background:var(--s3)}i.s4{background:var(--s4)}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card);border:1px solid var(--line)}th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--dim);font-weight:600;font-size:12px}table:not(.dense) td{white-space:normal}
ul{padding-left:20px}li{margin:4px 0}.lede{font-size:17px;line-height:1.55;margin:4px 0 12px}.kicker{color:var(--accent);font-weight:600;font-size:13px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 6px}.take{border:1px solid var(--line);border-left:3px solid var(--s2);background:var(--card);padding:10px 14px;border-radius:0 8px 8px 0;margin:16px 0}ul.toc li{margin:7px 0}.cmp{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}.cmp pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12px;white-space:pre-wrap;word-break:break-word;margin:4px 0}@media (max-width:700px){.cmp{grid-template-columns:1fr}}a{color:var(--accent)}ol li{margin:6px 0}code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:13px}@media (max-width:600px){h1{font-size:24px}}"""

# ------------------------------------------------------------------ numbers
order = sorted(runs.values(), key=lambda r: (["0.8B", "2B", "4B"].index(r["model"]), METHODS.index(r["method"]), r["adapter"]))
def status(r):
    tot = r["cfg"].get("total_steps", 0); st = last_step(r)
    return "done" if r["done"] else f"running ({st}/{tot})" if st else "started"
def stats(r):
    base = r["ev"][0][1] if r["ev"] else None; fin = r["ev"][-1][1] if len(r["ev"]) > 1 else None
    best = max((e[1] for e in r["ev"]), default=None)
    step_s = col(r, "time/step_s"); peak = col(r, "mem/peak_gb"); swap = col(r, "sys/swap_gb_max"); ram = col(r, "sys/ram_used_gb_max")
    return dict(base=base, fin=fin, best=best, step_s=float(np.mean(step_s)) if step_s else None, peak=max(peak) if peak else None, swap=max(swap) if swap else None,
                ram=max(ram) if ram else None, elapsed=max((x.get("time/elapsed_s", 0) for x in r["met"]), default=0) / 3600, toks=float(np.mean(col(r, "train/tokens_per_s"))) if col(r, "train/tokens_per_s") else None,
                gen=float(np.mean(col(r, "gen/tokens_per_s"))) if col(r, "gen/tokens_per_s") else None)
S = {r["name"]: stats(r) for r in order}
f = lambda v, fmt="{:.2f}": "–" if v is None else fmt.format(v)
pct = lambda v: "–" if v is None else f"{100*v:.0f}%"

rows = []
for r in order:
    s = S[r["name"]]; c = r["cfg"]
    rows.append([r["model"], NAMES[r["method"]], r["adapter"], status(r), pct(s["base"]), pct(s["fin"]), pct(s["best"]),
                 "–" if s["fin"] is None else f"{100*(s['fin']-s['base']):+.0f} pts", f"{c.get('trainable_params',0)/1e6:.1f}M", c.get("lr"), f(s["step_s"], "{:.1f}"), f(s["peak"], "{:.1f}"), f(s["swap"], "{:.1f}"), f(s["elapsed"], "{:.1f}")])
T_res = tbl(["Model", "Method", "Adapter", "Status", "Baseline acc", "Final acc", "Best acc", "Gain", "Trainable", "LR", "Step s", "Peak GB", "Swap GB", "Hours"], rows, "dense")

ndone = sum(r["done"] for r in order)
acc_charts = ""
for model in ["0.8B", "2B", "4B"]:
    ser = [(f"{NAMES[r['method']]} {r['adapter']}", MCOL[r["method"]], r["adapter"] == "full", [(e, a) for e, a, _ in r["ev"]]) for r in order if r["model"] == model and r["ev"]]
    if ser: acc_charts += fig(f"Qwen3.5-{model}: GSM8K test accuracy (first 100 test questions, greedy) after each epoch. Epoch 0 is the untrained model. Dashed = full fine-tuning.", lines(ser, "epoch", "accuracy", ymax=1.0, fmt="{:.0%}"))
rl = [r for r in order if r["method"] != "sft" and col(r, "reward/mean")]
rew_chart = lines([(f"{r['model']} {NAMES[r['method']]} {r['adapter']}", MCOL[r["method"]], r["adapter"] == "full", list(enumerate(smooth(col(r, "reward/mean")), 1))) for r in rl if r["model"] == "0.8B"], "optimizer step", "mean reward (25-step average)", ymax=1.0, fmt="{:.0%}") if rl else '<p class="dim">no RL runs yet</p>'
len_chart = lines([(f"{r['model']} {NAMES[r['method']]} {r['adapter']}", MCOL[r["method"]], r["adapter"] == "full", list(enumerate(smooth(col(r, "completion/len_mean")), 1))) for r in rl if r["model"] == "0.8B"], "optimizer step", "mean completion length (tokens, 25-step avg)") if rl else ""
sft_runs = [r for r in order if r["method"] == "sft" and col(r, "train/loss")]
sft_chart = lines([(f"{r['model']} SFT {r['adapter']}", "s4" if r["adapter"] == "lora" else "s3", r["adapter"] == "full", list(enumerate(smooth(col(r, "train/loss"), 10), 1))) for r in sft_runs], "optimizer step", "training loss (10-step avg)") if sft_runs else '<p class="dim">no SFT runs yet</p>'
names = [f"{r['model']} {NAMES[r['method']]} {r['adapter']}" for r in order]
c_step = bars(names, [("mean seconds per optimizer step", "s0", [S[r["name"]]["step_s"] for r in order])], fmt="{:.1f}", ylabel="s / step")
c_mem = bars(names, [("peak MLX (GPU) memory", "s0", [S[r["name"]]["peak"] for r in order]), ("max system RAM used", "s1", [S[r["name"]]["ram"] for r in order]), ("max swap", "s3", [S[r["name"]]["swap"] for r in order])], ymax=18, ylabel="GB", yref=(16, "16 GB unified memory"))
c_time = bars(names, [("hours so far", "s0", [S[r["name"]]["elapsed"] for r in order])], fmt="{:.1f}", ylabel="hours")

# ------------------------------------------------------------------ qualitative changes in the model's answers (from eval_epoch*.jsonl)
import re as _re
def style(resp):
    return dict(fmt=bool(_re.search(r"The answer is:\s*-?\$?\d", resp)), boxed="\\boxed" in resp, latex="$$" in resp or "\\text{" in resp, md=("**" in resp or "###" in resp))
def qual(r):
    out = []
    for f_ in sorted(glob.glob(f"{RUNS}/{r['name']}/eval_epoch*.jsonl"), key=lambda x: int(_re.search(r"epoch(\d+)", x).group(1))):
        e = int(_re.search(r"epoch(\d+)", f_).group(1)); recs = read_jsonl(f_)
        if not recs: continue
        st_ = [style(x["response"]) for x in recs]; n = len(recs)
        out.append(dict(epoch=e, acc=sum(x["correct"] for x in recs) / n, trunc=sum(x["truncated"] for x in recs) / n, toks=sum(x["gen_tokens"] for x in recs) / n,
                        fmt=sum(x["fmt"] for x in st_) / n, boxed=sum(x["boxed"] for x in st_) / n, latex=sum(x["latex"] for x in st_) / n, md=sum(x["md"] for x in st_) / n, recs=recs))
    return out
QUAL = {r["name"]: qual(r) for r in order}
q_rows = []
for r in order:
    for q in QUAL[r["name"]]:
        q_rows.append([r["model"], NAMES[r["method"]], r["adapter"], q["epoch"], pct(q["acc"]), pct(q["trunc"]), f"{q['toks']:.0f}", pct(q["fmt"]), pct(q["boxed"]), pct(q["md"]), pct(q["latex"])])
T_qual = tbl(["Model", "Method", "Adapter", "Epoch", "Accuracy", "Truncated", "Mean tokens", "\"The answer is:\"", "\\boxed{}", "Markdown bold/headers", "LaTeX"], q_rows, "dense")
q_fmt_chart = lines([(f"{r['model']} {NAMES[r['method']]} {r['adapter']}", MCOL[r["method"]], r["adapter"] == "full", [(q["epoch"], q["fmt"]) for q in QUAL[r["name"]]]) for r in order if QUAL[r["name"]]], "epoch", 'share of test answers ending with "The answer is: N"', ymax=1.0, fmt="{:.0%}")
q_tok_chart = lines([(f"{r['model']} {NAMES[r['method']]} {r['adapter']}", MCOL[r["method"]], r["adapter"] == "full", [(q["epoch"], q["toks"]) for q in QUAL[r["name"]]]) for r in order if QUAL[r["name"]]], "epoch", "mean generated tokens per test answer")
def tail(t, n=420): t = t.strip(); return ("…" if len(t) > n else "") + t[-n:]
def example_block(r):
    qs = QUAL[r["name"]]
    if len(qs) < 2: return ""
    a, b = qs[0], qs[-1]
    pick = next((i for i in range(len(a["recs"])) if i < len(b["recs"]) and style(a["recs"][i]["response"])["fmt"] and not style(b["recs"][i]["response"])["fmt"]), None)
    if pick is None: pick = 0
    ra, rb = a["recs"][pick], b["recs"][pick]
    mk = lambda lab, q, rec: f'<div><div class="dim">{lab} · answer {esc(rec["pred"])} (gold {esc(rec["gold"])}) · {"correct" if rec["correct"] else "wrong"} · {rec["gen_tokens"]} tokens</div><pre>{esc(tail(rec["response"]))}</pre></div>'
    return f'<h3>Same question, before and after: {r["model"]} {NAMES[r["method"]]} {r["adapter"]}</h3><p class="dim">Test question {pick}; the end of the answer at epoch 0 (untrained) and at epoch {b["epoch"]}.</p><div class="cmp">{mk("epoch 0", a, ra)}{mk(f"epoch {b["epoch"]}", b, rb)}</div>'
EXAMPLES = "".join(example_block(r) for r in order if len(QUAL[r["name"]]) > 1)

# capability estimates for training (clearly labelled)
BASE_GB = 6.5; USABLE = 16 - BASE_GB
def est_row(label, params_b, mode):
    if mode == "lora": gb = params_b * 2.0 + 0.5; note = "bf16 weights + LoRA + activations (measured activations dominate for small models)"
    elif mode == "qlora": gb = params_b * 0.61 + 0.5; note = "4-bit base + LoRA (--quantize-base 4)"
    else: gb = params_b * 14 + 0.5; note = "bf16 weights 2 + fp32 master 4 + Adam m,v 8 = 14 bytes/param + activations"
    st = "fits" if gb <= USABLE - 2.0 else ("tight (swap)" if gb <= USABLE + 1.5 else "does not fit")
    return [label, {"lora": "LoRA (bf16)", "qlora": "LoRA on 4-bit base", "full": "full fine-tune"}[mode], f"≈{gb:.1f}", st, esc(note)]
T_est = tbl(["Model", "Training mode", "State GB (est.)", f"On this Mac (usable ≈{USABLE:.1f} GB)", "How estimated"],
            [est_row("0.8B", 0.76, "lora"), est_row("0.8B", 0.76, "full"), est_row("2B", 2.0, "lora"), est_row("2B", 2.0, "full"), est_row("4B", 4.2, "lora"), est_row("4B", 4.2, "full"),
             est_row("9B", 9.0, "qlora"), est_row("9B", 9.0, "lora")])
meas = [[r["model"], NAMES[r["method"]], r["adapter"], f(S[r['name']]['peak'], "{:.1f}"), f(S[r['name']]['ram'], "{:.1f}"), f(S[r['name']]['swap'], "{:.1f}"), f(S[r['name']]['step_s'], "{:.1f}"),
         f(S[r['name']]['toks'], "{:.0f}"), f(S[r['name']]['gen'], "{:.0f}")] for r in order]
T_meas = tbl(["Model", "Method", "Adapter", "Peak MLX GB", "Max RAM GB", "Max swap GB", "Step s", "Train tok/s", "Rollout tok/s"], meas, "dense")

# findings that are safe to state from whatever has finished
finds = []
for model in ["0.8B", "2B", "4B"]:
    done = [r for r in order if r["model"] == model and r["done"] and S[r["name"]]["fin"] is not None]
    if done:
        b = max(done, key=lambda r: S[r["name"]]["fin"])
        finds.append(f"<b>{model}:</b> best finished run is {NAMES[b['method']]} {b['adapter']} at {pct(S[b['name']]['fin'])} (untrained baseline {pct(S[b['name']]['base'])}); "
                     + ", ".join(f"{NAMES[r['method']]} {r['adapter']} {pct(S[r['name']]['fin'])}" for r in done if r is not b) + ".")
grpo_peak = (f"{S['0.8B__grpo__lora']['peak']:.1f} GB" if "0.8B__grpo__lora" in S and S["0.8B__grpo__lora"]["peak"] else "about 3.5 GB")
mp = json.load(open(f"{OUT}/memory_profile.json")) if os.path.exists(f"{OUT}/memory_profile.json") else []
c_prof = bars([f"{'SFT-like' if x['seq_len'] == 400 else 'RL-like'}, micro-batch {x['batch']}" + (" + ckpt" if x["grad_checkpointing"] else "") for x in mp], [("peak MLX memory (GB)", "s0", [x["peak_gb"] for x in mp])], ymax=18, ylabel="GB", yref=(USABLE, "usable"), height=340) if mp else ""
sft = runs.get("0.8B__sft__lora"); sft_s = S.get("0.8B__sft__lora")
sft_txt = (f"On the 0.8B, LoRA SFT took accuracy from {pct(sft_s['base'])} to {pct(sft_s['fin'])} after five epochs (it was {pct(sft['ev'][1][1])} after the first and peaked at {pct(max(e[1] for e in sft['ev'][1:]))} in between), while the training loss fell to about 0.01. The model memorised the 500 reference solutions and lost the stronger, more verbose reasoning style it already had.") if sft and sft_s and sft["done"] else "The SFT result will appear here when the run finishes."
live = [r for r in order if not r["done"]]
live_txt = ("Currently running: " + ", ".join(f"{r['model']} {NAMES[r['method']]} {r['adapter']} (step {last_step(r)}/{r['cfg'].get('total_steps','?')})" for r in live) + ".") if live else ""
HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Post-training Qwen3.5 on a MacBook: what works, what breaks</title><style>{CSS}</style></head><body><main>
<p class="kicker">Mac LLM/VLM Post-training Testing</p>
<h1>Post-training Qwen3.5 on a 16 GB MacBook: what works, what breaks</h1>
<p class="sub">{datetime.date.today().strftime('%B %d, %Y')} · Apple M5 MacBook Pro, 16 GB · MLX · {ndone} of {len(order)} started runs finished · work in progress</p>

<h2 id="aim">The question</h2>
<p class="lede">Our companion post measured how far a 16 GB MacBook goes at <i>running</i> Qwen3.5. This one asks the harder question: how far does it go at <i>training</i> it?</p>
<p>Running a model streams weights through the chip once per token. Training does that and more: it keeps activations for the backward pass, gradients, optimizer state, and for reinforcement learning (RL) it also has to generate whole batches of answers before it can learn from them. On a laptop where macOS alone takes about {BASE_GB:.1f} GB of the 16, each of those costs matters.</p>
<p>We set out to compare four post-training methods on the same task: <b>SFT</b> (imitate reference solutions), and three critic-free RL methods, <b>GRPO</b>, <b>RLOO</b> and <b>GSPO</b> (learn from the model's own answers, rewarded when correct). Each is tried with LoRA adapters on the 0.8B, 2B and 4B models, and with full fine-tuning on the 0.8B. The task is GSM8K grade-school math: 500 training questions for 5 epochs, scored on 100 held-out test questions, with one line per (model, method, adapter) in Weights &amp; Biases and every rollout kept.</p>
<p>This is a work-in-progress report: it regenerates from whatever has finished. {live_txt} What it already contains is a good deal of what <i>broke</i> on the way, which turned out to be the most informative part.</p>

<h3>The short version so far</h3>
<ul class="toc">
<li><b>Memory is the first wall, and it comes much sooner than for inference.</b> A single 0.8B training step peaked at {max((x['peak_gb'] for x in mp), default=0):.1f} GB before we turned on gradient checkpointing, and {min((x['peak_gb'] for x in mp if x['grad_checkpointing']), default=0):.1f} GB after.</li>
<li><b>Training required re-implementing a layer.</b> Qwen3.5's fast attention kernel has no gradient; the library fallback was 50x too slow.</li>
<li><b>Three separate hyperparameter and reward mistakes each wrecked a run</b>, and each looked like "the method doesn't work" until we read the curves.</li>
<li><b>SFT on 500 human solutions makes this model <i>worse</i>.</b> {sft_txt.split('. ')[0] if sft and sft['done'] else 'Result pending.'}.</li>
</ul>

<h2 id="setup">1. The setup, in plain words</h2>
<ul>
<li><b>SFT</b>: show the model a reference solution and train it to reproduce it token by token.</li>
<li><b>GRPO</b>: sample 4 answers per question, reward the correct ones, and nudge the model toward answers that did better than their group average.</li>
<li><b>RLOO</b>: same sampling, but each answer is compared to the average of the <i>other</i> answers for that question (a leave-one-out baseline), with a plain policy gradient.</li>
<li><b>GSPO</b>: like GRPO, but the update is weighted at the whole-sequence level instead of per token.</li>
<li><b>LoRA</b> trains small adapter matrices (10.8M parameters on the 0.8B) on top of frozen weights; <b>full fine-tuning</b> updates every transformer weight and needs fp32 master weights and Adam state (about 14 bytes per parameter).</li>
</ul>
<p>All RL methods here are critic-free and have no reference-model KL penalty, so only one model sits in memory. Each RL step uses 4 prompts x 4 rollouts at temperature 1.0 and up to 640 new tokens. The learning rates are 2e-5 (SFT, LoRA), 1e-5 (SFT, full), 2e-6 (RL, LoRA) and 1e-6 (RL, full). The 0.8B runs use the full 500 samples x 5 epochs; the 2B and 4B use LoRA for 1 epoch to fit the time budget.</p>

<h2 id="broke">2. The first thing that broke: no gradients through Qwen3.5</h2>
<p class="lede">Qwen3.5 is not a plain transformer. Most of its layers are "gated delta" linear-attention layers, and MLX ships a fast Metal kernel for them that has no backward pass.</p>
<p>The first training step crashed with <code>Primitive::vjp not implemented for CustomKernel</code>. The library's pure-operations fallback works, but it loops over tokens in Python: <b>50 seconds</b> for one step on two short sequences, which would have made the whole project impossible. We wrote a chunk-parallel, matmul-based version of the recurrence (<code>pt/gdn.py</code>) that autograd can differentiate. It went through two bugs on the way (a numerically unstable matrix inverse that produced garbage on real, correlated keys, replaced by forward substitution) and now matches the reference: 4% relative error in the gradients on the real model, the same loss at step 1 (1.09 on both paths), and about <b>1 second</b> per SFT step.</p>
<div class="take"><b>Takeaway.</b> On Apple silicon, "can I fine-tune model X?" can depend on whether every layer type has a backward kernel. Check before you plan.</div>

<h2 id="sft">3. SFT: imitation made the model worse</h2>
<p class="lede">{sft_txt}</p>
<div class="fig"><div class="cap">SFT training loss. It falls to ~0.01: the model has memorised the 500 solutions.</div>{sft_chart}</div>
{acc_charts}
<p>Our first attempt, at a learning rate of 1e-4, was far worse: accuracy collapsed from 54% to 4% with degenerate repeated output. LoRA with scale 20 is very sensitive; 2e-5 trains stably but still lowers accuracy. A reasonable reading: Qwen3.5 already reasons well in its own style, and 500 terse human-written solutions over five epochs pull it toward a style it cannot execute as well. That is a finding, not a bug: it is exactly the situation RL is supposed to improve on, because RL trains on the model's own outputs.</p>

<h2 id="rl">4. RL: two ways to teach a model to stop thinking</h2>
<p class="lede">Both of our early GRPO failures were the same disease: the model found a way to get reward without being correct, or was pushed too hard by noise.</p>
<h3>Reward hacking</h3>
<p>The first GRPO run collapsed from 54% to 1% accuracy within the first epoch. The reason is in the reward. We capped generation at 400 tokens, and this model is verbose: 44% of rollouts were truncated and scored 0. A short, wrong answer in the right format earned a 0.1 format bonus, which beat 0. Within 50 steps the policy had learned that skipping the reasoning pays: completion length fell from 288 to about 35 tokens and every group looked the same. Fix: reward correctness only (the format flag is logged but not rewarded) and allow 640 tokens.</p>
<h3>Learning-rate drift</h3>
<p>The second attempt no longer collapsed in length, but reward and format rate drifted down steadily (0.42 to 0.1 and 0.89 to 0.39 over 25 steps) at a learning rate of 5e-6. With only 16 rollouts per step, and about half the groups carrying no signal, small noisy updates accumulate quickly under LoRA scale 20. At 2e-6 the run is stable. We verified the gradient code against the reference before blaming the hyperparameters.</p>
<div class="fig"><div class="cap">RL mean reward on the 0.8B (25-step moving average). A flat or rising line is what we want to see; both failed attempts fell.</div>{rew_chart}</div>
{fig("RL completion length: shorter completions can mean fewer truncations or can mean the policy abandoned its reasoning; read it together with the reward.", len_chart) if len_chart else ""}
<div class="take"><b>Takeaway.</b> In RL on a small model, read the reward and the completion-length curves in the first 50 steps. Reward hacking and drift show up there long before the accuracy number does. All discarded attempts are kept in W&amp;B (group <code>discarded</code>) and under <code>runs_discarded/</code>.</div>

<h2 id="style">5. What changed besides accuracy</h2>
<p class="lede">Accuracy is one number. The answers themselves change too: their format, length and style, and some of those changes are side effects nobody asked for.</p>
<p>The clearest example is the answer format. Our prompt ends with "End with a final line: 'The answer is: &lt;number&gt;'", and the untrained model follows it almost always. During RL we stopped rewarding that format (after it had been exploited, chapter 4). The result: in the sampled training rollouts the share ending with "The answer is: N" fell from about 0.8 to about 0.13 over the first two to three epochs, and on the greedy test answers it fell from 86% to 56% by epoch 2 (so far, in the GRPO run), while accuracy did not suffer, because the evaluator reads the last number in the answer. The model silently stopped following a formatting instruction that nothing rewarded any more. That is harmless for this benchmark and could be a real problem in a pipeline that parses the format. GRPO also gradually dropped LaTeX (80% of answers at epoch 0, 36% at epoch 2) while keeping its markdown headers.</p>
<p><b>SFT changes the style far more.</b> After SFT the model stops writing the long, markdown-heavy, LaTeX-heavy step-by-step answers it started with (95% and 80% of answers) and writes terse plain-text solutions like the GSM8K references: mean length falls from 347 to about 90 tokens and markdown and LaTeX disappear entirely. It follows the "The answer is:" format more reliably (99-100%), but gets fewer answers right, and in the first epochs it sometimes degenerates into a repeating arithmetic loop that runs to the token cap (see the example below). So the SFT story is as much about style as about accuracy: the model traded its own, stronger way of reasoning for a shorter one it executes worse.</p>
<div class="fig"><div class="cap">Share of the 100 test answers that end with the requested "The answer is: N" line, by epoch (epoch 0 = untrained).</div>{q_fmt_chart}</div>
<div class="fig"><div class="cap">Mean generated tokens per test answer, by epoch: how verbose the model is.</div>{q_tok_chart}</div>
{EXAMPLES}
<h3>Style statistics per run and epoch</h3>{T_qual}
<div class="take"><b>Takeaway.</b> Track style alongside accuracy: format-following, length, markdown and LaTeX use. A reward that ignores a behaviour lets that behaviour drift, and a benchmark score will not tell you.</div>

<h2 id="memory">6. Memory: the wall comes sooner</h2>
<p class="lede">A training step that peaked at almost 16 GB on a model with 1.6 GB of weights.</p>
<p>While the second GRPO attempt was running, system swap climbed to 5 GB and step times swung between 11 and 46 seconds. Profiling one forward and backward pass explained it: without gradient checkpointing, the activations of the 0.8B for a batch of two 800-token sequences needed almost the entire 16 GB. Turning on checkpointing (recomputing activations in the backward pass) cut that to {min((x['peak_gb'] for x in mp if x['grad_checkpointing'] and x['batch']==2), default=0):.1f} GB at the cost of some speed, and the restarted GRPO run peaks at {grpo_peak} so far.</p>
<div class="fig"><div class="cap">Peak MLX memory of one LoRA training step on the 0.8B (<code>tests/profile_memory.py</code>). Batch size and gradient checkpointing change it by 2-3x.</div>{c_prof}</div>
<p>This is the number to carry to the larger models, and it is not reassuring. Even with checkpointing, the 0.8B needs about 5 GB of working memory on top of its 1.6 GB of weights. The 4B has 8.5 GB of weights and bigger activations, so as configured it very likely does <b>not</b> fit in the ≈{USABLE:.1f} GB we have: it would need a micro-batch of 1, shorter sequences, or a more memory-efficient backward pass for the gated-delta layers. We will measure the 2B first and decide whether the 4B runs are feasible before spending days on them.</p>
{fig("Peak MLX memory, system RAM and swap per run so far, against the 16 GB limit.", c_mem)}

<h2 id="speed">7. What a run costs in time</h2>
<p class="lede">SFT is cheap; RL is generation-bound and slow.</p>
<p>An SFT step (4 sequences) takes about a second, so five epochs of 500 questions finish in roughly 15 minutes. An RL step must first generate 16 answers of up to 640 tokens at decode speeds of a few hundred tokens/s, then train on them: 15-45 seconds on the 0.8B. A 625-step RL run is therefore several hours, and the 2B and 4B are roughly 2.5x and 5x slower still, which is why those use one epoch.</p>
{fig("Mean seconds per optimizer step (SFT: 4 sequences; RL: 4 prompts x 4 rollouts).", c_step)}
{fig("Wall-clock hours used per run so far.", c_time)}

<h2 id="capability">8. So what can this MacBook train?</h2>
<p class="lede">Small models, comfortably, with LoRA; the full fine-tune of even the smallest model is a stretch.</p>
<ul>
<li><b>Budget:</b> macOS takes ≈{BASE_GB:.1f} GB, leaving ≈{USABLE:.1f} GB for weights, gradients, optimizer state, activations and (for RL) the generation cache.</li>
<li><b>LoRA is the practical route.</b> The adapter is ~11M parameters, so optimizer state is negligible; the real costs are the frozen weights and the activations (use gradient checkpointing and micro-batches of 1-2).</li>
<li><b>Full fine-tuning</b> needs about 14 bytes per parameter (bf16 weights, fp32 master copy, two fp32 Adam moments): ≈11 GB for the 0.8B, so it only fits the smallest model and leaves little room for activations. Larger models are out of reach for full fine-tuning.</li>
<li><b>RL is slow, not impossible.</b> The cost is generation time, which is set by memory bandwidth exactly as in the inference post.</li>
<li><b>Bigger models:</b> a 9B would need LoRA on a 4-bit base; we did not try it.</li>
</ul>
<h3>What fits (estimates)</h3>{T_est}
<p class="dim">Estimates are weights plus optimizer state and a 0.5 GB allowance; activations come on top and, as chapter 5 shows, can dominate. Treat "fits" as "should fit with small batches and checkpointing".</p>

<h2 id="conclusions">9. Conclusions so far</h2>
<ol>
<li><b>A 16 GB MacBook can post-train small Qwen3.5 models, but you must budget for activations, not just weights.</b> Gradient checkpointing changed a 15.8 GB step into a 6.9 GB one.</li>
<li><b>Check backward support layer by layer.</b> Qwen3.5's hybrid architecture needed a custom differentiable kernel before anything trained.</li>
<li><b>SFT on a small human-written set hurt a model that already reasons well.</b> More data or a gentler recipe would be needed for it to help.</li>
<li><b>RL needs a careful reward and a small learning rate.</b> Reward that can be earned without being correct will be found within 50 steps; watch reward and length together.</li>
<li><b>Unrewarded behaviour drifts.</b> After we stopped rewarding the "The answer is:" format, the share of answers using it fell from ~0.8 to ~0.13 without hurting the score. Track style, not just accuracy.</li>
<li><b>Time scales brutally with model size for RL.</b> Plan for hours per run on the 0.8B and days for the larger models.</li>
</ol>
<h3>What we did not measure, and caveats</h3>
<p>Test accuracy is on 100 questions (about ±10 points) with one seed per setup; learning rates were corrected after the failures above rather than swept, so the method comparison is not a tuned head-to-head; power draw and batteries are untested.</p>
<h3>What is next</h3>
<p>The queue continues: the remaining 0.8B setups (RLOO, GSPO, then the full fine-tuning variants), then SFT and the three RL methods with LoRA on the 2B and 4B at one epoch. This page will fill in the comparison between methods as they finish; the final write-up will update the conclusions above. The inference-side companion post covers running these same models.</p>

<h2 id="appendix">Appendix</h2>
<h3>A. Results table</h3>{T_res}
<h3>B. Resources per run</h3>{T_meas}
<h3>C. Method and data notes</h3>
<ul>
<li><b>Data:</b> 500 GSM8K train questions (seeded subset saved with each run); evaluation on the first 100 test questions, greedy, thinking off, 512-token cap.</li>
<li><b>Hyperparameters:</b> LoRA rank 16, scale 20, no dropout; gradient clipping 1.0; gradient checkpointing on for all runs except the first SFT run (0.8B LoRA), which finished before checkpointing was enabled for the 0.8B (it only changes memory and speed, not the maths); seed fixed.</li>
<li><b>Everything is kept:</b> per-run metrics, rollouts, evaluation generations and adapters under <code>runs/</code>, the discarded attempts under <code>runs_discarded/</code>; W&amp;B project <code>macbook-posttrain</code> has one line per (model, method, adapter).</li>
<li>Regenerate this page with <code>python analysis/make_report.py</code>.</li>
</ul>
</main></body></html>"""
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/report.html", "w") as fh: fh.write(HTML)
# machine-readable table too
import csv
with open(f"{OUT}/comparison.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "status", "baseline_acc", "final_acc", "best_acc", "step_s", "peak_mem_gb", "max_ram_gb", "max_swap_gb", "hours"])
    for r in order:
        s = S[r["name"]]; w.writerow([r["name"], status(r), s["base"], s["fin"], s["best"], s["step_s"], s["peak"], s["ram"], s["swap"], s["elapsed"]])
print("wrote", f"{OUT}/report.html", f"{len(HTML)/1024:.0f} KB", "|", len(order), "runs,", ndone, "done")
