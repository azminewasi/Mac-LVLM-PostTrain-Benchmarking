"""Token-level and speed analytics for the visual-QA post-training runs (runs_spatial/): answer lengths, tokens per correct answer, batched
prefill/decode throughput per epoch, RL sampling throughput over training, training throughput. If results/vision_infer_speed.json exists
(tests/vision_infer_speed.py, single-stream timing), a second part with time-to-first-token, per-token latency and decode speed vs context length is added.
Usage (from the project root):  python analysis/vision_analytics.py      -> results/vision_analytics.html, results/vision_analytics.csv"""
import csv, glob, json, os, sys, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import lines, bars, tbl, fig, CSS, esc

RUNS, OUT = "runs_spatial", "results"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "0.8B"; SUF = "" if MODEL == "0.8B" else "_" + MODEL
METHODS = ["sft", "sftgentle", "sftself", "grpo", "rloo", "gspo"]
NAMES = {"sft": "SFT", "sftgentle": "SFT gentle", "sftself": "SFT self-distilled", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
MCOL = {"sft": "s4", "sftgentle": "s5", "sftself": "s6", "grpo": "s0", "rloo": "s1", "gspo": "s2"}
ADD_CSS = ".s5{fill:var(--s5)}.s6{fill:var(--s6)}path.ln.s5{stroke:var(--s5)}path.ln.s6{stroke:var(--s6)}i.s5{background:var(--s5)}i.s6{background:var(--s6)}"
read = lambda p: [json.loads(l) for l in open(p)] if os.path.exists(p) else []
smooth = lambda v, w=25: [float(np.mean(v[max(0, i - w + 1): i + 1])) for i in range(len(v))]

R = {}
for m in METHODS:
    d = f"{RUNS}/{MODEL}__{m}__lora"; met = read(f"{d}/metrics.jsonl")
    if not met: continue
    ep = {}
    for f in sorted(glob.glob(f"{d}/eval_epoch*.jsonl")):
        e = int(f.split("epoch")[-1].split(".")[0]); ep[e] = read(f)
    evm = {int(round(x["epoch"])): x for x in met if "eval/accuracy" in x}
    R[m] = dict(met=met, ep=ep, evm=evm, done=os.path.exists(f"{d}/DONE"),
                tr=[x for x in met if "train/lr" in x and "time/step_s" in x], hours=max(x.get("time/elapsed_s", 0) for x in met) / 3600)
if not R: raise SystemExit("no runs yet")
order = [m for m in METHODS if m in R]
LN = lambda m: NAMES[m] + ("" if R[m]["done"] else " (running)")

def stat(recs):
    g = np.array([x["gen_tokens"] for x in recs]); ok = np.array([x["correct"] for x in recs])
    return dict(n=len(g), mean=float(g.mean()), med=float(np.median(g)), p90=float(np.percentile(g, 90)), mx=int(g.max()), trunc=float(np.mean([x["truncated"] for x in recs])),
                acc=float(ok.mean()), per_correct=float(g.sum() / ok.sum()) if ok.sum() else None, fmt=float(np.mean([x["fmt"] for x in recs])))

rows_csv, T = [], {}
for m in order:
    for e, recs in sorted(R[m]["ep"].items()):
        s = stat(recs); x = R[m]["evm"].get(e, {})
        s |= dict(time_s=x.get("eval/time_s"), dec_tps=x.get("eval/gen_tokens_per_s"), pre_tps=x.get("eval/prompt_tokens_per_s"), peak=x.get("eval/peak_mem_gb"))
        s["s_per_correct"] = s["time_s"] / (s["acc"] * s["n"]) if s["time_s"] and s["acc"] else None
        T[(m, e)] = s; rows_csv.append(dict(run=m, epoch=e, **s))
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/vision_analytics{SUF}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)

def series(key, scale=1.0):
    return [(LN(m), MCOL[m], False, [(e, T[(m, e)][key] * scale) for e in sorted(R[m]["ep"]) if T[(m, e)].get(key) is not None]) for m in order]
c_tok = lines(series("mean"), "epoch", "mean generated tokens per answer", fmt="{:.0f}")
c_corr = lines(series("per_correct"), "epoch", "generated tokens per correct answer", fmt="{:.0f}")
c_acc = lines(series("acc"), "epoch", "test accuracy", ymax=1.0, fmt="{:.0%}")
c_dec = lines(series("dec_tps"), "epoch", "batched decode tokens/s (25 answers at once)", fmt="{:.0f}")
c_pre = lines(series("pre_tps"), "epoch", "prefill tokens/s (prompt incl. 196 image tokens)", fmt="{:.0f}")
c_spc = lines(series("s_per_correct"), "epoch", "evaluation seconds per correct answer", fmt="{:.2f}")

RL = [m for m in ("grpo", "rloo", "gspo") if m in R]
def curve(key, w=25): return [(LN(m), MCOL[m], False, list(enumerate(smooth([x[key] for x in R[m]["tr"] if key in x and x[key] == x[key]], w), 1))) for m in RL]
c_rl_len = lines(curve("completion/len_mean"), "optimizer step", "sampled tokens per answer (25-step average)", fmt="{:.0f}") if RL else ""
c_rl_tps = lines(curve("gen/tokens_per_s"), "optimizer step", "sampling tokens/s, 16 answers at once (25-step average)", fmt="{:.0f}") if RL else ""
c_rl_step = lines(curve("time/step_s"), "optimizer step", "seconds per RL step (25-step average)", fmt="{:.0f}") if RL else ""
mean = lambda m, k: float(np.mean([x[k] for x in R[m]["tr"] if k in x and x[k] == x[k]])) if any(k in x for x in R[m]["tr"]) else None
c_train = bars([LN(m) for m in order], [("training tokens/s", "s0", [mean(m, "train/tokens_per_s") for m in order])], fmt="{:.0f}", ylabel="tokens / s")

f0 = lambda v, fmt="{:.0f}": "–" if v is None else fmt.format(v)
rows = []
for m in order:
    for e in sorted(R[m]["ep"]):
        s = T[(m, e)]
        if e in (0, max(R[m]["ep"])):
            rows.append([NAMES[m] + ("" if R[m]["done"] else " (running)"), "untrained" if e == 0 else f"epoch {e}", f0(100 * s["acc"], "{:.0f}%"), f0(s["mean"]), f0(s["med"]), f0(s["p90"]), s["mx"], f0(100 * s["trunc"], "{:.0f}%"),
                         f0(s["per_correct"]), f0(s["dec_tps"]), f0(s["pre_tps"]), f0(s["s_per_correct"], "{:.2f}")])
T_main = tbl(["Run", "Model", "Accuracy", "Mean tokens", "Median", "90th pct", "Max", "Truncated", "Tokens / correct", "Decode tok/s", "Prefill tok/s", "s / correct"], rows, "dense")

# distributions: untrained vs final, by question type
qrows = []
for m in order:
    last = max(R[m]["ep"])
    for q in ("direction", "which_object", "count"):
        a = [x for x in R[m]["ep"][0] if x["qtype"] == q]; b = [x for x in R[m]["ep"][last] if x["qtype"] == q]
        if a and b: qrows.append([NAMES[m], q, len(b), f"{np.mean([x['gen_tokens'] for x in a]):.0f}", f"{np.mean([x['gen_tokens'] for x in b]):.0f}",
                                  f"{100*np.mean([x['correct'] for x in a]):.0f}% → {100*np.mean([x['correct'] for x in b]):.0f}%"])
T_q = tbl(["Run", "Question type", "n", "Tokens untrained", "Tokens final", "Accuracy"], qrows, "dense")

base = T[(order[0], 0)]; best = max((m for m in order if R[m]["done"]), key=lambda m: T[(m, max(R[m]["ep"]))]["acc"], default=None)
take = ""
if best:
    fb = T[(best, max(R[best]["ep"]))]
    take = (f"The untrained model writes about {base['mean']:.0f} tokens per answer ({base['per_correct']:.0f} tokens for each correct one). After training, {NAMES[best]} writes {fb['mean']:.0f} tokens per answer "
            f"and {fb['per_correct']:.0f} per correct answer, so the same question costs {base['per_correct'] / fb['per_correct']:.0f}x fewer generated tokens.")

# ---- optional single-stream token timing (tests/vision_infer_speed.py)
SS = ""
sp = f"{OUT}/vision_infer_speed.json" if MODEL == "0.8B" else ""
if sp and os.path.exists(sp):
    d = json.load(open(sp)); runs = d["runs"]
    names = [r["label"] for r in runs]
    SS = "<h2 id='single'>Single-stream token timing</h2><p>One question at a time, greedy, the same 100 test questions, measured per token on the MacBook (no batching).</p>"
    SS += fig("Time to first token (prefill of the ~300-token prompt: 196 image tokens + text) and total time per answer, mean over the test set.", bars(names, [("time to first token (s)", "s0", [r["ttft_mean"] for r in runs]), ("total time per answer (s)", "s1", [r["total_mean"] for r in runs])], fmt="{:.2f}", ylabel="seconds"))
    SS += fig("Decode speed (tokens/s) and prefill speed (tokens/s ÷ 10), mean over the test set.", bars(names, [("decode tokens/s", "s0", [r["decode_tps_mean"] for r in runs]), ("prefill tokens/s ÷ 10", "s3", [r["prefill_tps_mean"] / 10 for r in runs])], fmt="{:.0f}", ylabel="tokens / s"))
    bins = d["bins"]
    ser = [(r["label"], MCOL.get(r.get("method"), "s4"), False, [(b, v) for b, v in zip(bins, r["itl_by_pos_ms"]) if v is not None]) for r in runs if r.get("itl_by_pos_ms")]
    if ser: SS += fig("Inter-token latency (ms) by token position in the answer: does decoding slow down as the answer grows?", lines(ser, "token position in the answer", "ms per token", fmt="{:.1f}"))
    SS += tbl(["Run", "Accuracy", "Prompt tokens", "Gen tokens", "TTFT s", "Prefill tok/s", "Decode tok/s", "ITL p50 ms", "ITL p95 ms", "Total s", "Peak MLX GB"],
              [[r["label"], f"{100*r['acc']:.0f}%", f"{r['prompt_tokens_mean']:.0f}", f"{r['gen_tokens_mean']:.0f}", f"{r['ttft_mean']:.2f}", f"{r['prefill_tps_mean']:.0f}", f"{r['decode_tps_mean']:.0f}", f"{r['itl_p50_ms']:.1f}", f"{r['itl_p95_ms']:.1f}", f"{r['total_mean']:.2f}", f"{r['peak_gb']:.2f}"] for r in runs], "dense")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual-QA post-training: token and speed analytics</title><style>{CSS}{ADD_CSS}</style></head><body><main>
<p class="kicker">Mac LLM/VLM Post-training Testing · visual QA (SpatialEval spatial-map)</p>
<h1>Token-level and speed analytics: what post-training does to inference cost</h1>
<p class="sub">{datetime.date.today().strftime('%B %d, %Y')} · Qwen3.5-{MODEL} with LoRA · Apple M5 MacBook Pro, 16 GB · {len(order)} runs{' (still running)' if any(not R[m]['done'] for m in order) else ''}</p>
<p class="lede">Training changes what the model writes, and what it writes decides what inference costs. This page measures it per token: how long the answers are, how many tokens each correct answer costs, and how fast the MacBook produces them.</p>
{f'<div class="take"><b>Headline.</b> {take}</div>' if take else ''}
<h2>1. Answer length and cost per correct answer</h2>
{fig("Mean generated tokens per test answer after each epoch (epoch 0 = untrained).", c_tok)}
{fig("Generated tokens per correct answer: total tokens written divided by the number of correct answers.", c_corr)}
{fig("Test accuracy after each epoch, for reference.", c_acc)}
{T_main}
<h3>By question type</h3>
{T_q}
<h2>2. Speed of batched evaluation</h2>
<p>The test set is evaluated 25 answers at a time. These are throughput numbers (many answers at once), so they are higher than a single user would see; section 4 has single-stream timing if it has been measured.</p>
{fig("Batched decode throughput during evaluation.", c_dec)}
{fig("Prefill throughput during evaluation: each prompt carries about 300 image tokens.", c_pre)}
{fig("Evaluation wall-clock seconds per correct answer (accuracy and speed together).", c_spc)}
{"<h2>3. RL sampling and training throughput</h2>" + fig("Tokens per sampled answer during RL training.", c_rl_len) + fig("Sampling throughput during RL training: 4 prompts x 4 answers per step.", c_rl_tps) + fig("Seconds per RL step.", c_rl_step) if RL else ""}
{fig("Training throughput (forward + backward) by run.", c_train)}
{SS}
<p class="dim">Per-sample generated-token counts come from the saved evaluation answers; throughput numbers come from the per-epoch evaluation and per-step logs. Regenerate with python analysis/vision_analytics.py.</p>
</main></body></html>"""
open(f"{OUT}/vision_analytics{SUF}.html", "w").write(html)
print(f"wrote {OUT}/vision_analytics.html and .csv | runs: " + ", ".join(f"{m}:{'done' if R[m]['done'] else 'running'}" for m in order))
