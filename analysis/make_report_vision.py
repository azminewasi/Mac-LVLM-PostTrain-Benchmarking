"""Builds results/report_vision.html (blog-style) from runs_spatial/ (0.8B + 2B LoRA), results/vision_infer_speed.json and results/vision_ood*.json.
Usage (from the project root):  python analysis/make_report_vision.py"""
import json, os, sys, glob, datetime, collections, html as _html
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svgkit import lines, bars, tbl, fig, CSS, esc
from perfkit import perf_section, run_perf

RUNS, OUT = "runs_spatial", "results"
METHODS = ["sft", "sftgentle", "sftself", "grpo", "rloo", "gspo"]
NAMES = {"sft": "SFT", "sftgentle": "SFT gentle", "sftself": "SFT self-distilled", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
MCOL = {"sft": "s4", "sftgentle": "s5", "sftself": "s6", "grpo": "s0", "rloo": "s1", "gspo": "s2"}
QT = [("direction", "Direction"), ("which_object", "Which object"), ("count", "Count")]
ADD_CSS = (".s5{fill:var(--s5)}.s6{fill:var(--s6)}path.ln.s5{stroke:var(--s5)}path.ln.s6{stroke:var(--s6)}i.s5{background:var(--s5)}i.s6{background:var(--s6)}"
           "pre.ex{white-space:pre-wrap;font-size:.82rem;line-height:1.45;padding:.7rem .9rem;border-radius:8px;border:1px solid var(--grid,#ddd);background:var(--card,rgba(127,127,127,.08));margin:.3rem 0 .9rem}"
           ".exhead{font-weight:600;margin:.8rem 0 0}")
read = lambda p: [json.loads(l) for l in open(p)] if os.path.exists(p) else []
smooth = lambda v, w=25: [float(np.mean(v[max(0, i - w + 1): i + 1])) for i in range(len(v))]
pc = lambda v, d=0: f"{100*v:.{d}f}%"

R = {}
for mod in ("0.8B", "2B"):
    for m in METHODS:
        d = f"{RUNS}/{mod}__{m}__lora"; met = read(f"{d}/metrics.jsonl")
        if not met: continue
        ep = {int(f.split("epoch")[-1].split(".")[0]): read(f) for f in sorted(glob.glob(f"{d}/eval_epoch*.jsonl"))}
        ev = [x for x in met if "eval/accuracy" in x]
        R[(mod, m)] = dict(met=met, ep=ep, ev=ev, tr=[x for x in met if "time/step_s" in x and "reward/mean" in x], done=os.path.exists(f"{d}/DONE"),
                           hours=max(x.get("time/elapsed_s", 0) for x in met) / 3600, fin=ev[-1], base=ev[0])
M8 = [m for m in METHODS if ("0.8B", m) in R]; M2 = [m for m in METHODS if ("2B", m) in R]
r8 = lambda m: R[("0.8B", m)]; r2 = lambda m: R[("2B", m)]
acc = lambda x, f="accuracy": x["eval/accuracy"] if f == "accuracy" else x["eval/acc_" + f]
base8 = r8("sft")["base"]; base2 = r2("sft")["base"]
U8 = r8("sft")["ep"][0]; U2 = r2("sft")["ep"][0]
RL8 = [m for m in ("grpo", "rloo", "gspo") if m in M8]

def stats(recs):
    g = np.array([x["gen_tokens"] for x in recs]); ok = np.array([x["correct"] for x in recs])
    return dict(mean=float(g.mean()), med=float(np.median(g)), per_correct=float(g.sum() / ok.sum()) if ok.sum() else None, trunc=float(np.mean([x["truncated"] for x in recs])))
last = lambda mod, m: R[(mod, m)]["ep"][max(R[(mod, m)]["ep"])]
ST8 = {m: stats(last("0.8B", m)) for m in M8}; ST8["base"] = stats(U8); ST2 = {m: stats(last("2B", m)) for m in M2}; ST2["base"] = stats(U2)

# ------------------------------------------------------------ quantitative charts and tables
c_base = bars([n for _, n in QT], [("accuracy", "s0", [acc(base8, q) * 100 for q, _ in QT]), ("answers cut off at the 640-token cap", "s1", [100 * np.mean([x["truncated"] for x in U8 if x["qtype"] == q]) for q, _ in QT])],
              fmt="{:.0f}", ymax=100, ylabel="% of the test questions of that type")
c_ep8 = lines([(NAMES[m], MCOL[m], False, [(int(round(x["epoch"])), x["eval/accuracy"]) for x in r8(m)["ev"]]) for m in M8], "epoch", "test accuracy", ymax=1.0, fmt="{:.0%}")
c_ep2 = lines([(NAMES[m], MCOL[m], False, [(int(round(x["epoch"])), x["eval/accuracy"]) for x in r2(m)["ev"]]) for m in M2], "epoch", "test accuracy (2B)", ymax=1.0, fmt="{:.0%}")
c_final = bars([n for _, n in QT], [(NAMES[m], MCOL[m], [acc(r8(m)["fin"], q) * 100 for q, _ in QT]) for m in M8], fmt="{:.0f}", ymax=100, ylabel="final test accuracy, %", height=340)
c_rew = lines([(NAMES[m], MCOL[m], False, list(enumerate(smooth([x["reward/mean"] for x in r8(m)["tr"]]), 1))) for m in RL8], "optimizer step", "training reward (25-step average)", ymax=1.0, fmt="{:.0%}")
c_zero = lines([(NAMES[m], MCOL[m], False, list(enumerate(smooth([x["reward/frac_zero_std_groups"] for x in r8(m)["tr"]]), 1))) for m in RL8], "optimizer step", "share of prompt groups with no learning signal", ymax=1.0, fmt="{:.0%}")
c_len = lines([(NAMES[m], MCOL[m], False, list(enumerate(smooth([x["completion/len_mean"] for x in r8(m)["tr"]]), 1))) for m in RL8], "optimizer step", "tokens per sampled answer (25-step average)", fmt="{:.0f}")
rows = [["Untrained", "–"] + [pc(acc(base8, q)) for q, _ in QT] + [f"<b>{pc(acc(base8))}</b>", f"{base8['eval/gen_tokens_mean']:.0f}", pc(base8["eval/truncated_frac"]), "–"]]
for m in M8:
    x = r8(m)["fin"]; rows.append([NAMES[m], f"{int(round(x['epoch']))} ep"] + [pc(acc(x, q)) for q, _ in QT] + [f"<b>{pc(acc(x))}</b>", f"{x['eval/gen_tokens_mean']:.0f}", pc(x["eval/truncated_frac"]), f"{r8(m)['hours']:.1f}"])
T8 = tbl(["Run (0.8B, LoRA)", "Training", "Direction", "Which object", "Count", "All", "Tokens / answer", "Truncated", "Hours"], rows)
rows = [["Untrained", "–"] + [pc(acc(base2, q)) for q, _ in QT] + [f"<b>{pc(acc(base2))}</b>", f"{base2['eval/gen_tokens_mean']:.0f}", pc(base2["eval/truncated_frac"]), "–"]]
for m in M2:
    x = r2(m)["fin"]; rows.append([NAMES[m], f"{int(round(x['epoch']))} ep"] + [pc(acc(x, q)) for q, _ in QT] + [f"<b>{pc(acc(x))}</b>", f"{x['eval/gen_tokens_mean']:.0f}", pc(x["eval/truncated_frac"]), f"{r2(m)['hours']:.1f}"])
T2 = tbl(["Run (2B, LoRA)", "Training", "Direction", "Which object", "Count", "All", "Tokens / answer", "Truncated", "Hours"], rows)
sh = [m for m in M2 if m in M8]
T_cmp = tbl(["Method", "0.8B: untrained → final", "2B: untrained → final", "0.8B hours", "2B hours"],
            [[NAMES[m], f"{pc(acc(base8))} → <b>{pc(acc(r8(m)['fin']))}</b>", f"{pc(acc(base2))} → <b>{pc(acc(r2(m)['fin']))}</b>", f"{r8(m)['hours']:.1f}", f"{r2(m)['hours']:.1f}"] for m in sh])
c_cmp = bars([NAMES[m] for m in ["base"] + sh] if False else ["Untrained"] + [NAMES[m] for m in sh], [("0.8B", "s0", [acc(base8) * 100] + [acc(r8(m)["fin"]) * 100 for m in sh]), ("2B", "s1", [acc(base2) * 100] + [acc(r2(m)["fin"]) * 100 for m in sh])], fmt="{:.0f}", ymax=100, ylabel="test accuracy, %")
ep_rows = [[f"{mod} {NAMES[m]}", int(round(x["epoch"])), pc(acc(x)), pc(acc(x, "direction")), pc(acc(x, "which_object")), pc(acc(x, "count")), f"{x['eval/gen_tokens_mean']:.0f}", pc(x["eval/truncated_frac"])]
           for mod, MS in (("0.8B", M8), ("2B", M2)) for m in MS for x in R[(mod, m)]["ev"]]
T_ep = tbl(["Run", "Epoch", "All", "Direction", "Which object", "Count", "Tokens", "Truncated"], ep_rows, "dense")

# ------------------------------------------------------------ qualitative
gold_counts = collections.Counter(x["gold"] for x in U8)
count_gold = collections.Counter(x["gold"] for x in U8 if x["qtype"] == "count")
def letters(recs, q=None): return collections.Counter(x["pred"] for x in recs if q is None or x["qtype"] == q)
fm = collections.Counter()
for x in U8:
    kind = "cut off at the token cap" if x["truncated"] else ("finished, no 'The answer is: X' line" if not x["fmt"] else ("correct" if x["correct"] else "wrong letter"))
    fm[kind] += 1
fm_rows = [[k, v] for k, v in fm.most_common()]
T_fm = tbl(["What the untrained model did on the 100 test questions", "Count"], fm_rows)
def ex(i, mods=(("Untrained", "0.8B", None), ("SFT", "0.8B", "sft"), ("SFT self-distilled", "0.8B", "sftself"), ("GRPO", "0.8B", "grpo"), ("RLOO", "0.8B", "rloo"), ("GSPO", "0.8B", "gspo")), cut=700):
    out = []
    for lab, mod, m in mods:
        r = U8[i] if m is None else last(mod, m)[i]
        t = r["response"].strip(); t = t if len(t) <= cut else t[:cut].rstrip() + " [...]"
        out.append(f'<p class="exhead">{lab} · {r["gen_tokens"]} tokens · {"correct" if r["correct"] else "wrong"}</p><pre class="ex">{_html.escape(t)}</pre>')
    return "".join(out)
IDX = 2
sp = json.load(open(f"{OUT}/vision_infer_speed.json")) if os.path.exists(f"{OUT}/vision_infer_speed.json") else None
# contradiction: untrained reasons to one direction but picks another
contra = None
for i, u in enumerate(U8):
    if u["qtype"] == "direction" and not u["correct"] and not u["truncated"] and u["fmt"] and "✅" in u["response"] and i != IDX: contra = i; break
count_rows = []
for lab, recs in (("Untrained", U8),) + tuple((NAMES[m], last("0.8B", m)) for m in M8):
    c = [x for x in recs if x["qtype"] == "count"]
    count_rows.append([lab, len(c), pc(np.mean([x["correct"] for x in c])), pc(np.mean([x["truncated"] for x in c])), f"{np.mean([x['gen_tokens'] for x in c]):.0f}", ", ".join(f"{k}: {v}" for k, v in sorted(letters(c).items(), key=lambda kv: str(kv[0])))])
T_count = tbl(["Run", "Count questions", "Accuracy", "Truncated", "Tokens", "Predicted letters"], count_rows, "dense")
# agreement between models on the final test answers
def agree(a, b): return float(np.mean([x["pred"] == y["pred"] for x, y in zip(a, b)]))
ag_rows = [[NAMES[a], *[("–" if a == b else pc(agree(last("0.8B", a), last("0.8B", b)))) for b in M8]] for a in M8]
T_agree = tbl(["Share of identical answers"] + [NAMES[b] for b in M8], ag_rows, "dense")
# letter prior sanity
maj = max(gold_counts.values()) / len(U8)
# OOD
O8 = json.load(open(f"{OUT}/vision_ood.json")); O2 = json.load(open(f"{OUT}/vision_ood_2B.json")) if os.path.exists(f"{OUT}/vision_ood_2B.json") else None
def ood_rows(O, tag): return [[f"{tag} {r['label']}", pc(r["acc"]), pc(r["trunc"]), pc(r["fmt"]), f"{r['gen_tokens']:.0f}", ", ".join(f"{k}: {v}" for k, v in sorted(r["pred_letters"].items()))] for r in O["runs"]]
T_ood = tbl(["Run", "Accuracy", "Truncated", "Answer line present", "Tokens", "Predicted letters"], ood_rows(O8, "0.8B") + (ood_rows(O2, "2B") if O2 else []), "dense")
c_ood = bars([r["label"] for r in O8["runs"]], [("0.8B", "s0", [100 * r["acc"] for r in O8["runs"]])] , fmt="{:.0f}", ymax=100, ylabel="% correct on maze-navigation questions")

# ------------------------------------------------------------ tokens and speed
sd = lambda m: ST8[m]
c_tok = bars(["Untrained"] + [NAMES[m] for m in M8], [("mean tokens per answer", "s0", [ST8["base"]["mean"]] + [ST8[m]["mean"] for m in M8]), ("tokens per correct answer", "s1", [ST8["base"]["per_correct"]] + [ST8[m]["per_correct"] for m in M8])], fmt="{:.0f}", ylabel="tokens")
ev_dec = lambda m: r8(m)["fin"]["eval/gen_tokens_per_s"]; ev_pre = lambda m: r8(m)["fin"]["eval/prompt_tokens_per_s"]
c_batch = bars(["Untrained"] + [NAMES[m] for m in M8], [("decode tokens/s (25 answers at once)", "s0", [base8["eval/gen_tokens_per_s"]] + [ev_dec(m) for m in M8]), ("prefill tokens/s ÷ 10", "s3", [base8["eval/prompt_tokens_per_s"] / 10] + [ev_pre(m) / 10 for m in M8])], fmt="{:.0f}", ylabel="tokens / s")
SS = ""
if sp:
    runs = sp["runs"]; names = [r["label"] for r in runs]
    c_ttft = bars(names, [("time to first token (s)", "s0", [r["ttft_mean"] for r in runs]), ("total time per answer (s)", "s1", [r["total_mean"] for r in runs])], fmt="{:.2f}", ylabel="seconds")
    c_ss = bars(names, [("decode tokens/s", "s0", [r["decode_tps_mean"] for r in runs]), ("prefill tokens/s ÷ 10", "s3", [r["prefill_tps_mean"] / 10 for r in runs])], fmt="{:.0f}", ylabel="tokens / s")
    ser = [(r["label"], MCOL.get(r.get("method"), "s4"), False, [(b, v) for b, v in zip(sp["bins"], r["itl_by_pos_ms"]) if v is not None]) for r in runs if r.get("itl_by_pos_ms")]
    c_itl = lines(ser, "token position in the answer", "ms per token", fmt="{:.1f}")
    T_ss = tbl(["Run", "Accuracy", "Prompt tokens", "Gen tokens", "TTFT s", "Prefill tok/s", "Decode tok/s", "ITL p50 ms", "ITL p95 ms", "Total s", "Peak MLX GB"],
               [[r["label"], pc(r["acc"]), f"{r['prompt_tokens_mean']:.0f}", f"{r['gen_tokens_mean']:.0f}", f"{r['ttft_mean']:.3f}", f"{r['prefill_tps_mean']:.0f}", f"{r['decode_tps_mean']:.0f}", f"{r['itl_p50_ms']:.1f}", f"{r['itl_p95_ms']:.1f}", f"{r['total_mean']:.2f}", f"{r['peak_gb']:.2f}"] for r in runs], "dense")
    un = runs[0]; sf = runs[1]; gs = next(r for r in runs if r["method"] == "gspo")
    SS = True

# ------------------------------------------------------------ Mac performance
ents = [dict(label=f"{mod} {NAMES[m]}", cls=MCOL[m], kind="rl" if m in ("grpo", "rloo", "gspo") else "sft", adapter="lora", met=R[(mod, m)]["met"]) for mod, MS in (("0.8B", M8), ("2B", M2)) for m in MS]
note = ("These are the nine LoRA runs of this study: six on Qwen3.5-0.8B and three on Qwen3.5-2B. The vision tower is frozen and its features are computed once per image, so the training steps are language-model steps over 196 image tokens plus the text; "
        "the cost of the image shows up as longer sequences, not as extra vision compute.")
PERF = perf_section(ents, 9, note, single_stream_tps=64, anchor="perf")
P = {(mod, m): run_perf(R[(mod, m)]["met"]) for mod, MS in (("0.8B", M8), ("2B", M2)) for m in MS}
rl_cmp = []
for m in ("grpo",):
    for mod in ("0.8B", "2B"):
        p = P[(mod, m)]; rl_cmp.append([f"{mod} {NAMES[m]}", f"{p['step_s']:.1f}", f"{p['gen_s']:.1f}", f"{p['train_s']:.1f}", f"{p['gen_tps']:.0f}", f"{p['tr_tps']:.0f}", f"{p['peak']:.1f}", f"{p['ram']:.1f}", f"{p['swap']:.1f}", f"{p['hours']:.1f}"])
for mod, MS in (("0.8B", M8), ("2B", M2)):
    for m in MS:
        if m in ("grpo",): continue
        p = P[(mod, m)]; rl_cmp.append([f"{mod} {NAMES[m]}", f"{p['step_s']:.1f}", f"{p['gen_s']:.1f}" if p["gen_s"] else "–", f"{p['train_s']:.1f}" if p["train_s"] else "–", f"{p['gen_tps']:.0f}" if p["gen_tps"] else "–", f"{p['tr_tps']:.0f}", f"{p['peak']:.1f}", f"{p['ram']:.1f}", f"{p['swap']:.1f}", f"{p['hours']:.1f}"])
T_perf = tbl(["Run", "s / step", "of which generation", "of which training", "gen tok/s", "train tok/s", "peak MLX GB", "RAM GB (max)", "swap GB (max)", "hours"], rl_cmp, "dense")
g2 = r2("grpo")["tr"]; first = g2[0]
step2 = [x["time/step_s"] for x in g2]; step8 = [x["time/step_s"] for x in r8("grpo")["tr"]]
c_step = lines([("0.8B GRPO", "s0", False, list(enumerate(smooth(step8), 1))), ("2B GRPO", "s1", False, list(enumerate(smooth(step2), 1)))], "optimizer step", "seconds per RL step (25-step average)", fmt="{:.0f}")
c_swap = lines([("0.8B GRPO", "s0", False, [(i, x["sys/swap_gb_max"]) for i, x in enumerate(r8("grpo")["tr"], 1)]), ("2B GRPO", "s1", False, [(i, x["sys/swap_gb_max"]) for i, x in enumerate(g2, 1)])], "optimizer step", "swap in use, GB", fmt="{:.1f}")
c_time = bars([f"{mod} {NAMES[m]}" for mod, MS in (("0.8B", M8), ("2B", M2)) for m in MS], [("hours", "s0", [R[(mod, m)]["hours"] for mod, MS in (("0.8B", M8), ("2B", M2)) for m in MS])], fmt="{:.1f}", ylabel="hours (training + evaluation)")
tot_h = sum(v["hours"] for v in R.values())
best8 = max(M8, key=lambda m: acc(r8(m)["fin"]))
rl_first = {m: acc(r8(m)["ev"][1]) for m in RL8}
zero_end = {m: np.mean([x["reward/frac_zero_std_groups"] for x in r8(m)["tr"][-100:]]) for m in RL8}

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Post-training a small vision-language model on a MacBook: spatial reasoning</title><style>{CSS}{ADD_CSS}</style></head><body><main>
<p class="kicker">Mac LLM/VLM Post-training Testing · visual question answering (SpatialEval spatial-map)</p>
<h1>Post-training a small vision-language model on a MacBook: SFT vs RL on spatial maps</h1>
<p class="sub">{datetime.date.today().strftime('%B %d, %Y')} · Qwen3.5-0.8B and 2B with LoRA · Apple M5 MacBook Pro, 16 GB · {len(R)} runs, {tot_h:.0f} hours of training and evaluation</p>
<p class="sub links"><a href="https://github.com/azminewasi/Mac-LVLM-PostTrain-Benchmarking">Code on GitHub</a> · <a class="selfblog" href="https://azminewasi.github.io/blog/mac-post-training-vision.html">Blog version</a> · <a href="https://azminewasi.github.io/blog/mac-post-training-gsm8k.html">GSM8K text post-training</a> · <a href="https://azminewasi.github.io/blog/mac-post-training-math-story.html">math-story post-training</a> · <a href="https://azminewasi.github.io/blog/mac-llm-inference-benchmark.html">Inference benchmark</a></p>

<h2 id="aim">The question</h2>
<p class="lede">The text-only studies showed what SFT and RL do to a small model on a MacBook. Does the same recipe work when the model has to <i>look at an image</i>, and what does it cost the machine?</p>
<p>We used <a href="https://huggingface.co/datasets/MilaWang/SpatialEval"><code>MilaWang/SpatialEval</code></a>, spatial-map split: each item is a picture of a map with named locations and a multiple-choice question about it, in three types: <b>direction</b> ("in which direction is X relative to Y?"), <b>which object</b> ("which object is in the northwest of Y?") and <b>count</b> ("how many objects are to the east of Y?"). We trained on a small slice, 500 items, and tested on 100 held-out items, because the point is to benchmark what the Mac does, not to chase a leaderboard. The same six recipes as before ran on Qwen3.5-0.8B; three of them were repeated on Qwen3.5-2B. All use LoRA (rank 16) on the language model, with the vision tower frozen.</p>
<h3>The short answers</h3>
<ul class="toc">
<li><b>Both families of recipe work, and they work fast.</b> On 0.8B, the untrained model scores {pc(acc(base8))}; after training, SFT reaches {pc(acc(r8('sft')['fin']))}, gentle SFT {pc(acc(r8('sftgentle')['fin']))}, and the RL methods {', '.join(f"{NAMES[m]} {pc(acc(r8(m)['fin']))}" for m in RL8)}. Scores come from 100 questions and one seed, so differences of up to about ten points are noise.</li>
<li><b>Most of the failure was format and length, not vision.</b> The untrained model wrote {base8['eval/gen_tokens_mean']:.0f} tokens per answer on average, {pc(base8['eval/truncated_frac'])} of its answers hit the 640-token cap, and {100-100*np.mean([x['fmt'] for x in U8]):.0f}% never produced the answer line. Training mostly taught it to finish and to commit.</li>
<li><b>Training changes inference cost by an order of magnitude.</b> The same questions cost {ST8['base']['per_correct']:.0f} generated tokens per correct answer untrained, {ST8['sft']['per_correct']:.0f} after SFT and {ST8['gspo']['per_correct']:.0f} after GSPO; one answer takes {sp['runs'][0]['total_mean']:.1f} s untrained and {sp['runs'][1]['total_mean']:.2f} s after SFT on the MacBook.</li>
<li><b>The 2B model fits for RL on 16 GB, barely.</b> GRPO peaked at {P[('2B','grpo')]['peak']:.1f} GB of MLX memory with up to {P[('2B','grpo')]['swap']:.1f} GB of swap, and the first RL step took {first['time/step_s']:.0f} s against about {np.mean(step8[:5]):.0f} s for the first five steps of 0.8B. It was not better than 0.8B after RL: {pc(acc(r2('grpo')['fin']))} vs {pc(acc(r8('grpo')['fin']))}.</li>
<li><b>The skill transfers a little, and not predictably.</b> On a different task (maze navigation) trained models scored {pc(min(r['acc'] for r in O8['runs'][1:] if r['method']!='sftself'))}-{pc(max(r['acc'] for r in O8['runs']))} against 25% chance, but the ranking of recipes changes, and one recipe (self-distillation) collapsed.</li>
</ul>

<h2 id="setup">1. Setup</h2>
<p class="lede">Images go through a frozen vision tower once; everything we train is the language model.</p>
<ul>
<li><b>Model path.</b> Qwen3.5 has a hybrid language model (gated delta-net and attention layers) and a vision tower. Maps are resized to 448 px, which becomes exactly 196 image tokens in the prompt (the mean prompt, image plus question and chat template, is {sp['runs'][0]['prompt_tokens_mean']:.0f} tokens). Vision features are computed once per image and cached; LoRA (rank 16, scale 20) sits on the language-model layers only, about 1.25% of the 0.8B parameters.</li>
<li><b>Data.</b> 500 training items and 100 test items, seeded, drawn from the spatial-map shard; question types are mixed as in the source ({', '.join(f"{n}: {sum(1 for x in U8 if x['qtype']==q)}" for q, n in QT)} in the test set).</li>
<li><b>Six recipes.</b> SFT (answer-only targets, <code>The answer is: C. Northwest</code>, lr 2e-5, 5 epochs), gentle SFT (lr 5e-6, 2 epochs), self-distilled SFT (rejection-sample the model's own correct answers, then train on them), and three critic-free RL methods: GRPO, RLOO and GSPO (4 prompts × 4 sampled answers per step, temperature 1.0, correctness-only reward that also needs the answer line, no KL penalty, lr 2e-6, 5 epochs = 625 steps).</li>
<li><b>Scoring.</b> An answer counts only if it finishes within 640 new tokens and ends in a parsable <code>The answer is: X</code> line with the right letter. This is strict on purpose: it is also the RL reward.</li>
</ul>

<h2 id="base">2. Where the untrained model fails</h2>
<p class="lede">The 0.8B model scores {pc(acc(base8))} and its biggest problem is counting.</p>
{fig("Untrained 0.8B accuracy and truncation by question type (test set).", c_base)}
{T_fm}
<p>Direction and which-object questions are only partly answered correctly ({pc(acc(base8,'direction'))} and {pc(acc(base8,'which_object'))}); count questions are almost never answered correctly ({pc(acc(base8,'count'))}), and {pc(np.mean([x['truncated'] for x in U8 if x['qtype']=='count']))} of them ran into the token cap while the model enumerated objects one by one. Chance is 25% on four options, and the most common gold letter covers {pc(maj)} of the test set, so a model that only guessed the most common letter would score about {pc(maj)}.</p>

<h2 id="sft">3. Supervised fine-tuning</h2>
<p class="lede">Five epochs of answer-only SFT lifts the 0.8B model from {pc(acc(base8))} to {pc(acc(r8('sft')['fin']))}.</p>
<p>The SFT targets contain only the final line, taken from the dataset's oracle option. That is a deliberate simplification: there are no reference reasoning traces for these maps, so SFT teaches the model to <i>answer directly</i>. It works because the model already has the visual skill; it only needed to commit.</p>
{fig("Test accuracy over the epochs on 0.8B, all runs (epoch 0 is untrained).", c_ep8)}
<p>The aggressive recipe (lr 2e-5, 5 epochs) climbs slowly ({', '.join(pc(acc(x)) for x in r8('sft')['ev'])}); the gentle recipe (lr 5e-6, 2 epochs) gets to {pc(acc(r8('sftgentle')['fin']))} in well under half the time. <b>Self-distilled SFT</b> ends at only {pc(acc(r8('sftself')['fin']))}: it trains on the model's own correct answers, so it can only reinforce what the model already did, including the rambling style, and its training set was thin on the hard type: only 19% of its examples were count questions (88 of 457), against 39% of the test set, because the untrained model rarely got those right.</p>

<h2 id="rl">4. Reinforcement learning</h2>
<p class="lede">GRPO, RLOO and GSPO need no reference answers, only the correct-or-not reward.</p>
{fig("Training reward (share of sampled answers that finish and are correct), 25-step moving average.", c_rew)}
<p>After the first epoch the three RL runs scored {', '.join(f"{NAMES[m]} {pc(rl_first[m])}" for m in RL8)}, and finished at {', '.join(f"{NAMES[m]} {pc(acc(r8(m)['fin']))}" for m in RL8)}. As in the text studies, RL also changes how the model writes, with no length penalty: shorter answers finish more often and so earn more reward.</p>
{fig("Mean tokens per sampled answer during RL training (25-step moving average).", c_len)}
{fig("Share of prompt groups where all four sampled answers got the same reward, so the group teaches nothing.", c_zero)}
<p>By the end, {', '.join(f"{pc(zero_end[m])} of {NAMES[m]}'s groups" for m in RL8)} carried no signal, which suggests the later epochs did little (we did not test shorter training).</p>
<h3>Final results, 0.8B</h3>
{fig("Final test accuracy by question type, all six recipes.", c_final)}
{T8}

<h2 id="b2">5. Does 2B do better?</h2>
<p class="lede">Slightly better untrained, equal after training.</p>
{T2}
{fig("Untrained and final accuracy, 0.8B vs 2B, for the three methods run on both.", c_cmp)}
{T_cmp}
{fig("Test accuracy over the epochs on 2B.", c_ep2)}
<p>The 2B model starts higher ({pc(acc(base2))} vs {pc(acc(base8))}), but after SFT both reach about {pc(acc(r8('sft')['fin']))}-{pc(acc(r2('sft')['fin']))}, and 2B GRPO ended at {pc(acc(r2('grpo')['fin']))} (best epoch {pc(max(acc(x) for x in r2('grpo')['ev']))}), no better than 0.8B GRPO ({pc(acc(r8('grpo')['fin']))}). With 100 test questions none of these gaps is significant. The honest reading is that on this small, templated task the extra parameters bought little; the cost of the larger model is in section 9.</p>

<h2 id="qual">6. What the models actually write</h2>
<p class="lede">The same map question, answered by six different trained models.</p>
<p>The test item below asks for the direction of Jackal's Jams relative to Giraffe's Guitars; the right answer is C, Northwest. Answers are truncated to 700 characters.</p>
{ex(IDX)}
<h3>What the examples show</h3>
<ul>
<li><b>The untrained model reasons correctly and then answers wrongly.</b> Above, it correctly says Jackal's Jams is "up and to the left", then maps that to <i>Southwest</i> and ticks option A. Its visual reading was right; the translation to a letter was not. This is why format-and-commit training pays off so quickly.</li>
<li><b>SFT removes the reasoning altogether.</b> Seven tokens: <code>The answer is: C. Northwest</code>. It is the cheapest and, here, among the most accurate recipes, because the model has to read the map and answer in one step.</li>
<li><b>Self-distillation keeps the old style.</b> It still writes a long worked answer, and it still makes the same kind of mistake on counts.</li>
<li><b>The RL methods each found a different style</b>, from GRPO's long paragraph, through RLOO's short bullet list, to GSPO's near-direct answer ({ST8['grpo']['mean']:.0f}, {ST8['rloo']['mean']:.0f} and {ST8['gspo']['mean']:.0f} tokens on average). None was rewarded for style; they drifted to whatever earned reward at the lowest cost.</li>
</ul>
{('<h3>Another failure of the untrained model</h3><p>A different test question, where it picks a letter that contradicts its own working:</p>' + ex(contra, (("Untrained", "0.8B", None), ("GSPO", "0.8B", "gspo")), 900)) if contra is not None else ''}
<h3>Counting stays the hard part</h3>
{T_count}
<p>Counting is where the models differ most. The untrained model almost never gets it right; after training, direction and which-object accuracy is high for most recipes, while count accuracy ranges from {pc(min(acc(r8(m)['fin'],'count') for m in M8))} to {pc(max(acc(r8(m)['fin'],'count') for m in M8))}. Whether these gains are counting or learned shortcuts is not something 100 questions can settle; the maze test below asks a related question.</p>
<h3>Do the recipes agree?</h3>
{T_agree}
<p>Recipes with similar accuracy do not necessarily give the same answers, so the shared score hides different mistakes. This is a reason to prefer larger evaluation sets in follow-up work.</p>

<h2 id="ood">7. Does it transfer? A maze-navigation test</h2>
<p class="lede">Different image type, different skill, same answer format.</p>
<p>We took 100 random questions from the maze-navigation task of the same dataset ("how many right turns are on the blue path?", four options, chance 25%, majority-letter baseline {pc(O8['majority_baseline'])}) and evaluated every adapter without any further training. The prompt and answer line are the same.</p>
{fig("0.8B maze-navigation accuracy (no training on mazes).", c_ood)}
{T_ood}
<ul>
<li>The untrained model scores near zero mostly because it runs out of tokens ({pc(O8['runs'][0]['trunc'])} truncated at 640): it rambles, and never gets to the answer line. We did not rerun with a larger cap, so this is a statement about the cap.</li>
<li>The adapters that learned to answer directly (SFT, gentle SFT) and the RL adapters all score well above chance. Part of this is that they always produce an answer; part may be real transfer. Without a control (for example a blank image) we cannot separate the two.</li>
<li>The self-distilled adapter collapses to {pc(next(r['acc'] for r in O8['runs'] if r['method']=='sftself'))}: its rambling style carried over unchanged and kept hitting the cap.</li>
<li>2B behaves differently: {', '.join(f"{r['label']} {pc(r['acc'])}" for r in O2['runs'][1:]) if O2 else ''}. Its GRPO adapter leaves {pc(O2['runs'][-1]['trunc']) if O2 else ''} of answers truncated on mazes.</li>
</ul>

<h2 id="tokens">8. Token-level inference cost</h2>
<p class="lede">Training changed how many tokens each answer costs, and with them the time per answer.</p>
{fig("Mean tokens per answer and tokens per correct answer, at the end of training (test set, greedy).", c_tok)}
<p>The untrained model writes {ST8['base']['mean']:.0f} tokens per answer, {ST8['base']['per_correct']:.0f} per <i>correct</i> answer. SFT writes {ST8['sft']['mean']:.0f}; GSPO {ST8['gspo']['mean']:.0f}; RLOO {ST8['rloo']['mean']:.0f}; GRPO {ST8['grpo']['mean']:.0f}. Dividing generated tokens by correct answers is the number that matters for cost: <b>{ST8['base']['per_correct']/ST8['sft']['per_correct']:.0f}x</b> fewer for SFT and {ST8['base']['per_correct']/ST8['gspo']['per_correct']:.0f}x fewer for GSPO than the untrained model.</p>
<h3>Batched evaluation speed</h3>
{fig("Throughput while evaluating the 100 test questions 25 at a time (batched decode and prefill).", c_batch)}
<p>These are aggregate numbers for many answers in flight, so they are higher than what one user sees. Batched decode throughput was about {base8['eval/gen_tokens_per_s']:.0f} tokens/s untrained and {min(ev_dec(m) for m in M8):.0f}-{max(ev_dec(m) for m in M8):.0f} tokens/s after training; it was lowest for {min(M8, key=ev_dec) and NAMES[min(M8, key=ev_dec)]} ({ev_dec(min(M8, key=ev_dec)):.0f}), whose answers are the shortest of the RL runs, probably because with short answers the evaluation is dominated by prefill and start-up rather than steady decoding (not isolated). What matters more is wall-clock time: the 100-question evaluation took {base8['eval/time_s']:.0f} s untrained and {r8('sft')['fin']['eval/time_s']:.0f} s after SFT.</p>
<h3>Single stream: what one user would see</h3>
<p>The same 100 questions, one at a time, greedy, timed per token.</p>
{fig("Time to first token and total time per answer, mean over the test set.", c_ttft)}
{fig("Decode speed and prefill speed (÷10), mean over the test set.", c_ss)}
{fig("Inter-token latency by token position in the answer: does decoding slow down as the answer grows?", c_itl)}
{T_ss}
<ul>
<li><b>The image costs a fixed ~{sp['runs'][0]['ttft_mean']:.2f} s.</b> Time to first token is the same for every run ({min(r['ttft_mean'] for r in sp['runs']):.3f}-{max(r['ttft_mean'] for r in sp['runs']):.3f} s): prefill of about {sp['runs'][0]['prompt_tokens_mean']:.0f} tokens at about {sp['runs'][0]['prefill_tps_mean']:.0f} tokens/s, which includes the vision tower.</li>
<li><b>Decoding is flat.</b> Inter-token latency is about {sp['runs'][0]['itl_p50_ms']:.1f} ms (p95 {sp['runs'][0]['itl_p95_ms']:.1f} ms) at every position from the first token to the 512th; so within 512 tokens the cost per token did not grow with the answer length (we did not test longer ones). Single-stream decode speed is {min(r['decode_tps_mean'] for r in sp['runs']):.0f}-{max(r['decode_tps_mean'] for r in sp['runs']):.0f} tokens/s for every adapter.</li>
<li><b>So answer length is the whole story.</b> End-to-end time per answer falls from {sp['runs'][0]['total_mean']:.2f} s (untrained) to {sp['runs'][1]['total_mean']:.2f} s (SFT), almost entirely because of the number of tokens, since speed per token did not change. After SFT, 0.126 of the 0.25 s is the image prefill.</li>
<li><b>Memory at inference is small.</b> Peak MLX memory was about {max(r['peak_gb'] for r in sp['runs']):.1f} GB for 0.8B with the LoRA adapter applied.</li>
</ul>

{PERF}
<h3>0.8B versus 2B on the MacBook</h3>
{T_perf}
{fig("Seconds per RL step, GRPO: 0.8B vs 2B (25-step moving average).", c_step)}
<p>On average a 2B GRPO step took {P[('2B','grpo')]['step_s']:.0f} s against {P[('0.8B','grpo')]['step_s']:.0f} s for 0.8B, with generation {100*P[('2B','grpo')]['gen_share']:.0f}% of the step ({100*P[('0.8B','grpo')]['gen_share']:.0f}% for 0.8B). The first 2B step took {first['time/step_s']:.0f} s because untrained answers were long; steps got cheaper as the model learned to answer briefly, the same effect seen at 0.8B, and the 2B run took {r2('grpo')['hours']:.1f} h in total for {len(g2)} steps.</p>
<h3>Where it ran out of room</h3>
{fig("Swap in use during GRPO: the 2B run pushed several GB to disk.", c_swap)}
{fig("Wall-clock hours per run (training plus evaluations).", c_time)}
<ul>
<li><b>Memory is the limit, not compute.</b> The 2B GRPO run needed micro-batch 1 and peaked at {P[('2B','grpo')]['peak']:.1f} GB of MLX memory with system RAM at {P[('2B','grpo')]['ram']:.1f} GB and up to {P[('2B','grpo')]['swap']:.1f} GB of swap. It finished, but it left little headroom: longer answers, more images per prompt or a bigger model would push hard on 16 GB. We did not try to train 4B or larger models, so where exactly the limit lies is untested.</li>
<li><b>Images cost sequence length.</b> Each prompt carries 196 image tokens (and about 300 tokens in total), so the sequence limit was 1,100 tokens and the images were downscaled to 448 px; the number of images per example was kept at one.</li>
<li><b>No thermal throttling.</b> As in the text studies, the laptop never slowed itself, even through multi-hour RL runs.</li>
<li><b>Evaluation is not free.</b> One 100-question evaluation of the untrained 2B model took about 160 s, as long as several RL steps, because untrained answers are long. RL training also spent most of each step in generation, so cost scales with answer length, not with the optimiser.</li>
<li><b>Everything was measured with one machine and one seed.</b> The timing numbers should be read as typical for this laptop under this setup, not as hardware limits.</li>
</ul>

<h2 id="conclusions">10. Conclusions</h2>
<ol>
<li>On a small visual QA task, a few hundred examples and a few hours were enough to take a 0.8B model from {pc(acc(base8))} to roughly 85-90%, with either SFT or RL.</li>
<li>Most of the gain is format and commitment: finishing within the token budget and producing the answer line. Reading the image was mostly already possible.</li>
<li>RL needs no reference answers and learns its own short style; SFT is cheaper (minutes) when targets are as simple as a letter.</li>
<li>Self-distillation did not work here: it cannot teach what the model never gets right.</li>
<li>Training reduced cost per correct answer by roughly {ST8['base']['per_correct']/ST8['sft']['per_correct']:.0f}x with the cheapest recipe, because per-token speed is constant and answer length is what changed.</li>
<li>The 2B model fits for RL on a 16 GB MacBook, but only just, and it did not beat 0.8B on this task.</li>
</ol>
<h3>Caveats</h3>
<p>One seed per recipe and 100 test questions (±10 points at 95%). A single dataset slice of a templated, synthetic task. The maze-navigation results have no control (blank image, larger token cap), so transfer is suggestive only. Single-stream speed was measured on 0.8B only; the 2B figures are training and batched-evaluation numbers. We froze the vision tower; training it was not attempted.</p>

<h2 id="appendix">Appendix</h2>
<h3>A. Accuracy by epoch, all runs</h3>{T_ep}
<h3>B. Method and data notes</h3>
<ul>
<li><b>Data:</b> SpatialEval spatial-map, vision-QA split, shard 0; 500 train / 100 test, seed 0; images resized to 448 px (longest side). Maze test: SpatialEval maze-navigation, 100 random questions, seed 0.</li>
<li><b>Hyperparameters:</b> LoRA rank 16, scale 20 (language model only); SFT lr 2e-5 (5 epochs); gentle and self-distilled SFT lr 5e-6 (2 epochs); RL lr 2e-6, 4 prompts × 4 rollouts per step, temperature 1.0, up to 640 new tokens, no KL, 5 epochs; evaluation greedy, 100 questions, 25 at a time.</li>
<li><b>Software:</b> MLX 0.32 and mlx-vlm 0.7.12 on macOS 26, Apple M5, 16 GB; the vision features are cached per image and the language model runs in training mode so the differentiable gated-delta kernel is used.</li>
<li><b>W&amp;B project:</b> <code>macbook-posttrain-spatial</code>. Metrics, rollouts and evaluation answers are kept under <code>runs_spatial/</code>; adapter weights were not committed.</li>
<li><b>Other pages:</b> <a href="vision_analytics.html">token and speed analytics (0.8B)</a> · <a href="vision_analytics_2B.html">analytics (2B)</a>.</li>
</ul>
</main></body></html>"""
open(f"{OUT}/report_vision.html", "w").write(HTML)
print(f"wrote {OUT}/report_vision.html ({len(HTML)//1024} KB)")
