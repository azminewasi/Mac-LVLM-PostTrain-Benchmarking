"""Builds results/report_mathstory.html (blog-style) from runs_mathstory/ (+ runs/ for the GSM8K comparison).
Usage (from the project root):  python analysis/make_report_mathstory.py   (works on partial results)"""
import json, os, sys, glob, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svgkit import lines, bars, tbl, fig, CSS, esc

RUNS, GSM, OUT = "runs_mathstory", "runs", "results"
METHODS = ["sft", "sftgentle", "sftself", "grpo", "rloo", "gspo"]
NAMES = {"sft": "SFT (equation solutions)", "sftgentle": "SFT gentle", "sftself": "SFT self-distilled", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
MCOL = {"sft": "s4", "sftgentle": "s5", "sftself": "s6", "grpo": "s0", "rloo": "s1", "gspo": "s2"}
FORMS = [("eq", "Equation"), ("story_1", "Short story"), ("story_2", "Distractor items"), ("story_3", "Long story")]
ADD_CSS = ".s5{fill:var(--s5)}.s6{fill:var(--s6)}path.ln.s5{stroke:var(--s5)}path.ln.s6{stroke:var(--s6)}i.s5{background:var(--s5)}i.s6{background:var(--s6)}"

def read(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def load(base):
    out = {}
    for m in METHODS:
        d = f"{base}/0.8B__{m}__lora"; met = read(f"{d}/metrics.jsonl")
        if not met: continue
        ev = [x for x in met if "eval/accuracy" in x]
        tr = [x for x in met if "time/step_s" in x and "reward/mean" in x]
        out[m] = dict(ev=ev, tr=tr, done=os.path.exists(f"{d}/DONE"), hours=max(x.get("time/elapsed_s", 0) for x in met) / 3600,
                      peak=max((x["mem/peak_gb"] for x in met if "mem/peak_gb" in x), default=0))
    return out
R, G = load(RUNS), load(GSM)
def acc(x, f="accuracy"): return x["eval/acc_" + f] if f != "accuracy" else x["eval/accuracy"]
def fin(m): return R[m]["ev"][-1]
def pc(v, d=0):
    t = f"{100*v:.1f}" if 0.975 <= v < 1 else f"{100*v:.{d}f}"
    return (t[:-2] if t.endswith(".0") else t) + "%"
def smooth(v, w=25): return [float(np.mean(v[max(0, i - w + 1): i + 1])) for i in range(len(v))]
base = R["sft"]["ev"][0]
live = [m for m in METHODS if m in R and not R[m]["done"]]
D = [m for m in METHODS if m in R and R[m]["done"]]          # finished runs: the only ones used for "final" comparisons
LN = lambda m: NAMES[m] + ("" if R[m]["done"] else " (running)")

# ---------------------------------------------------------------- charts
e0 = read(f"{RUNS}/0.8B__sft__lora/eval_epoch0.jsonl")
base_trunc = {f: np.mean([x["truncated"] for x in e0 if x["form"] == f + "_qs"]) for f, _ in FORMS}
c_base = bars([n for _, n in FORMS], [("accuracy", "s0", [acc(base, f) * 100 for f, _ in FORMS]), ("answers cut off at the 512-token cap", "s1", [base_trunc[f] * 100 for f, _ in FORMS])], fmt="{:.0f}", ymax=100, ylabel="% of 100 test questions")
sd = read("data/self_mathstory_n500_seed0_k4.jsonl")
self_cover = len({x["src_id"] for x in sd}) / 500 if sd else None
c_epochs = lines([(LN(m), MCOL[m], False, [(int(x["epoch"]), acc(x)) for x in R[m]["ev"]]) for m in METHODS if m in R], "epoch", "test accuracy (all four forms)", ymax=1.0, fmt="{:.0%}")
c_final = bars([n for _, n in FORMS], [(NAMES[m], MCOL[m], [acc(fin(m), f) * 100 for f, _ in FORMS]) for m in D], fmt="{:.0f}", ymax=100, ylabel="final test accuracy, %", height=340)
c_len = bars([NAMES[m] for m in D], [("mean answer length at the end (tokens)", "s0", [fin(m)["eval/gen_tokens_mean"] for m in D])], fmt="{:.0f}", ylabel="tokens")
rl = [m for m in ("grpo", "rloo", "gspo") if m in R]
rl_done = [m for m in rl if R[m]["done"]]
c_rew = lines([(LN(m), MCOL[m], False, list(enumerate(smooth([x["reward/mean"] for x in R[m]["tr"]]), 1))) for m in rl], "optimizer step", "training reward (25-step average)", ymax=1.0, fmt="{:.0%}")
c_zero = lines([(LN(m), MCOL[m], False, list(enumerate(smooth([x["reward/frac_zero_std_groups"] for x in R[m]["tr"]]), 1))) for m in rl], "optimizer step", "share of question groups with no signal", ymax=1.0, fmt="{:.0%}")
c_step = lines([(LN(m), MCOL[m], False, list(enumerate(smooth([x["time/step_s"] for x in R[m]["tr"]]), 1))) for m in rl], "optimizer step", "seconds per step (25-step average)")
c_comp = lines([(LN(m), MCOL[m], False, list(enumerate(smooth([x["completion/len_mean"] for x in R[m]["tr"]]), 1))) for m in rl], "optimizer step", "mean answer length (tokens, 25-step average)")

# tables
def row(m):
    x = fin(m); return [NAMES[m], "done" if R[m]["done"] else "running"] + [pc(acc(x, f)) for f, _ in FORMS] + [f"<b>{pc(acc(x))}</b>", f"{x['eval/gen_tokens_mean']:.0f}", pc(x["eval/truncated_frac"]), f"{R[m]['hours']:.1f}"]
T_final = tbl(["Run", "Status", "Equation", "Short story", "Distractor items", "Long story", "All", "Tokens", "Truncated", "Hours"], [["Untrained", "–"] + [pc(acc(base, f)) for f, _ in FORMS] + [f"<b>{pc(acc(base))}</b>", f"{base['eval/gen_tokens_mean']:.0f}", pc(base["eval/truncated_frac"]), "–"]] + [row(m) for m in D], "dense")
ep_rows = [[NAMES[m], int(x["epoch"])] + [pc(acc(x, f)) for f, _ in FORMS] + [pc(acc(x)), f"{x['eval/gen_tokens_mean']:.0f}", pc(x["eval/truncated_frac"])] for m in METHODS if m in R for x in R[m]["ev"]]
T_epochs = tbl(["Run", "Epoch", "Equation", "Short story", "Distractor items", "Long story", "All", "Tokens", "Truncated"], ep_rows, "dense")
GSM_NAMES = {"sft": "SFT, original recipe (human solutions on GSM8K; equation solutions here)", "sftgentle": "SFT, gentle recipe", "sftself": "SFT, self-distilled", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
gsm_rows = []
for m in METHODS:
    if m in G and m in R and R[m]["done"]: gsm_rows.append([GSM_NAMES[m], pc(G[m]["ev"][0]["eval/accuracy"]) + " → " + pc(G[m]["ev"][-1]["eval/accuracy"]), f"{G[m]['hours']:.1f}", pc(acc(base)) + " → " + pc(acc(fin(m))), f"{R[m]['hours']:.1f}"])
T_gsm = tbl(["Method (LoRA)", "GSM8K (untrained → final)", "GSM8K hours", "math-story (untrained → final)", "math-story hours"], gsm_rows)

gb = lambda m: acc(fin(m)) if m in R else None
best_rl = max(rl, key=lambda m: acc(fin(m))) if rl else None
first_ep = {m: acc(R[m]["ev"][1]) for m in rl_done if len(R[m]["ev"]) > 1}
zero_end = {m: np.mean([x["reward/frac_zero_std_groups"] for x in R[m]["tr"][-100:]]) for m in rl_done}
len_start = base["eval/gen_tokens_mean"]

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Teaching a small model to ignore distractors: SFT vs RL</title><style>{CSS}{ADD_CSS}</style></head><body><main>
<p class="kicker">Mac LLM/VLM Post-training Testing · math-story-problems</p>
<h1>Teaching a 0.8B model to ignore distractors: SFT vs RL on math story problems</h1>
<p class="sub">{datetime.date.today().strftime('%B %d, %Y')} · Qwen3.5-0.8B with LoRA · Apple M5 MacBook Pro, 16 GB · {len(R)} runs{(' (still running: ' + ', '.join(NAMES[m] for m in live) + ')') if live else ''}</p>
<p class="sub links"><a href="https://github.com/azminewasi/Mac-LVLM-PostTrain-Benchmarking">Code on GitHub</a> · <a class="selfblog" href="https://azminewasi.github.io/blog/mac-post-training-math-story.html">Blog version</a> · <a class="selfrep" href="https://github.com/azminewasi/Mac-LVLM-PostTrain-Benchmarking/blob/main/results/report_mathstory.html">report.html source</a> · Companion: <a href="https://azminewasi.github.io/blog/mac-post-training-gsm8k.html">GSM8K post</a> · <a href="https://github.com/azminewasi/MacBook-Pro-LLM-VLM-Inference-Benchmarking">inference repo</a> · <a href="https://azminewasi.github.io/blog/mac-llm-inference-benchmark.html">inference post</a></p>

<h2 id="aim">The question</h2>
<p class="lede">On GSM8K, the model's own reasoning was already strong, and supervised fine-tuning made it <i>worse</i>. Would the picture change on a dataset where the answers are short and the failure mode is different?</p>
<p>We used <a href="https://huggingface.co/datasets/azminetoushikwasi/math-story-problems"><code>azminetoushikwasi/math-story-problems</code></a>. Every item is one arithmetic problem, three numbers joined by two plus or minus signs, written four ways: a bare equation, a short story, a story with distractor item types, and a long story with many irrelevant sentences. The answer is always a single integer. That makes it a clean test of one skill: <b>pull the right numbers out of a story and calculate</b>.</p>
<p>We trained on 500 items (each shown in one randomly chosen story form) and tested on 100 held-out items in <i>all four</i> forms, so each score rests on 400 prompts. Six recipes were compared on the same model: three kinds of supervised fine-tuning (SFT) and three reinforcement-learning (RL) methods.</p>

<h3>The short answers</h3>
<ul class="toc">
<li><b>The untrained model fails on length, not arithmetic.</b> It gets {pc(acc(base,'eq'))} of bare equations right but only {pc(acc(base,'story_3'))} of long stories, and {pc(base['eval/truncated_frac'])} of its answers overall run out of tokens.</li>
<li><b>SFT works here, when the reference solutions are executable.</b> Equation-derived solutions took the model from {pc(acc(base))} to {pc(gb('sft'))}, and a gentler learning rate to {pc(gb('sftgentle'))}. Training on the model's own correct answers barely helped ({pc(gb('sftself'))}).</li>
<li><b>RL needs no reference solutions and does as well or better.</b> {', '.join(f"{NAMES[m]} reached {pc(first_ep[m])} after one epoch" for m in first_ep)}; final scores: {', '.join(f"{NAMES[m]} {pc(gb(m))}" for m in rl_done)}{(' (' + ', '.join(NAMES[m] for m in live if m in rl) + ' still running)') if any(m in rl for m in live) else ''}.</li>
<li><b>RL also makes the model concise.</b> Mean answer length fell from {len_start:.0f} tokens to {', '.join(f"{fin(m)['eval/gen_tokens_mean']:.0f} ({NAMES[m]})" for m in rl_done)}.</li>
</ul>

<h2 id="base">1. Where the untrained model fails</h2>
<p class="lede">It can do the arithmetic. It cannot cope with long stories.</p>
{fig("Untrained accuracy and truncation by question form (100 test questions each).", c_base)}
<p>Accuracy falls steadily as the story gets longer: {pc(acc(base,'eq'))} for the bare equation, {pc(acc(base,'story_1'))} for the short story, {pc(acc(base,'story_2'))} with distractor items, and {pc(acc(base,'story_3'))} for the long story. In the long-story form the model writes about {base['eval/tokens_story_3']:.0f} tokens on average, and {pc(base_trunc['story_3'])} of its answers are cut off at the 512-token cap, so it is failing as much by rambling as by getting things wrong.</p>

<h2 id="sft">2. Supervised fine-tuning: it works, if the targets are executable</h2>
<p class="lede">Imitating short, correct arithmetic lifts the model from {pc(acc(base))} to as high as {pc(gb('sftgentle'))}.</p>
<p>The dataset has only final answers, so for SFT we generated the reference solutions from the equation: <code>2385 + 761 = 3146</code>, then <code>3146 - 1063 = 2083</code>, then <code>The answer is: 2083</code>. Three variants were tried: the original recipe (learning rate 2e-5, 5 epochs), a gentle one (5e-6, 2 epochs), and self-distillation, which trains on the model's own correct finished answers.</p>
{fig("Test accuracy over the training epochs, all runs (epoch 0 is the untrained model).", c_epochs)}
<p>The equation-derived solutions are a very different story from GSM8K's human-written ones. The model learns a ~{fin('sft')['eval/gen_tokens_mean']:.0f}-token style it can carry out reliably, and the long-story form jumps from {pc(acc(base,'story_3'))} to {pc(acc(fin('sft'),'story_3'))}. The gentle recipe gets there in a single epoch and never shows the early dip on bare equations that the original recipe does. Self-distillation, by contrast, only reinforces what the model already does: its data covers only the {pc(self_cover) if self_cover else 'part'} of questions the model already solves, so it adds little where the model is weakest (the long story reaches {pc(acc(fin('sftself'),'story_3'))}).</p>
<div class="take"><b>Takeaway.</b> SFT quality depends on whether the reference solutions are something the model can execute. Short, correct, templated solutions work; on GSM8K the human solutions did not.</div>

<h2 id="rl">3. Reinforcement learning: the same reward, no reference solutions</h2>
<p class="lede">The RL methods reach the top of the dataset, and they get there fast.</p>
<p>GRPO, RLOO and GSPO see only the questions and a correct-or-not reward: no reference solutions, no hints about the equation. Learning rate 2e-6, four sampled answers per question, up to 640 new tokens.</p>
{fig("Training reward (share of sampled answers that finish and are correct), 25-step moving average.", c_rew)}
<p>{' and '.join(f"{NAMES[m]} scored {pc(first_ep[m])} on the test set after a single epoch" for m in first_ep)}. By the end {', '.join(f"{NAMES[m]} reached {pc(gb(m))}" for m in rl_done)}{(' while ' + ', '.join(NAMES[m] for m in live if m in rl) + ' is still training') if any(m in rl for m in live) else ''}. At this level (about 400 prompts, one seed) the differences between methods are within noise; what is clear is that RL matches the best SFT recipe without needing its reference solutions.</p>
{fig("Final test accuracy by question form, all runs.", c_final)}
<h3>RL also changes how the model writes</h3>
<p>Because there is no length penalty, the answer length shrinking is a by-product: shorter answers finish more often and so earn more reward.</p>
{fig("Mean answer length during RL training (25-step moving average).", c_comp)}
{fig("Seconds per optimizer step during RL training: faster as answers shorten, which made these runs far cheaper than the GSM8K ones.", c_step)}
<h3>Why the late steps change little</h3>
<p>Once the model answers almost everything correctly, most question groups have four identical rewards and therefore no gradient: by the end, {', '.join(f"{pc(zero_end[m])} ({NAMES[m]})" for m in rl_done)} of groups carry no signal. Training keeps running, but very little changes after the first epochs.</p>
{fig("Share of question groups where all four sampled answers score the same (no learning signal).", c_zero)}

<h2 id="compare">4. SFT versus RL, side by side</h2>
{fig("Mean answer length at the end of training.", c_len)}
{T_final}
<h2 id="gsm">5. Same methods, two datasets</h2>
<p class="lede">The same recipes behave very differently on GSM8K and on math-story-problems.</p>
{T_gsm}
<p>On GSM8K the model already reasons well in its own long, step-by-step style, supervised fine-tuning on human solutions pulled it away from that style, and RL gave gains of 5 to 13 points (up to 16 with full fine-tuning). Here the untrained model is held back by rambling, a short templated style is something it can execute, and so both SFT on clean solutions and RL give very large gains. The shared lesson: <b>post-training helps when it teaches something the model can actually do, and the cheapest reliable signal is a verifiable reward.</b></p>

<h2 id="conclusions">6. Conclusions</h2>
<ol>
<li>On a clean, verifiable task, RL with a correctness-only reward matched the best supervised recipe without reference solutions.</li>
<li>SFT works when reference solutions are short, correct and executable; a gentler learning rate was better than the aggressive one.</li>
<li>Self-distillation preserves a model's style but cannot teach what the model never gets right.</li>
<li>RL makes answers shorter as a side effect, which also made the RL runs several times cheaper than their GSM8K counterparts.</li>
<li>After the first epoch, most RL groups carry no signal, which suggests shorter training would have been enough (not tested).</li>
</ol>
<h3>Caveats</h3>
<p>One seed per recipe and 400 test prompts (100 per form), so differences of a few points are not meaningful near the 98-100% ceiling. The SFT targets are an easy deterministic template. Test equations never appear in the training data, but the problem templates do, so this measures skill on this family of problems, not general math. Late in some RL runs (e.g. RLOO), accuracy slipped a few points as answers kept shrinking; the final-epoch score is reported.</p>

<h2 id="appendix">Appendix</h2>
<h3>A. Accuracy by epoch, all runs</h3>{T_epochs}
<h3>B. Method and data notes</h3>
<ul>
<li><b>Data:</b> train 500 items from the 131,072-item training split (seeded; each uses one of the three story forms); evaluation on the first 100 items of the 3,072-item test split, in all four forms; greedy decoding, 512-token cap.</li>
<li><b>Hyperparameters:</b> LoRA rank 16, scale 20; SFT lr 2e-5 (5 epochs); gentle SFT and self-distilled SFT lr 5e-6 (2 epochs); RL lr 2e-6, 4 prompts x 4 rollouts per step, temperature 1.0, up to 640 new tokens, correctness-only reward, 5 epochs; gradient checkpointing, clipping 1.0.</li>
<li><b>W&amp;B project:</b> <code>macbook-posttrain-mathstory</code>; all metrics, rollouts and evaluation answers are kept under <code>runs_mathstory/</code>; model weights were deleted locally after the runs (re-run <code>train.py</code> to regenerate). See also the research journal (<code>JOURNAL.md</code>) and the GSM8K report.</li>
</ul>
</main></body></html>"""
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/report_mathstory.html", "w").write(HTML)
print("wrote", f"{OUT}/report_mathstory.html", f"{len(HTML)/1024:.0f} KB", "|", ", ".join(f"{m}:{'done' if R[m]['done'] else 'running'}" for m in R))
