"""Single-stream, per-token inference timing for the untrained model and every trained adapter (visual QA, SpatialEval spatial-map).
One question at a time (no batching), greedy, the same 100 test questions as the post-training evaluation. Per sample: prompt tokens (incl. image tokens),
time to first token, prefill and decode tokens/s, and the latency of every generated token (inter-token latency) so decode speed can be studied by token position.
Needs the GPU to itself: run it after the training queue has finished.
Usage: python -m tests.vision_infer_speed [--runs-dir runs_spatial] [--n 100] [--dry]
Output: results/vision_infer_speed.json (summary per run), results/vision_infer_speed_samples.jsonl (per-sample incl. per-token latencies in ms)."""
import argparse, glob, json, os, time
import numpy as np
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", default="runs_spatial"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-tokens", type=int, default=640); ap.add_argument("--dry", action="store_true", help="list what would run, load nothing")
a = ap.parse_args()
NAMES = {"sft": "SFT", "sftgentle": "SFT gentle", "sftself": "SFT self-distilled", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
adapters = [(m, glob.glob(f"{a.runs_dir}/0.8B__{m}__lora/adapter_epoch*.safetensors")) for m in NAMES]
adapters = [(m, sorted(p)[-1]) for m, p in adapters if p and os.path.exists(f"{a.runs_dir}/0.8B__{m}__lora/DONE")]
print("untrained +", [(NAMES[m], os.path.basename(p)) for m, p in adapters])
if a.dry: raise SystemExit(0)

import pt.vision as V
from mlx_vlm import load, stream_generate
from mlx_vlm.utils import load_config
BINS = [0, 8, 16, 32, 64, 128, 256, 512]
model, processor = load(V.MODEL_ID); cfg = load_config(V.MODEL_ID)
V.apply_lora(model, 16, 20.0); model.eval()           # fresh LoRA has B = 0, i.e. identical to the untrained model
_, ev = V.load_spatial(500, a.n, 0)
os.makedirs("results", exist_ok=True); fs = open("results/vision_infer_speed_samples.jsonl", "w")

def measure(label, method):
    recs = []; mx.reset_peak_memory(); t_run = time.time()
    for k, r in enumerate(ev):
        prompt = V.build_prompt(processor, cfg, r); t0 = time.perf_counter(); stamps = []; text = ""; last = None
        for g in stream_generate(model, processor, prompt, image=r["image"], max_tokens=a.max_tokens, temperature=0.0):
            stamps.append(time.perf_counter() - t0); text += g.text; last = g
        itl = [1000 * (stamps[i] - stamps[i - 1]) for i in range(1, len(stamps))]
        ok = int(V.reward_fn(text, r["gold"], last.finish_reason == "length")[1] > 0)
        rec = dict(run=label, id=r["id"], qtype=r["qtype"], correct=ok, prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens, ttft_s=stamps[0], total_s=stamps[-1],
                   prefill_tps=last.prompt_tps, decode_tps=last.generation_tps, finish=last.finish_reason, itl_ms=[round(x, 2) for x in itl])
        recs.append(rec); fs.write(json.dumps(rec) + "\n"); fs.flush()
        if k % 20 == 0: print(f"{label} {k + 1}/{len(ev)} gen={rec['gen_tokens']} ttft={rec['ttft_s']:.2f}s decode={rec['decode_tps']:.0f} tok/s", flush=True)
    allitl = np.array([x for r in recs for x in r["itl_ms"]]) if any(r["itl_ms"] for r in recs) else np.array([np.nan])
    by_pos = []
    for lo, hi in zip(BINS, BINS[1:] + [10 ** 9]):
        v = [x for r in recs for i, x in enumerate(r["itl_ms"], 1) if lo < i <= hi]
        by_pos.append(float(np.mean(v)) if v else None)
    m = lambda k: float(np.mean([r[k] for r in recs]))
    return dict(label=label, method=method, acc=float(np.mean([r["correct"] for r in recs])), prompt_tokens_mean=m("prompt_tokens"), gen_tokens_mean=m("gen_tokens"), ttft_mean=m("ttft_s"),
                total_mean=m("total_s"), prefill_tps_mean=m("prefill_tps"), decode_tps_mean=m("decode_tps"), itl_p50_ms=float(np.nanpercentile(allitl, 50)), itl_p95_ms=float(np.nanpercentile(allitl, 95)),
                itl_by_pos_ms=by_pos, peak_gb=mx.get_peak_memory() / 2 ** 30, wall_s=time.time() - t_run)

runs = [measure("Untrained", None)]
for meth, path in adapters:
    model.load_weights(path, strict=False); mx.eval(model.parameters()); runs.append(measure(NAMES[meth], meth))
json.dump(dict(bins=BINS, n=a.n, runs=runs, model=V.MODEL_ID, max_tokens=a.max_tokens, note="single stream, greedy, 448 px images"), open("results/vision_infer_speed.json", "w"), indent=1)
print("wrote results/vision_infer_speed.json")
for r in runs: print(f"{r['label']:20s} acc {r['acc']:.2f} gen {r['gen_tokens_mean']:.0f} tok  ttft {r['ttft_mean']:.2f}s  decode {r['decode_tps_mean']:.0f} tok/s  total {r['total_mean']:.2f}s")
