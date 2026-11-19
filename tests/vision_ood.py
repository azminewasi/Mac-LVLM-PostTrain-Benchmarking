"""Out-of-distribution test for the visual post-training runs: do the trained adapters still work on a different image type and skill?
Test set: SpatialEval maze navigation (mazenav): "how many right/left turns along the blue path?", 4-option multiple choice, same letter format, maze images
instead of maps. 100 random questions (seed 0). Chance is 25%. Evaluates the untrained 0.8B and every finished adapter; batched, greedy, 640-token cap.
Needs the GPU to itself. Usage: python -m tests.vision_ood [--runs-dir runs_spatial] [--n 100] [--model 0.8B] [--dry]
Output: results/vision_ood.json, results/vision_ood_samples.jsonl"""
import argparse, collections, glob, io, json, os, random
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", default="runs_spatial"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="0.8B"); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
NAMES = {"sft": "SFT", "sftgentle": "SFT gentle", "sftself": "SFT self-distilled", "grpo": "GRPO", "rloo": "RLOO", "gspo": "GSPO"}
adapters = []
for m in NAMES:
    d = f"{a.runs_dir}/{a.model}__{m}__lora"; p = sorted(glob.glob(f"{d}/adapter_epoch*.safetensors"))
    if p and os.path.exists(f"{d}/DONE"): adapters.append((m, p[-1]))
print("untrained +", [(NAMES[m], os.path.basename(p)) for m, p in adapters])
if a.dry: raise SystemExit(0)

import pandas as pd
import mlx.core as mx
from PIL import Image
import pt.vision as V
from mlx_vlm import load
from mlx_vlm.utils import load_config

def load_maze(n, seed):
    df = pd.read_parquet(V._parquet()); df = df[df.id.str.startswith("mazenav")].reset_index(drop=True)
    os.makedirs("data/maze_images", exist_ok=True); rows = []
    for i in random.Random(seed).sample(range(len(df)), n):
        r = df.iloc[i]; path = f"data/maze_images/{r['id']}.png"
        if not os.path.exists(path):
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"); im.thumbnail((V.SIZE, V.SIZE)); im.save(path)
        rows.append(dict(id=r["id"], source_index=int(i), question=r["text"], gold=r["oracle_option"], qtype="maze", image=path))
    return rows

rows = load_maze(a.n, a.seed)
model, processor = load(V.MODELS[a.model] if hasattr(V, "MODELS") else V.MODEL_ID); cfg = load_config(V.MODELS[a.model] if hasattr(V, "MODELS") else V.MODEL_ID)
V.apply_lora(model, 16, 20.0); model.eval()
prep = V.prep_rows(model, processor, cfg, rows)
os.makedirs("results", exist_ok=True); fs = open("results/vision_ood_samples.jsonl", "w")
gold_counts = collections.Counter(r["gold"] for r in rows)
out = []
def run(label, method):
    m, recs = V.evaluate(model, processor, cfg, rows, prep, max_tokens=640)
    for r in recs: fs.write(json.dumps(dict(run=label, **r)) + "\n")
    fs.flush()
    preds = collections.Counter(r["pred"] for r in recs)
    out.append(dict(label=label, method=method, acc=m["eval/accuracy"], trunc=m["eval/truncated_frac"], fmt=m["eval/format_rate"], gen_tokens=m["eval/gen_tokens_mean"],
                    pred_letters=dict(preds), tok_s=m["eval/gen_tokens_per_s"]))
    print(f"{label:20s} acc {m['eval/accuracy']:.2f}  trunc {m['eval/truncated_frac']:.2f}  fmt {m['eval/format_rate']:.2f}  tokens {m['eval/gen_tokens_mean']:.0f}  preds {dict(preds)}", flush=True)
run("Untrained", None)
for meth, path in adapters:
    model.load_weights(path, strict=False); mx.eval(model.parameters()); run(NAMES[meth], meth)
json.dump(dict(task="SpatialEval mazenav (out-of-distribution)", n=a.n, chance=0.25, gold_letters=dict(gold_counts), majority_baseline=max(gold_counts.values()) / a.n, runs=out), open("results/vision_ood.json", "w"), indent=1)
print("wrote results/vision_ood.json")
