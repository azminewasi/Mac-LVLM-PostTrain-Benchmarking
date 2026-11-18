"""Probe: untrained Qwen3.5 on SpatialEval spatial-map (4-option MC). Measures accuracy, tokens, speed and memory vs image size.
Usage: python -m tests.probe_vision [--n 100] [--size 448] [--model Qwen/Qwen3.5-0.8B] [--max-tokens 192]"""
import argparse, glob, io, json, random, re, time
import mlx.core as mx, pandas as pd
from PIL import Image
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--size", type=int, default=448)
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B"); ap.add_argument("--max-tokens", type=int, default=192); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = pd.read_parquet(glob.glob("tmp/hf/**/vqa/test-00000-of-00002.parquet", recursive=True)[0])
df = df[df.id.str.startswith("spatialmap")].reset_index(drop=True)
idx = random.Random(a.seed).sample(range(len(df)), a.n)
model, proc = load(a.model); cfg = load_config(a.model)
def qtype(t): return "direction" if "In which direction" in t else "which_object" if "Which object" in t else "count"
res = []; t_all = time.time(); mx.reset_peak_memory()
for k, i in enumerate(idx):
    r = df.iloc[i]; img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"); img.thumbnail((a.size, a.size)); p = f"tmp/probe_{k}.png"; img.save(p)
    prompt = apply_chat_template(proc, cfg, r["text"] + "\nThink step by step, then end with a final line: 'The answer is: <letter>'", num_images=1, enable_thinking=False)
    out = ""; last = None
    for g in stream_generate(model, proc, prompt, image=p, max_tokens=a.max_tokens, temperature=0.0):
        out += g.text; last = g
    ms = re.findall(r"answer is:?\s*\**\(?([A-D])\b", out, re.I) or re.findall(r"answer:?\s*\**\(?([A-D])\b[.)]", out, re.I)
    letter = ms[-1].upper() if ms else None; fmt = int(bool(re.findall(r"answer is:?\s*\**\(?([A-D])\b", out, re.I)))
    ok = int(letter == r["oracle_option"])
    res.append(dict(type=qtype(r["text"]), ok=ok, ptok=last.prompt_tokens, gtok=last.generation_tokens, dtps=last.generation_tps, ptps=last.prompt_tps, trunc=int(last.finish_reason == "length"), letter=letter, fmt=fmt))
    if k % 20 == 0: print(k, res[-1], flush=True)
d = pd.DataFrame(res)
print(f"\nsize={a.size} n={a.n} acc={d.ok.mean():.2f} prompt_tok={d.ptok.mean():.0f} gen_tok={d.gtok.mean():.0f} trunc={d.trunc.mean():.2f} fmt={d.fmt.mean():.2f} decode={d.dtps.mean():.0f} tok/s prefill={d.ptps.mean():.0f} tok/s "
      f"peak={mx.get_peak_memory()/2**30:.2f} GB total={time.time()-t_all:.0f}s")
print(d.groupby("type").ok.agg(["mean", "count"]).round(2).to_string()); print("letters", d.letter.value_counts(dropna=False).to_dict())
