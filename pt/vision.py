"""Visual question answering on a small spatial-reasoning set: SpatialEval spatial-map (4-option multiple choice on map images).

Pieces for train_vision.py: data (500 train / 100 eval, images resized to 448 px), prompt, strict answer parsing / reward,
a cache of frozen-vision-tower features (so training never re-runs the vision tower), the embeds hook for the loss helpers,
LoRA on the language side only, and batched generation with images (mlx-vlm batch_generate)."""
import glob, io, json, os, random, re, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_unflatten

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODELS = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B"}
SIZE = 448
INSTR = "\nThink step by step, then end with a final line: 'The answer is: <letter>'"
_FIN = re.compile(r"answer is:?\s*\**\(?([A-D])\b", re.I)


def qtype(t):
    return "direction" if "In which direction" in t else "which_object" if "Which object" in t else "count"


def _parquet():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("MilaWang/SpatialEval", "vqa/test-00000-of-00002.parquet", repo_type="dataset", cache_dir="tmp/hf")


def load_spatial(n_train=500, n_eval=100, seed=0):
    """Returns (train_rows, eval_rows); disjoint random samples of the 1,500 spatial-map questions, saved to data/."""
    pt_, pe_ = f"data/train_spatial_n{n_train}_seed{seed}.jsonl", f"data/eval_spatial_n{n_eval}_seed{seed}.jsonl"
    if os.path.exists(pt_) and os.path.exists(pe_):
        rd = lambda p: [json.loads(l) for l in open(p)]
        tr, ev = rd(pt_), rd(pe_)
        if all(os.path.exists(r["image"]) for r in tr + ev): return tr, ev
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(_parquet()); df = df[df.id.str.startswith("spatialmap")].reset_index(drop=True)
    idx = random.Random(seed).sample(range(len(df)), n_train + n_eval)
    os.makedirs("data/spatial_images", exist_ok=True); rows = []
    for i in idx:
        r = df.iloc[i]; path = f"data/spatial_images/{r['id']}.png"
        if not os.path.exists(path):
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"); im.thumbnail((SIZE, SIZE)); im.save(path)
        rows.append(dict(id=r["id"], source_index=int(i), question=r["text"], gold=r["oracle_option"], gold_text=r["oracle_answer"],
                         full=r["oracle_full_answer"], qtype=qtype(r["text"]), image=path, target=f"The answer is: {r['oracle_full_answer']}"))
    tr, ev = rows[:n_train], rows[n_train:]
    for p, rs in ((pt_, tr), (pe_, ev)):
        with open(p, "w") as f:
            for r in rs: f.write(json.dumps(r) + "\n")
    return tr, ev


def parse_letter(text):
    m = _FIN.findall(text)
    return m[-1].upper() if m else None


def reward_fn(text, gold, truncated):
    """(reward, correct, format): correct = the final 'The answer is: X' names the gold option and the answer is not cut off."""
    letter = parse_letter(text)
    c = float(letter == gold and not truncated)
    return c, c, float(letter is not None)


def build_prompt(processor, cfg, row):
    from mlx_vlm.prompt_utils import apply_chat_template
    return apply_chat_template(processor, cfg, row["question"] + INSTR, num_images=1, enable_thinking=False)


def prep_rows(model, processor, cfg, rows):
    """Per row: prompt string, input ids (image placeholders expanded), vision-tower features (frozen), grid. Cached by id."""
    from mlx_vlm.utils import prepare_inputs
    out = {}; dt = model.vision_tower.patch_embed.proj.weight.dtype
    for r in rows:
        prompt = build_prompt(processor, cfg, r)
        inp = prepare_inputs(processor, images=[r["image"]], prompts=prompt, image_token_index=model.config.image_token_index)
        feats, _ = model.vision_tower(inp["pixel_values"].astype(dt), inp["image_grid_thw"]); mx.eval(feats)
        out[r["id"]] = dict(prompt=prompt, ids=inp["input_ids"][0].tolist(), feats=feats, grid=inp["image_grid_thw"][0])
    return out


def make_fwd(model, ctx):
    """EXTRA['fn'] for pt.losses: builds inputs_embeds (image features merged in) and mRoPE position ids for the current micro-batch.
    ctx: feats (list of mx arrays, one per row), grid (B,3), lens (B,) = valid lengths of the (shifted) inputs."""
    lm = model.language_model; img, vid = model.config.image_token_index, model.config.video_token_index
    def fn(inputs):
        emb = lm.model.embed_tokens(inputs)
        emb, _ = type(model).merge_input_ids_with_image_features(mx.concatenate(ctx["feats"], 0), emb, inputs, img, vid)
        mask = (mx.arange(inputs.shape[1])[None, :] < ctx["lens"][:, None]).astype(mx.int32)
        pos, _ = lm.get_rope_index(inputs, ctx["grid"], None, mask)
        return dict(inputs_embeds=emb, position_ids=pos)
    return fn


def apply_lora(model, rank=16, scale=20.0, dropout=0.0):
    """LoRA on every linear layer of the language model's blocks; vision tower, embeddings and head stay frozen."""
    from mlx_lm.tuner.lora import LoRALinear
    model.freeze()
    for layer in model.language_model.model.layers:
        mods = [(n, LoRALinear.from_base(m, r=rank, scale=scale, dropout=dropout)) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]
        layer.update_modules(tree_unflatten(mods))


def generate(model, processor, rows, prompts, max_tokens, temp, chunk=16):
    """Batched sampling with images. Returns (texts, truncated_flags, stats dict). Truncated = re-encoded length reached the cap."""
    from mlx_vlm.generate.ar import batch_generate
    from mlx_vlm.sample_utils import make_sampler
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    sampler = make_sampler(temp=temp); texts = []; gtok = 0; gtime = 0.0; ptok = 0; ptime = 0.0
    model.eval()
    for i in range(0, len(rows), chunk):
        r = batch_generate(model, processor, images=[x["image"] for x in rows[i:i + chunk]], prompts=prompts[i:i + chunk], max_tokens=max_tokens, sampler=sampler, verbose=False)
        texts += r.texts; s = r.stats
        gtok += s.generation_tokens; gtime += s.generation_time; ptok += s.prompt_tokens; ptime += s.prompt_time
    trunc = [len(tok.encode(t)) >= max_tokens - 1 for t in texts]
    return texts, trunc, dict(gen_tps=gtok / max(gtime, 1e-9), prompt_tps=ptok / max(ptime, 1e-9), gen_tokens=gtok)


def evaluate(model, processor, cfg, rows, prep, max_tokens=640, chunk=25):
    t0 = time.time(); mx.reset_peak_memory()
    prompts = [prep[r["id"]]["prompt"] for r in rows]
    texts, trunc, st = generate(model, processor, rows, prompts, max_tokens, 0.0, chunk)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    recs = []
    for r, tx, tr in zip(rows, texts, trunc):
        c, _, f = reward_fn(tx, r["gold"], tr)
        recs.append(dict(id=r["id"], qtype=r["qtype"], gold=r["gold"], pred=parse_letter(tx), correct=int(c), fmt=int(f), truncated=int(tr), gen_tokens=len(tok.encode(tx)), response=tx))
    m = {"eval/accuracy": float(np.mean([x["correct"] for x in recs])), "eval/truncated_frac": float(np.mean([x["truncated"] for x in recs])),
         "eval/format_rate": float(np.mean([x["fmt"] for x in recs])), "eval/gen_tokens_mean": float(np.mean([x["gen_tokens"] for x in recs])),
         "eval/time_s": time.time() - t0, "eval/gen_tokens_per_s": st["gen_tps"], "eval/prompt_tokens_per_s": st["prompt_tps"], "eval/peak_mem_gb": mx.get_peak_memory() / 2**30}
    for q in ("direction", "which_object", "count"):
        xs = [x["correct"] for x in recs if x["qtype"] == q]
        if xs: m[f"eval/acc_{q}"] = float(np.mean(xs))
    return m, recs


def build_self_data(model, processor, cfg, rows, prep, path, k=4, per_q=2, max_new=640, temp=1.0):
    """Rejection sampling: k samples per question from the current model, keep up to per_q correct finished ones as SFT targets (cached)."""
    if os.path.exists(path): return [json.loads(l) for l in open(path)]
    out = []; tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor; t0 = time.time(); step = 4
    for i in range(0, len(rows), step):
        chunk = rows[i:i + step]; flat = [r for r in chunk for _ in range(k)]
        texts, trunc, _ = generate(model, processor, flat, [prep[r["id"]]["prompt"] for r in flat], max_new, temp, chunk=len(flat))
        for j, r in enumerate(chunk):
            good = [texts[j * k + g].strip() for g in range(k) if reward_fn(texts[j * k + g], r["gold"], trunc[j * k + g])[1] > 0][:per_q]
            for g in good: out.append(dict(r, target=g))
        mx.clear_cache()
        if (i // step) % 25 == 0: print(f"  self-distillation data: {i + step}/{len(rows)} questions, {len(out)} kept, {time.time() - t0:.0f}s", flush=True)
    with open(path, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    return out
