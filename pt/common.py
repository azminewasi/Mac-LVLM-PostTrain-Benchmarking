"""Shared helpers: prompts, data, scoring, system monitor, batched generation, eval."""
import json, os, random, re, subprocess, threading, time
import numpy as np, psutil
import mlx.core as mx

MODELS = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B", "4B": "Qwen/Qwen3.5-4B"}
# per-model memory defaults for a 16 GB machine (micro-batch = sequences per forward/backward pass)
MODEL_DEFAULTS = {"0.8B": dict(micro_bs=2, grad_ckpt=False), "2B": dict(micro_bs=2, grad_ckpt=True),
                  "4B": dict(micro_bs=1, grad_ckpt=True)}
# Same prompt/scoring as inference-testing/bench.py so numbers are directly comparable.
SUFFIX = "\nSolve step by step. End with a final line: 'The answer is: <number>'."


# ------------------------------------------------------------------ prompts / data
def build_prompt_text(tokenizer, question):
    msgs = [{"role": "user", "content": question + SUFFIX}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

def encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))

def end_token_id(tokenizer):
    t = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return t if isinstance(t, int) and t >= 0 else tokenizer.eos_token_id

def clean_solution(answer):
    body, final = answer.split("####")
    body = re.sub(r"<<.*?>>", "", body).strip()
    final = final.strip().replace(",", "")
    return f"{body}\nThe answer is: {final}", final

def load_train(n, seed, out_dir="data"):
    """n random GSM8K *train* samples (fixed seed). Saved to disk so the exact data is kept."""
    path = f"{out_dir}/train_gsm8k_n{n}_seed{seed}.jsonl"
    if os.path.exists(path):
        return [json.loads(l) for l in open(path)]
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = random.Random(seed).sample(range(len(ds)), n)
    rows = []
    for i, j in enumerate(idx):
        target, gold = clean_solution(ds[j]["answer"])
        rows.append(dict(id=i, source_index=j, question=ds[j]["question"], target=target, gold=gold))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return rows

def load_eval(n=100):
    """First n GSM8K *test* samples, identical to the inference benchmark."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [dict(id=i, question=r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", "")) for i, r in enumerate(ds)]


# ------------------------------------------------------------------ scoring
def extract_answer(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None

def is_correct(text, gold):
    p = extract_answer(text)
    try: return p is not None and abs(float(p) - float(gold)) < 1e-4
    except ValueError: return False

def has_format(text):
    return bool(re.search(r"The answer is:\s*-?\$?\d", text))

def reward_fn(text, gold, truncated):
    """correct (1.0) + small format bonus (0.1, only for completed generations). Returns (total, correct, fmt)."""
    c = 1.0 if (not truncated and is_correct(text, gold)) else 0.0
    f = 0.1 if (not truncated and has_format(text)) else 0.0
    return c + f, c, f


# ------------------------------------------------------------------ system monitor
def _ioreg_gpu():
    try:
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', o)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def thermal_limit():
    try:
        o = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", o)
        return int(m.group(1)) if m else 100
    except Exception:
        return None

class Monitor(threading.Thread):
    """Samples CPU / RAM / swap / GPU util every `period` s; drain() returns aggregates since last drain."""
    def __init__(self, period=1.0):
        super().__init__(daemon=True); self.period, self.s, self._stop_flag, self.lock = period, [], False, threading.Lock()
    def run(self):
        psutil.cpu_percent(None)
        while not self._stop_flag:
            time.sleep(self.period)
            r = dict(cpu=psutil.cpu_percent(None), ram=psutil.virtual_memory().used / 2**30,
                     swap=psutil.swap_memory().used / 2**30, gpu=_ioreg_gpu())
            with self.lock: self.s.append(r)
    def drain(self):
        with self.lock: s, self.s = self.s, []
        out = {}
        for k, name in (("cpu", "cpu_pct"), ("gpu", "gpu_util_pct"), ("ram", "ram_used_gb"), ("swap", "swap_gb")):
            v = [x[k] for x in s if x[k] is not None]
            if v: out[f"sys/{name}_avg"], out[f"sys/{name}_max"] = float(np.mean(v)), float(np.max(v))
        out["sys/thermal_speed_limit"] = thermal_limit()
        out["sys/rss_gb"] = psutil.Process().memory_info().rss / 2**30
        out["mem/peak_gb"] = mx.get_peak_memory() / 2**30
        out["mem/active_gb"] = mx.get_active_memory() / 2**30
        return out
    def stop(self): self._stop_flag = True


# ------------------------------------------------------------------ generation / eval
def generate_batch(model, tokenizer, prompts_ids, max_tokens, temp, top_p=0.0, batch_size=16):
    """Batched sampling. Returns (texts, token_id_lists, truncated_flags, stats). Stop token is NOT in token_ids."""
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temp, top_p=top_p)
    r = batch_generate(model, tokenizer, prompts_ids, max_tokens=max_tokens, sampler=sampler,
                       completion_batch_size=batch_size, prefill_batch_size=min(8, batch_size),
                       return_token_ids=True, verbose=False)
    trunc = [len(t) >= max_tokens for t in r.token_ids]
    return r.texts, r.token_ids, trunc, r.stats

def evaluate(model, tokenizer, eval_rows, max_tokens=512, batch_size=25):
    """Greedy, thinking-off GSM8K accuracy on the benchmark's 100 test samples."""
    prompts = [encode(tokenizer, build_prompt_text(tokenizer, r["question"])) for r in eval_rows]
    t0 = time.perf_counter()
    texts, toks, trunc, stats = generate_batch(model, tokenizer, prompts, max_tokens, temp=0.0, batch_size=batch_size)
    dt = time.perf_counter() - t0
    recs, ok = [], 0
    for r, tx, tk, tr in zip(eval_rows, texts, toks, trunc):
        c = int((not tr) and is_correct(tx, r["gold"])); ok += c
        recs.append(dict(id=r["id"], gold=r["gold"], pred=extract_answer(tx), correct=c, truncated=int(tr), gen_tokens=len(tk), response=tx))
    n = len(eval_rows)
    m = {"eval/accuracy": ok / n, "eval/truncated_frac": float(np.mean([x["truncated"] for x in recs])),
         "eval/gen_tokens_mean": float(np.mean([x["gen_tokens"] for x in recs])), "eval/time_s": dt,
         "eval/gen_tokens_per_s": stats.generation_tps, "eval/prompt_tokens_per_s": stats.prompt_tps,
         "eval/peak_mem_gb": stats.peak_memory}
    return m, recs
