"""Post-training on Apple Silicon (MLX): SFT / GRPO / RLOO / GSPO, with LoRA or full fine-tuning.
One W&B run per (model, method, adapter) pair, named  <model>__<method>__<lora|full>  e.g. 0.8B__grpo__lora.

  python train.py --model 0.8B --method sft  --adapter lora
  python train.py --model 0.8B --method grpo --adapter full      # full FT: 0.8B only (fp32 master weights)
"""
import argparse, json, math, os, platform, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint

import contextlib, mlx_lm.models.qwen3_5 as _q35
# Qwen3.5's fast gated-delta Metal kernel has no gradient: use the chunked differentiable path (pt/gdn.py) while building training graphs only.
from pt.gdn import make_train_update
_orig_gdu, _TRAIN = _q35.gated_delta_update, [False]
_train_gdu = make_train_update(_orig_gdu)
_q35.gated_delta_update = lambda *a, **k: (_train_gdu if _TRAIN[0] else _orig_gdu)(*a, **k)
@contextlib.contextmanager
def train_graph():
    _TRAIN[0] = True
    try: yield
    finally: _TRAIN[0] = False

from pt.common import (MODELS, MODEL_DEFAULTS, build_prompt_text, encode, end_token_id, load_train, load_eval,
                       reward_fn, generate_batch, evaluate, Monitor)
from pt.optim import lr_at, MasterAdamW
from pt.losses import group_advantages, sft_loss, rl_loss, token_logprobs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--method", required=True, choices=["sft", "grpo", "rloo", "gspo"])
ap.add_argument("--adapter", required=True, choices=["lora", "full"])
ap.add_argument("--n-train", type=int, default=500); ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--n-eval", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--project", default="macbook-posttrain"); ap.add_argument("--entity", default=None)
ap.add_argument("--wandb-mode", default="online"); ap.add_argument("--out", default="runs")
ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "mathstory"], help="gsm8k, or azminetoushikwasi/math-story-problems (train on mixed story forms, evaluate on all four forms)")
# optimisation
ap.add_argument("--lr", type=float, default=None); ap.add_argument("--batch", type=int, default=None, help="SFT: sequences/step; RL: prompts/step")
ap.add_argument("--micro-bs", type=int, default=None); ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--max-seq", type=int, default=896)
ap.add_argument("--clip-norm", type=float, default=1.0)
ap.add_argument("--lora-rank", type=int, default=16); ap.add_argument("--lora-scale", type=float, default=20.0); ap.add_argument("--lora-dropout", type=float, default=0.0)
ap.add_argument("--quantize-base", type=int, default=0, help="4/8 = QLoRA-style quantised base (LoRA only)")
# RL
ap.add_argument("--group", type=int, default=4, help="completions per prompt (G)"); ap.add_argument("--max-new", type=int, default=640)
ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--mu", type=int, default=1, help="gradient steps per rollout batch")
ap.add_argument("--eps-lo", type=float, default=None); ap.add_argument("--eps-hi", type=float, default=None)
ap.add_argument("--gen-batch", type=int, default=16)
ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
args = ap.parse_args()

RL = args.method != "sft"; LORA = args.adapter == "lora"
if not LORA and args.model != "0.8B": raise SystemExit("full fine-tuning is only configured for 0.8B on a 16 GB machine")
D = MODEL_DEFAULTS[args.model]
args.micro_bs = args.micro_bs or D["micro_bs"]; args.grad_ckpt = args.grad_ckpt or D["grad_ckpt"]
if args.lr is None:
    args.lr = (2e-5 if LORA else 1e-5) if args.method == "sft" else (2e-6 if LORA else 1e-6)
args.batch = args.batch or (4 if not RL else 4)
tight = {"gspo": (3e-4, 4e-4)}.get(args.method, (0.2, 0.2))
args.eps_lo = args.eps_lo or tight[0]; args.eps_hi = args.eps_hi or tight[1]

run_name = f"{args.model}__{args.method}__{args.adapter}"
out = f"{args.out}/{run_name}"; os.makedirs(out, exist_ok=True)
if os.path.exists(f"{out}/DONE"): print(f"{run_name} already done, skipping"); raise SystemExit(0)
mx.random.seed(args.seed); np.random.seed(args.seed)

# ---------------------------------------------------------------- model
t0 = time.time()
model, tok = load(MODELS[args.model]); load_s = time.time() - t0
if args.quantize_base:
    assert LORA, "quantised base only with LoRA"; nn.quantize(model, group_size=64, bits=args.quantize_base)
model.freeze()
if LORA:
    linear_to_lora_layers(model, len(model.layers), dict(rank=args.lora_rank, scale=args.lora_scale, dropout=args.lora_dropout))
else:
    for l in model.layers: l.unfreeze()          # all transformer blocks; embeddings / lm_head stay frozen
_layer_cls = type(model.layers[0]); _orig_call = _layer_cls.__call__
def set_ckpt(on):
    """Gradient checkpointing patches the layer class; only enable it for forward/backward (it breaks cached generation)."""
    _layer_cls.__call__ = _orig_call
    if on and args.grad_ckpt: grad_checkpoint(model.layers[0])
n_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
n_total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
mx.eval(model.parameters())
opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0) if LORA else MasterAdamW(model)
print(f"{run_name}: loaded {load_s:.1f}s, trainable {n_train_params/1e6:.1f}M / {n_total_params/1e6:.1f}M", flush=True)

# ---------------------------------------------------------------- data
train = load_train(args.n_train, args.seed, dataset=args.dataset); evalset = load_eval(args.n_eval, dataset=args.dataset)
END = end_token_id(tok)
def sft_example(r):
    p = encode(tok, build_prompt_text(tok, r["question"])); c = encode(tok, r["target"]) + [END]
    return p, c
steps_per_epoch = math.ceil(len(train) / args.batch)
total_steps = steps_per_epoch * args.epochs * (args.mu if RL else 1)
if args.max_steps: total_steps = min(total_steps, args.max_steps)

# ---------------------------------------------------------------- wandb
import wandb
cfg = dict(vars(args)) | dict(run_name=run_name, model_name=MODELS[args.model], model_size=args.model, trainable_params=n_train_params,
          total_params=n_total_params, trainable_pct=100 * n_train_params / n_total_params, steps_per_epoch=steps_per_epoch,
          total_steps=total_steps, load_time_s=load_s, chip="Apple M5", ram_gb=16, mlx=mx.__version__, os=platform.platform(),
          algorithm_notes="critic-free, reference-free (KL=0)" if RL else "completion-only cross-entropy",
          reward="1.0 if finished and correct, else 0 (format is logged, not rewarded)" if RL else None)
run = wandb.init(project=args.project, entity=args.entity, name=run_name, group=args.method, job_type=args.adapter,
                 tags=[args.model, args.method, args.adapter, args.dataset], config=cfg, mode=args.wandb_mode, dir=out)
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1, default=str)
mlog = open(f"{out}/metrics.jsonl", "a"); rlog = open(f"{out}/rollouts.jsonl", "a") if RL else None
mon = Monitor(); mon.start()
state = dict(step=0, epoch=0.0, tokens=0, samples=0, t_start=time.time())

def log(d, step):
    d = {**d, "epoch": state["epoch"], "global_step": step, "time/elapsed_s": time.time() - state["t_start"]}
    wandb.log(d, step=step); mlog.write(json.dumps(d, default=float) + "\n"); mlog.flush()

def run_eval(tag):
    set_ckpt(False)
    m, recs = evaluate(model, tok, evalset)
    with open(f"{out}/eval_{tag}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    log(m, state["step"]); print(f"[eval {tag}] acc={m['eval/accuracy']:.3f} trunc={m['eval/truncated_frac']:.2f} {m['eval/gen_tokens_per_s']:.0f} tok/s", flush=True)
    return m

def save(tag):
    p = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(f"{out}/{'adapter' if LORA else 'weights'}_{tag}.safetensors", p)
    if LORA: json.dump(dict(rank=args.lora_rank, scale=args.lora_scale, dropout=args.lora_dropout, base=MODELS[args.model]), open(f"{out}/adapter_config.json", "w"))

def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    return mx.array([s + [0] * (L - len(s)) for s in seqs])

def microbatches(items, size):
    for i in range(0, len(items), size): yield items[i:i + size]

def accumulate(a, b): return b if a is None else tree_map(lambda x, y: x + y, a, b)

def apply_update(grads, step):
    lr = lr_at(step, total_steps, args.lr)
    grads, gnorm = optim.clip_grad_norm(grads, args.clip_norm)
    if LORA: opt.learning_rate = lr; opt.update(model, grads); mx.eval(model.parameters(), opt.state)
    else: opt.update(model, grads, lr)
    return lr, float(gnorm)

# ---------------------------------------------------------------- baseline eval (epoch 0)
base = run_eval("epoch0")
best_acc, hist = base["eval/accuracy"], [base["eval/accuracy"]]

# ---------------------------------------------------------------- training
rng = np.random.RandomState(args.seed)
done = False
for epoch in range(args.epochs):
    order = rng.permutation(len(train))
    for bi in range(steps_per_epoch):
        if done: break
        rows = [train[i] for i in order[bi * args.batch:(bi + 1) * args.batch]]
        ts = time.perf_counter(); m = {}
        if not RL:
            set_ckpt(True)
            # ------------------------------------------------ SFT step
            ex = [sft_example(r) for r in rows]
            ex = [(p, c[: args.max_seq - len(p)]) for p, c in ex]
            ntok_total = sum(len(c) for _, c in ex); grads, tot = None, 0.0
            for mb in microbatches(ex, args.micro_bs):
                ids = pad_batch([p + c for p, c in mb]); pl = mx.array([len(p) for p, _ in mb]); tl = mx.array([len(p) + len(c) for p, c in mb])
                with train_graph(): lv, g = nn.value_and_grad(model, lambda *_: sft_loss(model, ids, pl, tl, ntok_total))(ids)
                grads = accumulate(grads, g); mx.eval(lv, grads); tot += lv.item()
            state["step"] += 1; lr, gn = apply_update(grads, state["step"] - 1)
            state["tokens"] += ntok_total; state["samples"] += len(rows)
            m = {"train/loss": tot, "train/lr": lr, "train/grad_norm": gn, "train/tokens": ntok_total,
                 "train/tokens_per_s": ntok_total / (time.perf_counter() - ts)}
        else:
            # ------------------------------------------------ RL step: sample G completions per prompt, score, update
            set_ckpt(False)
            G = args.group; prompts = [encode(tok, build_prompt_text(tok, r["question"])) for r in rows]
            flat_prompts = [p for p in prompts for _ in range(G)]
            tg = time.perf_counter()
            texts, toks, trunc, gstats = generate_batch(model, tok, flat_prompts, args.max_new, args.temp, batch_size=args.gen_batch)
            t_gen = time.perf_counter() - tg
            set_ckpt(True)
            R = [reward_fn(tx, rows[i // G]["gold"], tr) for i, (tx, tr) in enumerate(zip(texts, trunc))]
            rewards = np.array([r[0] for r in R]); adv = group_advantages(rewards, G, args.method)
            zero_groups = float(np.mean(rewards.reshape(-1, G).std(1) < 1e-8))
            seqs = []                                 # (prompt_ids, completion_ids(+eos if finished), advantage, idx)
            for i in range(len(flat_prompts)):
                c = list(toks[i]) + ([] if trunc[i] else [END])
                c = c[: max(1, args.max_seq - len(flat_prompts[i]))]
                seqs.append((flat_prompts[i], c, float(adv[i]), i))
            for i, (p, c, a, _) in enumerate(seqs):   # keep every rollout
                rlog.write(json.dumps(dict(step=state["step"] + 1, epoch=epoch + 1, prompt_id=rows[i // G]["id"], k=i % G, response=texts[i],
                           reward=float(rewards[i]), correct=R[i][1], fmt=R[i][2], advantage=a, truncated=int(trunc[i]), gen_tokens=len(toks[i])))+"\n")
            rlog.flush()
            active = [s for s in seqs if abs(s[2]) > 1e-8]       # zero-advantage sequences give zero gradient
            tt = time.perf_counter(); lr = float("nan"); gn = loss_v = 0.0; aux_acc = np.zeros(4)   # skipped step (no signal): loss/grad 0
            if active:
                old = None
                if args.mu > 1:                                    # behaviour log-probs for ratio / clipping
                    old = {}
                    for mb in microbatches(active, args.micro_bs):
                        ids = pad_batch([p + c for p, c, _, _ in mb]); pl = mx.array([len(p) for p, *_ in mb]); tl = mx.array([len(p) + len(c) for p, c, *_ in mb])
                        lp, _ = token_logprobs(model, ids, pl, tl); mx.eval(lp)
                        for j, s in enumerate(mb): old[s[3]] = np.array(lp[j])
                for _ in range(args.mu):
                    grads, loss_v, aux_acc = None, 0.0, np.zeros(4)
                    for mb in microbatches(active, args.micro_bs):
                        ids = pad_batch([p + c for p, c, _, _ in mb]); pl = mx.array([len(p) for p, *_ in mb]); tl = mx.array([len(p) + len(c) for p, c, *_ in mb])
                        a = mx.array([s[2] for s in mb], dtype=mx.float32)
                        ol = None
                        if old is not None:
                            s0 = max(0, min(len(p) for p, *_ in mb) - 1); L = ids.shape[1] - 1 - s0
                            ol = mx.array(np.stack([np.pad(old[s[3]][: L], (0, max(0, L - len(old[s[3]][: L])))) for s in mb]).astype(np.float32))
                        def f(*_): return rl_loss(model, ids, pl, tl, a, ol, args.method, args.eps_lo, args.eps_hi, len(active))
                        with train_graph(): (lv, aux), g = nn.value_and_grad(model, f)(ids)
                        grads = accumulate(grads, g); mx.eval(lv, aux, grads); loss_v += lv.item(); aux_acc += np.array(aux)
                    state["step"] += 1; lr, gn = apply_update(grads, state["step"] - 1)
            else:
                state["step"] += 1
            t_train = time.perf_counter() - tt
            ntok_gen = sum(len(t) for t in toks); state["tokens"] += ntok_gen; state["samples"] += len(seqs)
            m = {"train/loss": loss_v, "train/lr": lr, "train/grad_norm": gn,
                 "reward/mean": float(rewards.mean()), "reward/std": float(rewards.std()), "reward/correct_rate": float(np.mean([r[1] for r in R])),
                 "reward/format_rate": float(np.mean([r[2] > 0 for r in R])), "reward/frac_zero_std_groups": zero_groups,
                 "adv/abs_mean": float(np.abs(adv).mean()), "completion/len_mean": ntok_gen / len(toks), "completion/len_max": max(len(t) for t in toks),
                 "completion/truncated_frac": float(np.mean(trunc)), "policy/active_seqs": len(active),
                 "policy/ratio_mean": float(aux_acc[0] / max(1, math.ceil(len(active) / args.micro_bs))) if active else float("nan"),
                 "policy/clip_frac": float(aux_acc[1] / max(1, len(active))) if active else float("nan"),
                 "policy/neg_logp_per_tok": float(aux_acc[3] / max(1, aux_acc[2])) if active else float("nan"),
                 "time/gen_s": t_gen, "time/train_s": t_train, "gen/tokens_per_s": gstats.generation_tps, "gen/prompt_tokens_per_s": gstats.prompt_tps,
                 "gen/tokens": ntok_gen, "train/tokens_per_s": sum(len(s[0]) + len(s[1]) for s in active) / max(t_train, 1e-9)}
        state["epoch"] = epoch + (bi + 1) / steps_per_epoch
        m |= {"time/step_s": time.perf_counter() - ts, "progress/samples_seen": state["samples"], "progress/tokens": state["tokens"]} | mon.drain()
        log(m, state["step"])
        if state["step"] % 5 == 0 or bi == 0:
            print(f"{run_name} ep{epoch+1} step {state['step']}/{total_steps} " + " ".join(f"{k.split('/')[-1]}={v:.3g}" for k, v in m.items() if k in ("train/loss", "reward/mean", "train/grad_norm", "time/step_s") and isinstance(v, float)), flush=True)
        if args.max_steps and state["step"] >= args.max_steps: done = True
    ev = run_eval(f"epoch{epoch+1}"); hist.append(ev["eval/accuracy"])
    if LORA or epoch + 1 == args.epochs: save(f"epoch{epoch+1}")
    best_acc = max(best_acc, ev["eval/accuracy"])
    if done: break

# ---------------------------------------------------------------- finish: summary + keep all data in W&B
wandb.summary.update({"final/eval_accuracy": hist[-1], "final/best_eval_accuracy": best_acc, "final/baseline_eval_accuracy": hist[0],
                      "final/accuracy_gain": hist[-1] - hist[0], "final/train_time_s": time.time() - state["t_start"],
                      "final/total_gen_or_train_tokens": state["tokens"], "final/peak_mem_gb": mx.get_peak_memory() / 2**30,
                      "final/steps": state["step"], "final/trainable_params": n_train_params})
tr_tab = wandb.Table(columns=["id", "source_index", "question", "target", "gold"], data=[[r["id"], r["source_index"], r["question"], r["target"], r["gold"]] for r in train])
wandb.log({"data/training_set": tr_tab}, step=state["step"])
art = wandb.Artifact(f"{run_name}-data", type="run-data")
for fn in os.listdir(out):
    if fn.endswith((".jsonl", ".json")) or (LORA and fn.endswith(".safetensors")): art.add_file(f"{out}/{fn}")
art.add_file(f"data/train_{'gsm8k' if args.dataset == 'gsm8k' else 'mathstory'}_n{args.n_train}_seed{args.seed}.jsonl")
run.log_artifact(art)
mon.stop(); run.finish(); open(f"{out}/DONE", "w").write(time.strftime("%F %T"))
print(f"{run_name} DONE: acc {hist[0]:.3f} -> {hist[-1]:.3f} (best {best_acc:.3f})")
