"""Visual question answering post-training on Apple Silicon (MLX): SFT / GRPO / RLOO / GSPO with LoRA on Qwen3.5-0.8B (image in, text answer out).
Task: SpatialEval spatial-map, 4-option multiple choice on a map image (500 train / 100 eval questions, images at 448 px). The vision tower is frozen
(its features are computed once and cached); LoRA trains the language model only.

  python train_vision.py --method sft  --project macbook-posttrain-spatial --out runs_spatial
  python train_vision.py --method grpo --project macbook-posttrain-spatial --out runs_spatial
"""
import argparse, json, math, os, platform, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_vlm import load
from mlx_vlm.utils import load_config
import pt.losses as L
from pt.common import end_token_id, Monitor
from pt.optim import lr_at
from pt.losses import group_advantages, sft_loss, rl_loss, token_logprobs
import pt.vision as V

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["sft", "sftgentle", "sftself", "grpo", "rloo", "gspo"])
ap.add_argument("--n-train", type=int, default=500); ap.add_argument("--n-eval", type=int, default=100); ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--project", default="macbook-posttrain-spatial"); ap.add_argument("--entity", default=None)
ap.add_argument("--wandb-mode", default="online"); ap.add_argument("--out", default="runs_spatial")
ap.add_argument("--lr", type=float, default=None); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--max-seq", type=int, default=1100); ap.add_argument("--clip-norm", type=float, default=1.0)
ap.add_argument("--lora-rank", type=int, default=16); ap.add_argument("--lora-scale", type=float, default=20.0)
ap.add_argument("--group", type=int, default=4); ap.add_argument("--max-new", type=int, default=640); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--mu", type=int, default=1); ap.add_argument("--eps-lo", type=float, default=None); ap.add_argument("--eps-hi", type=float, default=None)
ap.add_argument("--no-grad-ckpt", action="store_true"); ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
args = ap.parse_args()

RL = args.method in ("grpo", "rloo", "gspo")
if args.epochs is None: args.epochs = 2 if args.method in ("sftgentle", "sftself") else 5
if args.lr is None: args.lr = (2e-5 if args.method == "sft" else 5e-6) if not RL else 2e-6
tight = {"gspo": (3e-4, 4e-4)}.get(args.method, (0.2, 0.2)); args.eps_lo = args.eps_lo or tight[0]; args.eps_hi = args.eps_hi or tight[1]
run_name = f"0.8B__{args.method}__lora"; out = f"{args.out}/{run_name}"; os.makedirs(out, exist_ok=True)
if os.path.exists(f"{out}/DONE"): print(f"{run_name} already done, skipping"); raise SystemExit(0)
mx.random.seed(args.seed); np.random.seed(args.seed)

# ---------------------------------------------------------------- model + data
t0 = time.time(); model, processor = load(V.MODEL_ID); cfg = load_config(V.MODEL_ID); load_s = time.time() - t0
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
V.apply_lora(model, args.lora_rank, args.lora_scale)
layers = model.language_model.model.layers; _layer_cls = type(layers[0]); _orig_call = _layer_cls.__call__
def set_ckpt(on):
    _layer_cls.__call__ = _orig_call
    if on and not args.no_grad_ckpt: grad_checkpoint(layers[0])
n_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters())); n_total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
mx.eval(model.parameters()); opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
train, evalset = V.load_spatial(args.n_train, args.n_eval, args.seed)
prep = V.prep_rows(model, processor, cfg, train + evalset)
ctx = {}; L.EXTRA["fn"] = V.make_fwd(model, ctx)
END = end_token_id(tok)
print(f"{run_name}: loaded {load_s:.1f}s, trainable {n_train_params/1e6:.1f}M / {n_total_params/1e6:.1f}M, {len(train)} train / {len(evalset)} eval, prompt {len(prep[train[0]['id']]['ids'])} tokens", flush=True)
if args.method == "sftself":
    train = V.build_self_data(model, processor, cfg, train, prep, f"data/self_spatial_n{args.n_train}_seed{args.seed}_k4.jsonl", k=4, per_q=2, max_new=args.max_new, temp=1.0)
steps_per_epoch = math.ceil(len(train) / args.batch); total_steps = steps_per_epoch * args.epochs * (args.mu if RL else 1)
if args.max_steps: total_steps = min(total_steps, args.max_steps)

# ---------------------------------------------------------------- wandb
import wandb
cfg_log = dict(vars(args)) | dict(run_name=run_name, model_name=V.MODEL_ID, dataset="spatialeval-spatialmap", image_size=V.SIZE, trainable_params=n_train_params, total_params=n_total_params,
              trainable_pct=100 * n_train_params / n_total_params, steps_per_epoch=steps_per_epoch, total_steps=total_steps, load_time_s=load_s, chip="Apple M5", ram_gb=16,
              mlx=mx.__version__, os=platform.platform(), algorithm_notes="vision tower frozen (features cached); LoRA on the language model; " + ("critic-free, reference-free (KL=0)" if RL else "completion-only cross-entropy"),
              reward="1.0 if finished and the final 'The answer is: X' is the gold option, else 0 (format is logged, not rewarded)" if RL else None)
run = wandb.init(project=args.project, entity=args.entity, name=run_name, group=args.method, job_type="lora", tags=["0.8B", args.method, "lora", "spatial", "vision"], config=cfg_log, mode=args.wandb_mode, dir=out)
json.dump(cfg_log, open(f"{out}/config.json", "w"), indent=1, default=str)
mlog = open(f"{out}/metrics.jsonl", "a"); rlog = open(f"{out}/rollouts.jsonl", "a") if RL else None
mon = Monitor(); mon.start(); state = dict(step=0, epoch=0.0, tokens=0, samples=0, t_start=time.time())
def log(d, step):
    d = {**d, "epoch": state["epoch"], "global_step": step, "time/elapsed_s": time.time() - state["t_start"]}
    wandb.log(d, step=step); mlog.write(json.dumps(d, default=float) + "\n"); mlog.flush()
def run_eval(tag):
    set_ckpt(False); m, recs = V.evaluate(model, processor, cfg, evalset, prep, max_tokens=args.max_new)
    with open(f"{out}/eval_{tag}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    log(m, state["step"]); print(f"[eval {tag}] acc={m['eval/accuracy']:.3f} trunc={m['eval/truncated_frac']:.2f} fmt={m['eval/format_rate']:.2f} {m['eval/gen_tokens_per_s']:.0f} tok/s", flush=True); return m
def save(tag): mx.save_safetensors(f"{out}/adapter_{tag}.safetensors", dict(tree_flatten(model.trainable_parameters())))
def pad_batch(seqs): n = max(len(s) for s in seqs); return mx.array([s + [0] * (n - len(s)) for s in seqs])
def microbatches(items, size):
    for i in range(0, len(items), size): yield items[i:i + size]
def accumulate(a, b): return b if a is None else tree_map(lambda x, y: x + y, a, b)
def set_ctx(rids, tl):
    ctx["feats"] = [prep[r]["feats"] for r in rids]; ctx["grid"] = mx.stack([prep[r]["grid"] for r in rids]); ctx["lens"] = tl - 1
def apply_update(grads, step):
    lr = lr_at(step, total_steps, args.lr); grads, gnorm = optim.clip_grad_norm(grads, args.clip_norm)
    opt.learning_rate = lr; opt.update(model, grads); mx.eval(model.parameters(), opt.state); return lr, float(gnorm)
def batch_of(mb):   # mb: list of (rid, prompt_ids, completion_ids, adv)
    ids = pad_batch([p + c for _, p, c, _ in mb]); pl = mx.array([len(p) for _, p, _, _ in mb]); tl = mx.array([len(p) + len(c) for _, p, c, _ in mb])
    set_ctx([r for r, *_ in mb], tl); return ids, pl, tl

base = run_eval("epoch0"); best_acc, hist = base["eval/accuracy"], [base["eval/accuracy"]]
rng = np.random.RandomState(args.seed); done = False
for epoch in range(args.epochs):
    order = rng.permutation(len(train))
    for bi in range(steps_per_epoch):
        if done: break
        rows = [train[i] for i in order[bi * args.batch:(bi + 1) * args.batch]]; ts = time.perf_counter(); m = {}
        if not RL:
            set_ckpt(True); model.train()
            ex = []
            for r in rows:
                p = prep[r["id"]]["ids"]; c = tok.encode(r["target"], add_special_tokens=False) + [END]; ex.append((r["id"], p, c[: args.max_seq - len(p)], 0.0))
            ntok_total = sum(len(c) for _, _, c, _ in ex); grads, tot = None, 0.0
            for mb in microbatches(ex, args.micro_bs):
                ids, pl, tl = batch_of(mb)
                lv, g = nn.value_and_grad(model, lambda *_: sft_loss(model, ids, pl, tl, ntok_total))(ids)
                grads = accumulate(grads, g); mx.eval(lv, grads); tot += lv.item()
            state["step"] += 1; lr, gn = apply_update(grads, state["step"] - 1); state["tokens"] += ntok_total; state["samples"] += len(rows)
            m = {"train/loss": tot, "train/lr": lr, "train/grad_norm": gn, "train/tokens": ntok_total, "train/tokens_per_s": ntok_total / (time.perf_counter() - ts)}
        else:
            set_ckpt(False); G = args.group
            flat = [r for r in rows for _ in range(G)]; tg = time.perf_counter()
            texts, trunc, gst = V.generate(model, processor, flat, [prep[r["id"]]["prompt"] for r in flat], args.max_new, args.temp, chunk=len(flat))
            t_gen = time.perf_counter() - tg; set_ckpt(True)
            R = [V.reward_fn(tx, flat[i]["gold"], tr) for i, (tx, tr) in enumerate(zip(texts, trunc))]
            rewards = np.array([x[0] for x in R]); adv = group_advantages(rewards, G, args.method); zero_groups = float(np.mean(rewards.reshape(-1, G).std(1) < 1e-8))
            toks = [tok.encode(t, add_special_tokens=False) for t in texts]; seqs = []
            for i, r in enumerate(flat):
                c = toks[i] + ([] if trunc[i] else [END]); p = prep[r["id"]]["ids"]; seqs.append((r["id"], p, c[: max(1, args.max_seq - len(p))], float(adv[i]), i))
                rlog.write(json.dumps(dict(step=state["step"] + 1, epoch=epoch + 1, prompt_id=r["id"], k=i % G, response=texts[i], reward=float(rewards[i]), correct=R[i][1], fmt=R[i][2],
                                           advantage=float(adv[i]), truncated=int(trunc[i]), gen_tokens=len(toks[i]))) + "\n")
            rlog.flush(); active = [s for s in seqs if abs(s[3]) > 1e-8]
            tt = time.perf_counter(); lr = float("nan"); gn = loss_v = 0.0; aux_acc = np.zeros(4)
            if active:
                model.train(); old = None
                if args.mu > 1:
                    old = {}
                    for mb in microbatches(active, args.micro_bs):
                        ids, pl, tl = batch_of([s[:4] for s in mb]); lp, _ = token_logprobs(model, ids, pl, tl); mx.eval(lp)
                        for j, s in enumerate(mb): old[s[4]] = np.array(lp[j])
                for _ in range(args.mu):
                    grads, loss_v, aux_acc = None, 0.0, np.zeros(4)
                    for mb in microbatches(active, args.micro_bs):
                        ids, pl, tl = batch_of([s[:4] for s in mb]); a = mx.array([s[3] for s in mb], dtype=mx.float32); ol = None
                        if old is not None:
                            s0 = max(0, min(len(s[1]) for s in mb) - 1); Lh = ids.shape[1] - 1 - s0
                            ol = mx.array(np.stack([np.pad(old[s[4]][:Lh], (0, max(0, Lh - len(old[s[4]][:Lh])))) for s in mb]).astype(np.float32))
                        def f(*_): return rl_loss(model, ids, pl, tl, a, ol, args.method, args.eps_lo, args.eps_hi, len(active))
                        (lv, aux), g = nn.value_and_grad(model, f)(ids)
                        grads = accumulate(grads, g); mx.eval(lv, aux, grads); loss_v += lv.item(); aux_acc += np.array(aux)
                    state["step"] += 1; lr, gn = apply_update(grads, state["step"] - 1)
            else: state["step"] += 1
            t_train = time.perf_counter() - tt; ntok_gen = sum(len(t) for t in toks); state["tokens"] += ntok_gen; state["samples"] += len(seqs)
            m = {"train/loss": loss_v, "train/lr": lr, "train/grad_norm": gn, "reward/mean": float(rewards.mean()), "reward/std": float(rewards.std()),
                 "reward/correct_rate": float(np.mean([x[1] for x in R])), "reward/format_rate": float(np.mean([x[2] > 0 for x in R])), "reward/frac_zero_std_groups": zero_groups,
                 "adv/abs_mean": float(np.abs(adv).mean()), "completion/len_mean": ntok_gen / len(toks), "completion/len_max": max(len(t) for t in toks),
                 "completion/truncated_frac": float(np.mean(trunc)), "policy/active_seqs": len(active),
                 "policy/ratio_mean": float(aux_acc[0] / max(1, math.ceil(len(active) / args.micro_bs))) if active else float("nan"),
                 "policy/clip_frac": float(aux_acc[1] / max(1, len(active))) if active else float("nan"),
                 "policy/neg_logp_per_tok": float(aux_acc[3] / max(1, aux_acc[2])) if active else float("nan"),
                 "time/gen_s": t_gen, "time/train_s": t_train, "gen/tokens_per_s": gst["gen_tps"], "gen/prompt_tokens_per_s": gst["prompt_tps"], "gen/tokens": ntok_gen,
                 "train/tokens_per_s": sum(len(s[1]) + len(s[2]) for s in active) / max(t_train, 1e-9)}
        state["epoch"] = epoch + (bi + 1) / steps_per_epoch
        m |= {"time/step_s": time.perf_counter() - ts, "progress/samples_seen": state["samples"], "progress/tokens": state["tokens"], "mem/peak_gb": mx.get_peak_memory() / 2**30} | mon.drain()
        log(m, state["step"])
        if state["step"] % 5 == 0 or bi == 0:
            print(f"{run_name} ep{epoch+1} step {state['step']}/{total_steps} " + " ".join(f"{k.split('/')[-1]}={v:.3g}" for k, v in m.items() if k in ("train/loss", "reward/mean", "train/grad_norm", "time/step_s", "mem/peak_gb") and isinstance(v, float)), flush=True)
        if args.max_steps and state["step"] >= args.max_steps: done = True
    ev = run_eval(f"epoch{epoch+1}"); hist.append(ev["eval/accuracy"]); best_acc = max(best_acc, ev["eval/accuracy"])
    if epoch + 1 == args.epochs or done: save(f"epoch{epoch+1}")
    if done: break

wandb.summary.update({"final/eval_accuracy": hist[-1], "final/best_eval_accuracy": best_acc, "final/baseline_eval_accuracy": hist[0], "final/accuracy_gain": hist[-1] - hist[0],
                      "final/train_time_s": time.time() - state["t_start"], "final/total_gen_or_train_tokens": state["tokens"], "final/peak_mem_gb": mx.get_peak_memory() / 2**30,
                      "final/steps": state["step"], "final/trainable_params": n_train_params})
art = wandb.Artifact(f"{run_name}-data", type="run-data")
for fn in os.listdir(out):
    if fn.endswith((".jsonl", ".json")): art.add_file(f"{out}/{fn}")
for fn in (f"data/train_spatial_n{args.n_train}_seed{args.seed}.jsonl", f"data/eval_spatial_n{args.n_eval}_seed{args.seed}.jsonl"): art.add_file(fn)
if args.method == "sftself": art.add_file(f"data/self_spatial_n{args.n_train}_seed{args.seed}_k4.jsonl")
run.log_artifact(art); mon.stop(); run.finish(); open(f"{out}/DONE", "w").write(time.strftime("%F %T"))
print(f"{run_name} DONE: acc {hist[0]:.3f} -> {hist[-1]:.3f} (best {best_acc:.3f})")
