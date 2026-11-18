# Mac LLM/VLM Post-training Testing

Can you actually post-train LLMs on a 16 GB MacBook? This project fine-tunes **Qwen3.5-0.8B / 2B / 4B** on an Apple M5 with
[MLX](https://github.com/ml-explore/mlx) using **SFT** and three critic-free RL methods (**GRPO, RLOO, GSPO**), with **LoRA** on all
models and **full fine-tuning** on the 0.8B, then measures what each setup costs and gains (accuracy, speed, memory, system load).
It is the companion of *Mac LLM/VLM Inference Testing* (same models, prompt and eval set, so base-model numbers carry over).

**Status (2026-11-18):** the planned runs are complete. GSM8K (0.8B, untrained 0.54): LoRA SFT 0.22 / gentle 0.37 / self-distilled 0.54, LoRA GRPO 0.64 / RLOO 0.67 / GSPO 0.59, full fine-tuning GRPO 0.68 / RLOO 0.68 / GSPO 0.70. Math-story-problems (untrained 0.565): SFT 0.94 / gentle 0.978 / self-distilled 0.635, GRPO 0.99 / RLOO 0.98 / GSPO 0.99. One seed, 100 (GSM8K) or 400 (math-story) test prompts: small differences are not meaningful. Not run: 2B/4B, SFT-full on GSM8K, full fine-tuning on math-story.

**Datasets:** GSM8K (500 train / first 100 test) and, second, [`azminetoushikwasi/math-story-problems`](https://huggingface.co/datasets/azminetoushikwasi/math-story-problems) (short-answer arithmetic in four forms; `--dataset mathstory`, W&B project `macbook-posttrain-mathstory`, runs in `runs_mathstory/`). The default queue runs the eight 0.8B setups on each; 2B/4B live in `scripts/run_queue_larger.sh`.

**Math-story report:** `results/report_mathstory.html` (blog-style: where the untrained model fails, SFT vs RL, GSM8K vs math-story).

**Reports:** the blog-style write-up is generated at `results/report.html` (`python report.py`, regenerates from `runs/`), and **[`JOURNAL.md`](JOURNAL.md)** is the running research journal: what we tried, what happened, why, and what is still open (including the failed attempts, kept in `runs_discarded/`).

**Goals**
1. Find the practical limits of local post-training on 16 GB unified memory (which model/method/adapter combinations fit, how fast).
2. Compare SFT vs GRPO vs RLOO vs GSPO and LoRA vs full fine-tuning on one verifiable task (GSM8K), one W&B line per (model, method, adapter) pair.
3. Keep all training data, rollouts, adapters and metrics so every run can be analysed afterwards.

Everything is logged to W&B with **one run per (model, method, adapter) pair**,
named `<model>__<method>__<lora|full>` (e.g. `2B__gspo__lora`), so each pair is one line in every chart.

**Status: code written and unit-tested on a stub model (CPU); not yet run on real weights.** Run `scripts/smoke_test.sh` first. It is queued behind the inference benchmark, which must finish first so measurements are not disturbed.

## Setup
Task: GSM8K. Train = 500 random train samples (seed 0, saved to `data/train_gsm8k_n500_seed0.jsonl`), 5 epochs.
Eval = the same first-100 GSM8K test samples, prompt and scoring as `../inference-testing/bench.py` (thinking off,
greedy), measured before training (epoch 0) and after every epoch -> directly comparable with the inference benchmark.

| setup | models | notes |
|---|---|---|
| SFT + LoRA / full | LoRA: 0.8B, 2B, 4B; full: 0.8B | completion-only CE on cleaned GSM8K solutions ending `The answer is: N` |
| GRPO / RLOO / GSPO + LoRA / full | LoRA: 0.8B, 2B, 4B; full: 0.8B | 4 prompts x G=4 samples/step, temp 1.0, max 400 new tokens |

16 runs total (`scripts/run_queue.sh`). Reward = 1.0 if the final answer is correct + 0.1 format bonus; truncated = 0.
- **GRPO**: group-normalised advantage, token-level ratio + PPO clip, per-sequence token mean.
- **RLOO**: leave-one-out baseline (`r_i - mean of other samples`), plain REINFORCE.
- **GSPO**: group-normalised advantage, *sequence-level* ratio, tight clip (3e-4 / 4e-4).
- All RL variants are critic-free and reference-free (KL = 0): one model in memory. With `--mu 1` (default, one update per
  rollout) training is exactly on-policy, so clipping is inactive and variants differ in weighting; use `--mu 2+` to exercise clipping.
- **Full fine-tuning** trains all transformer blocks (embeddings / lm_head frozen) with an **fp32-master-weight AdamW**:
  the model stays bf16 but updates accumulate in fp32 (pure bf16 silently drops updates at fine-tuning LRs; see `tests/`).
- LoRA: rank 16, scale 20, all linear layers, all layers. LR: SFT 1e-4 (LoRA) / 1e-5 (full); RL 2e-5 (LoRA) / 2e-6 (full); cosine, 5% warmup, grad-clip 1.0.
- 4B (9.3 GB bf16) is tight on 16 GB: micro-batch 1 + gradient checkpointing by default. If it OOMs/swaps, add `--quantize-base 4` (QLoRA-style).
- Text-only (GSM8K) — the vision tower is not loaded for training.

## What is logged (every step unless noted)
- **train/**: loss, lr, grad_norm, tokens, tokens_per_s. **time/**: step_s, gen_s, train_s, elapsed_s. **progress/**: samples_seen, tokens. `epoch` (fractional) and `global_step` are in every row (use either as x-axis).
- **reward/** (RL): mean, std, correct_rate, format_rate, frac_zero_std_groups. **adv/abs_mean**, **completion/**: len_mean, len_max, truncated_frac. **policy/**: ratio_mean, clip_frac, neg_logp_per_tok, active_seqs. **gen/**: tokens_per_s, prompt_tokens_per_s, tokens.
- **sys/**: CPU %, GPU util %, RAM, swap (avg+max per step), thermal speed limit, process RSS. **mem/**: MLX peak + active GB.
- **eval/** (epoch 0..5): accuracy, truncated_frac, gen_tokens_mean, time_s, gen/prompt tokens per s, peak mem.
- **config**: model, method, adapter, all hyper-parameters, trainable params (and %), steps, load time, hardware.
- **summary**: baseline/final/best accuracy, gain, total train time, peak memory, steps, trainable params.
- **Kept data** (local in `runs/<name>/` and as a W&B artifact `<name>-data`): the 500 training samples (also a W&B table), `metrics.jsonl`,
  **every RL rollout** (`rollouts.jsonl`: response, reward, advantage, ...), per-epoch eval responses (`eval_epochN.jsonl`), LoRA adapters per epoch (full: final weights stay local only).
- Not available: watts (needs `sudo powermetrics`).

## Run
```
scripts/smoke_test.sh                       # ~tiny end-to-end check of every path (offline W&B); run first, after inference benchmark finishes
scripts/run_queue.sh --wait-for-inference   # starts all 16 runs once the inference benchmark has finished (../inference-testing/results/9B-4bit-think8x_summary.csv exists)
python train.py --model 2B --method gspo --adapter lora     # single run
python report.py                      # cross-run table + curves -> results/ and W&B run "00_comparison" (also runs at end of the queue)
```
Uses `../inference-testing/.venv` and the key in `../inference-testing/.wandb_key`. Finished runs are skipped (`runs/<name>/DONE`); a crashed run restarts from scratch (delete its folder first).
Knobs: `--epochs`, `--n-train`, `--group`, `--max-new`, `--mu`, `--lr`, `--micro-bs`, `--max-steps`.

## Expect it to be slow (unmeasured guesses)
Each run = 625 optimiser steps. RL steps are dominated by generation (16 samples x up to 400 tokens, batched). Rough guess: 0.8B runs a few hours
each, 2B ~half a day, 4B about a day -> the whole queue likely takes several days. Use the smoke test timings to extrapolate, then cut with
`--epochs`/`--max-steps` or run a subset if needed.
