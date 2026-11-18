# Mac LLM/VLM Post-training Benchmarking

Can you post-train an LLM on a 16 GB MacBook? This project fine-tunes **Qwen3.5-0.8B** on an Apple M5 with [MLX](https://github.com/ml-explore/mlx)
using **SFT** and three critic-free RL methods (**GRPO, RLOO, GSPO**), with **LoRA** and with **full fine-tuning**, on two verifiable tasks
(GSM8K and a short-answer math-story dataset), and measures what each setup costs and gains (accuracy, speed, memory, system load, answer style).
Companion of [MacBook-Pro-LLM-VLM-Inference-Benchmarking](https://github.com/azminewasi/MacBook-Pro-LLM-VLM-Inference-Benchmarking) (same prompt and eval set, so base-model numbers carry over).

**Links:** [GitHub repo](https://github.com/azminewasi/Mac-LVLM-PostTrain-Benchmarking) · blog posts: [GSM8K](https://azminewasi.github.io/blog/mac-post-training-gsm8k.html), [math-story](https://azminewasi.github.io/blog/mac-post-training-math-story.html) · companion: [inference repo](https://github.com/azminewasi/MacBook-Pro-LLM-VLM-Inference-Benchmarking) and [blog post](https://azminewasi.github.io/blog/mac-llm-inference-benchmark.html) · reports: [`results/report.html`](results/report.html), [`results/report_mathstory.html`](results/report_mathstory.html) · research log: [`JOURNAL.md`](JOURNAL.md)

## Results (0.8B, final-epoch test accuracy, one seed)
| | untrained | SFT | SFT gentle | SFT self-distilled | GRPO | RLOO | GSPO |
|---|---|---|---|---|---|---|---|
| GSM8K, LoRA (n=100) | 0.54 | 0.22 | 0.37 | 0.54 | 0.64 | 0.67 | 0.59 |
| GSM8K, full fine-tuning | 0.54 | not run | not run | not run | 0.68 | 0.68 | 0.70 |
| math-story, LoRA (n=400) | 0.565 | 0.94 | 0.978 | 0.635 | 0.99 | 0.98 | 0.99 |

With 100 (GSM8K) or 400 (math-story) test prompts and one seed, small differences are noise (about ±10 points on GSM8K). What the data does support:
- SFT on human GSM8K solutions made the model *worse*; RL on its own answers improved it. On math-story, SFT on executable (equation-derived) solutions works.
- All three RL methods are indistinguishable; they also shorten the answers a lot, and silently drop formats nobody rewards.
- Memory is the first wall: a LoRA step peaked at 15.8 GB until gradient checkpointing (6.9 GB); full fine-tuning peaked at ~11 GB. RL runs took ~3 h each on GSM8K, 1-2 h on math-story.
- Not run: 2B/4B, SFT-full on GSM8K, full fine-tuning on math-story.

## Setup
- **Data:** GSM8K (500 random train questions, seed 0, saved in `data/`; the first 100 test questions, same prompt and scoring as the inference repo: greedy, thinking off, 512-token cap) and [`azminetoushikwasi/math-story-problems`](https://huggingface.co/datasets/azminetoushikwasi/math-story-problems) (500 train items in mixed story forms; test = 100 items in all four forms, 400 prompts). 5 epochs (2 for the gentle SFT variants).
- **Methods:** `sft` (reference solutions, lr 2e-5 LoRA / 1e-5 full); `sftgentle` (lr 5e-6, 2 epochs); `sftself` (rejection-sampled, the model's own correct answers); `grpo`, `rloo`, `gspo` (4 prompts x 4 rollouts per step, temperature 1.0, up to 640 new tokens, lr 2e-6 LoRA / 1e-6 full, no KL, no critic).
- **Reward (RL):** 1 if the answer finished and is correct, else 0. Format is logged, not rewarded (a format bonus plus truncation was reward-hacked in the first attempt, see the journal).
- **LoRA:** rank 16, scale 20, all linear layers. **Full fine-tuning:** all transformer blocks (embeddings frozen) with an fp32-master-weight AdamW (~14 bytes/param). Gradient checkpointing on.
- **Qwen3.5's gated-delta layers have no backward kernel in MLX**, so `pt/gdn.py` implements a chunk-parallel differentiable version (checked against the library reference in `tests/test_gdn.py`).
- Text-only: the vision tower is not loaded.

## What is logged
One W&B run per (model, method, adapter) named `<model>__<method>__<lora|full>`, in project `macbook-posttrain` (GSM8K) and `macbook-posttrain-mathstory`. Failed attempts are in group `discarded` and kept under `runs_discarded/`.
- **train/ time/ sys/ mem/** every step: loss, lr, grad norm, step time, CPU/GPU/RAM/swap, MLX peak memory; **reward/ completion/ policy/** for RL (reward, format rate, length, truncation, ratio, clip fraction); **eval/** after each epoch (accuracy, truncation, tokens, speed, per-form accuracy on math-story).
- Local data in `runs/<name>/` and `runs_mathstory/<name>/`: `config.json`, `metrics.jsonl`, every RL rollout (`rollouts.jsonl`), per-epoch evaluation answers (`eval_epochN.jsonl`).
- Model weights and adapters were deleted after the runs (about 5 GB); re-run `train.py` to regenerate them. Power draw (watts) was not measured.

## Layout
```
train.py            one run: --model --method --adapter --dataset {gsm8k,mathstory} (+ --epochs --lr --micro-bs --max-new ...)
pt/                 common (data, prompts, reward, eval), gdn (differentiable gated delta), losses, optim (MasterAdamW)
scripts/            run_queue.sh (the queue used), smoke_test.sh (tiny end-to-end check), watchdog_check.py
tests/              unit tests, gated-delta gradient check, memory profiler
analysis/, report.py  build results/report.html and results/report_mathstory.html (inline SVG, no plotting dependency)
data/               training sets and the self-distillation data
runs/, runs_mathstory/, runs_discarded/   per-run metrics, rollouts, evaluation answers
results/            comparison.csv, memory_profile.json, the two reports
```

## Run
```
source ../inference-testing/.venv/bin/activate && export WANDB_API_KEY=...   # the queue script reads ../inference-testing/.wandb_key
scripts/smoke_test.sh                              # optional, ~15 min
python train.py --model 0.8B --method grpo --adapter lora [--dataset mathstory --project macbook-posttrain-mathstory --out runs_mathstory]
scripts/run_queue.sh                               # the full queue; finished runs are skipped
python report.py                                   # regenerate both reports from the run folders
```
Never run two model jobs at once on 16 GB: the GPU memory limit is shared and the second job crashes the first.
