# Research journal: Mac LLM/VLM Post-training Testing

A running, dated log of experiments, surprises, decisions and open questions. Newest entries go at the bottom of the log.
Each entry: **what we tried → what happened → why we think so (and how sure we are) → what we decided**.
Times are local and approximate where marked `~`. Run data referred to here lives in `runs/` (kept) and `runs_discarded/` (failed attempts, kept as evidence); the W&B project is `macbook-posttrain` (failed attempts are in group `discarded`).

## Standing questions
| # | Question | Status |
|---|---|---|
| Q1 | Which of SFT / GRPO / RLOO / GSPO improves GSM8K accuracy most on a 0.8B model, with LoRA vs full fine-tuning? | LoRA arm answered, within noise: SFT 0.22, GRPO 0.64, RLOO 0.67, GSPO 0.59 (baseline 0.54): the three RL methods are indistinguishable and all beat SFT; full fine-tuning not run |
| Q2 | Do the 2B and 4B fit in 16 GB for training? | open; 4B likely does not fit as configured (see 2026-11-17 ~11:00) |
| Q3 | Does GRPO's drop in "The answer is:" format hurt anything downstream? | open; no effect on the scored accuracy so far |
| Q4 | Would SFT help with more data or a lower learning rate? | untested (we ran one recipe) |
| Q6 | On short-answer arithmetic (math-story-problems) do the same methods help, and which question forms (eq / story_1 / story_2 / story_3) improve? | open: code ready and smoke-tested, runs not started |

## Log

### 2026-11-17 ~08:00 · Smoke test, and the gradient wall
- **Tried:** 6-setup smoke test (8 training samples) of SFT/GRPO/RLOO/GSPO with LoRA and full fine-tuning on the 0.8B.
- **Observed:** first step crashed: `Primitive::vjp not implemented for CustomKernel`. Qwen3.5's gated-delta layers use a Metal kernel with no backward pass. The library's ops-based fallback worked but took **50 s per step** (Python loop over tokens).
- **Why:** hybrid architecture; the fast path is inference-only. (Confirmed by reading `mlx_lm/models/gated_delta.py`.)
- **Decision:** write a chunk-parallel, matmul-based differentiable version (`pt/gdn.py`).

### 2026-11-17 ~08:10 · Chunked gated-delta, two versions
- **v1 (Neumann-series triangular inverse):** passed a random-input unit test, loss became NaN in the model, then the layer outputs were wrong (errors up to 1e14 in later layers).
- **Why:** real keys are highly correlated and decay is near 1, so powers of the strictly-lower matrix blow up. Random test inputs hid this. (Confirmed layer by layer against the reference.)
- **v2 (forward substitution, chunk 32):** matches the reference on the real model (logit diff 0.32, same as kernel-vs-ops; argmax agreement 100%), gradients within 4% relative L2, step time 0.6-1.6 s.
- **Lesson:** test numerical code on real activations, not only random ones.

### 2026-11-17 ~08:20 · Smoke-test oddities
- Mean reward 0 in smoke RL runs (64-token cap) gave NaN loss in logging: all groups had zero variance, so the step was skipped. Cosmetic; now logs loss 0.
- A `--mu 2` path crashed (`mx.array` of a list of numpy arrays): fixed. The smoke test never reached it because rewards were 0, so we added a second check with a 384-token cap.
- A learning-rate-0 run reproduced baseline accuracy exactly: generation after training is correct. The accuracy drops seen on 4-8 sample smoke evals were noise.

### 2026-11-17 08:17 · Queue bug
- The queue passed an empty argument (`"${EXTRA[@]}"` in zsh) and every run failed instantly. Fixed by dropping the quotes.

### 2026-11-17 08:40 · SFT-LoRA at lr 1e-4 collapses
- **Observed:** 0.8B SFT LoRA, lr 1e-4: accuracy 0.54 → 0.17 after epoch 1, → 0.04 at the end; outputs degenerate to repeated "000…".
- **Why:** lr ~10x too hot for LoRA scale 20 (the mlx-lm default is 1e-5). Confirmed by rerunning at 2e-5 (no collapse).
- **Decision:** SFT lr 2e-5 (LoRA), 1e-5 (full); RL 5e-6 / 2e-6 (later lowered).

### 2026-11-17 08:56 · First finished run: SFT-LoRA 0.8B (lr 2e-5)
- Accuracy 0.54 (baseline) → 0.17 / 0.29 / 0.27 / 0.24 / 0.22 over 5 epochs; training loss → 0.01 (memorisation of the 500 solutions).
- **Qualitative:** mean answer length 347 → ~90 tokens; markdown 95% → 0%, LaTeX 80% → 0%; "The answer is:" 86% → 99%. In epoch 1, 22% of answers degenerate into a repeating arithmetic loop to the token cap.
- **Why (hypothesis, not tested):** Qwen3.5 already reasons well in a long step-by-step style; imitating 500 terse human solutions replaces that with a shorter style it executes worse. **Decision:** keep as a result; RL is the intended comparison.

### 2026-11-17 09:48 · GRPO v1: reward hacking
- **Observed:** accuracy 0.54 → 0.02 after epoch 1. Within 50 steps completion length fell 288 → 35 tokens and the mean reward settled at exactly 0.1 (the format bonus alone).
- **Why (confirmed from curves):** `max_new=400` truncated 44% of rollouts (reward 0), while a short, wrong, well-formatted answer scored 0.1. GRPO's group normalisation amplified that gap.
- **Decision:** reward = correctness only (format logged, not rewarded); `max_new` 640; `max_seq` 896. Discarded run kept as `runs_discarded/…v1-reward-hack` (W&B: `DISCARDED-v1`).

### 2026-11-17 ~10:20 · GRPO v2: drift at lr 5e-6
- **Observed:** after the fix, reward 0.42 → 0.1 and format rate 0.89 → 0.39 between steps 25 and 50; zero-signal groups up to 80%. Length no longer collapsed.
- **Checked first:** gradients of the chunked layer vs the reference on the real model (total grad norm 82.3 vs 82.7, 4% rel L2): not a code bug.
- **Why (hypothesis):** only 16 rollouts per step and about half the groups carry no signal; Adam turns small noisy gradients into full-size steps, and LoRA scale 20 amplifies them. **Decision:** RL lr 2e-6 (LoRA), 1e-6 (full). The restart did not drift (see ~12:00).

### 2026-11-17 ~10:35 · Scope decision
- Measured RL steps at 17-46 s on the 0.8B, so the 16-run queue was ~130 h. We chose: all 8 setups on the 0.8B at full spec, 2B and 4B with LoRA at 1 epoch (~50 h).

### 2026-11-17 ~10:40 · GRPO v3: memory thrash
- **Observed:** peak MLX memory 15.6 GB, swap up to 5.5 GB, step times 11-46 s on a model with 1.6 GB of weights.
- **Profile (`tests/profile_memory.py`):** one step, micro-batch 2, 800 tokens: 15.8 GB without gradient checkpointing, 6.9 GB with it; micro-batch 1 without checkpointing 8.8 GB; micro-batch 4 with checkpointing 12.2 GB.
- **Decision:** gradient checkpointing on for the 0.8B (it was off). Restart: peak 6.8 GB, swap ~4 GB (system), steps ~19-25 s. **Open risk:** the 4B (8.5 GB weights) probably does not fit; measure the 2B first (Q2).
- Housekeeping: killed runs stay "running" in W&B until the heartbeat times out; discarded runs are now in W&B group `discarded`.

### 2026-11-17 12:04-12:51 · GRPO v4 (current) is learning
- Greedy test accuracy 0.54 → 0.56 (epoch 1) → 0.56 (epoch 2); truncation 14% → 8% → 27%. Training correct-rate 0.42 → 0.67 over steps 0-300.
- **Side effect noticed:** the "The answer is:" format share fell in sampled rollouts from ~0.8 to ~0.13, and in greedy test answers from 86% to 56% (epoch 2); LaTeX use 80% → 36%. Accuracy unaffected because the scorer reads the last number. **Why:** we stopped rewarding the format. (Hypothesis, consistent with timing; not tested by re-adding the bonus.)
- Added a qualitative section (format, length, markdown, LaTeX, before/after examples) to the report.

### 2026-11-17 14:20 · GRPO-LoRA (0.8B) finished: +10 points
- Greedy test accuracy 0.54 (baseline) → 0.56 → 0.56 → 0.63 → 0.63 → **0.64**; truncation 14% → 8% → 27% → 20% → 15% → 13%. Wall-clock ~3.8 h (about 22 s/step; ~69% generation, ~31% training; ~10% of steps had no signal and skipped the update).
- **Caveat:** +10 points on 100 test questions is at the edge of the ±10 noise band; one seed. The trend was monotone from epoch 2 on, which is encouraging but not proof.
- **Contrast with SFT:** same model, same data, SFT lowered accuracy (0.22 final) while GRPO raised it. Consistent with the hypothesis that training on the model's own answers preserves its reasoning style (untested directly).
- **Side effect:** "The answer is:" format share fell in training rollouts (~0.8 → ~0.13) because it is not rewarded.
- Next in the queue: RLOO-LoRA, GSPO-LoRA, then the full fine-tuning variants.

### 2026-11-17 ~16:20 · New experiment set: math-story-problems (0.8B only)
- **Request:** after all 0.8B GSM8K runs finish, stop (no 2B/4B), and repeat the 0.8B experiments on the Hugging Face dataset `azminetoushikwasi/math-story-problems`, which has short answers, logging to a new W&B project (`macbook-posttrain-mathstory`).
- **Dataset facts (measured):** 131,072 train / 1,024 validation / 3,072 test. Each row is one arithmetic problem in four forms: `eq_qs` (bare equation, e.g. `2385 + 761 - 1063 = ?`), `story_1_qs` (short story), `story_2_qs` (story with a distractor item type), `story_3_qs` (long story with many distractor sentences), plus a numeric `answer`. Always three numbers and two +/- operators; answers 25-15,784 (all positive); prompts 17 / 39 / 40 / 105 tokens for the four forms; 0 of 1,000 test equations appear in train.
- **Design (decided with the user):** train on 500 items, each using one random story form (1/3 each); evaluate 100 test items in **all four forms** (400 prompts) and log accuracy per form (`eval/acc_eq`, `eval/acc_story_1|2|3`) plus overall; 5 epochs, same hyperparameters, same eight setups as GSM8K (4 methods x LoRA/full).
- **No reference solutions exist** (only the final answer), so SFT targets are generated deterministically from the equation: `2385 + 761 = 3146 / 3146 - 1063 = 2083 / The answer is: 2083` (verified against the dataset answer; note it shows the arithmetic but not the story-to-equation step). This makes SFT here a different recipe from GSM8K SFT (human solutions), so cross-dataset SFT comparisons need that caveat.
- **Code:** `--dataset mathstory` in `train.py`; separate output dir `runs_mathstory/`. A 2-run smoke test passed (SFT and GRPO, per-form metrics logged). In that tiny test the untrained model got eq 100% / story_1 75% / story_2 0% / story_3 0% (n=4 each: anecdotal only).
- **Queue:** `scripts/run_queue.sh` now runs Phase A (rest of the GSM8K 0.8B set) then Phase B (math-story); 2B/4B moved to the optional `scripts/run_queue_larger.sh`. Config metadata fix: the stored `reward` description now says "correct only" (earlier runs' configs still carry the stale "+0.1 format" text; the reward actually used was correctness-only since the GRPO v2 fix).
- **Expectation (hypothesis, untested):** shorter answers mean much faster RL steps (generation dominates), so Phase B should cost far less wall-clock than Phase A; the distractor forms (story_2/3) should be the hard ones.

### 2026-11-17 17:55 · RLOO-LoRA (0.8B) finished: +13 points
- Greedy test accuracy 0.54 (baseline) → 0.48 → 0.57 → 0.66 → 0.60 → **0.67**; truncation 14% → 6% → 6% → 17% → 17% → 12%. About 3.4 h; ~16-28 s/step.
- Training behaviour differed from GRPO: RLOO shortened answers faster (mean completion ~230-260 tokens by step ~100 vs ~340-390 for GRPO) and kept the "The answer is:" format share higher early on.
- **Reading:** GRPO 0.64 vs RLOO 0.67 is a 3-point gap on 100 test questions (±10): not distinguishable with one seed. The epoch-to-epoch swings (0.66 → 0.60) are the same size as the gap. Both RL methods are clearly ahead of SFT (0.22).
- Next: GSPO-LoRA, then the four full fine-tuning runs.

### 2026-11-17 21:05 · GSPO-LoRA (0.8B) finished; stopping here for the day
- Greedy test accuracy 0.54 (baseline) → 0.52 → 0.53 → 0.62 → 0.66 → **0.59**; best epoch 0.66, final 0.59; truncation 2%-12%. ~3.1 h.
- **Reading:** the last-epoch drop (0.66 → 0.59) is within the ±10 noise of 100 test questions, so GSPO's "final" and "best" are not meaningfully different from each other or from GRPO (0.64) and RLOO (0.67). We report final-epoch numbers (no peeking at the test set to pick epochs); the best-epoch number is shown only as context.
- **Decision (user):** stop after GSPO-LoRA because the laptop had been running for ~20 hours. The queue was cancelled; nothing is running. **Not run:** the four 0.8B full fine-tuning setups on GSM8K, all eight math-story-problems runs (the code and queue for them are ready and smoke-tested), and the 2B/4B runs.
- **To resume:** `caffeinate -ims zsh scripts/run_queue.sh > logs/queue.log 2>&1 &` skips every finished run (it prints "already done, skipping") and continues with the full fine-tuning runs, then the math-story set. Expect the full runs to be memory-tight (about 11 GB of weights + optimizer state for the 0.8B against ≈9.5 GB usable; swap was already 3.3 GB during LoRA runs), so watch the first one and stop it if steps slow badly.

## Results so far (0.8B, LoRA, GSM8K: 500 train questions x 5 epochs, first 100 test questions, one seed)
| Run | Final acc | Best epoch | Hours | Peak MLX GB |
|---|---|---|---|---|
| untrained | 0.54 | – | – | – |
| SFT | 0.22 | 0.54 (epoch 0) | 0.3 | 9.8 (before checkpointing) |
| GRPO | 0.64 | 0.64 | 3.8 | 6.8 |
| RLOO | 0.67 | 0.67 | 3.4 | 6.9 |
| GSPO | 0.59 | 0.66 | 3.1 | 6.8 |

## Methodology notes
- Evaluation: first 100 GSM8K test questions, greedy, thinking off, 512-token cap; ±10 points at 95% confidence; one seed per setup.
- Failed attempts were kept deliberately; they explain the final settings.
- Learning rates were corrected after failures, not swept, so cross-method comparisons are not tuned head-to-heads.

## Next
1. Resume with `scripts/run_queue.sh` (see the 21:05 entry): the four full fine-tuning runs, then the eight math-story runs. Consider multiple seeds for the LoRA RL runs, since the method differences are inside the noise.
2. 2B/4B are out of scope for now (`scripts/run_queue_larger.sh` kept for later; measure the 2B before the 4B).
3. Write a math-story report (per-form accuracy by epoch, answer length, speed) when Phase B has results; update the GSM8K report's conclusions when the methods can be compared.
