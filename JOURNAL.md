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
| Q7 | Does SFT stop hurting with (a) a gentle recipe on human solutions, or (b) self-distillation on the model's own correct answers? | answered (one seed, n=100): (a) partly: 0.37 vs 0.22 (baseline 0.54); (b) yes: 0.54, no drop |
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

### 2026-11-17 22:36 · Resumed after a restart: full fine-tuning (0.8B, GSM8K)
- The laptop had been shut down; on restart swap was 0 and memory 78% free. The post-train repo now has a GitHub remote (`azminewasi/post-train-testing`, in sync); the inference repo was also pushed by the user.
- `scripts/run_queue.sh` resumed (finished runs skipped): SFT-full first.
- **Measured (first 17 steps of SFT-full):** peak MLX memory **11.1 GB**, system RAM up to 13.8 GB, swap 3.4-3.5 GB of 4 GB (nearly exhausted), ~2.4 s/step. Trainable parameters 498M of 752M (all transformer blocks; embeddings frozen), lr 1e-5, gradient checkpointing on. This matches the earlier estimate (fp32 master weights + two Adam moments + bf16 weights ≈ 14 bytes/param).
- **Risk (hypothesis):** the RL full runs add rollout generation and activations on top of ~11 GB, so they may not fit in the ≈9.5 GB usable (16 GB minus ≈6.5 GB for macOS), i.e. swap exhaustion or thrash. **Mitigation tried:** `--micro-bs 1` for all RL full runs (queue script updated, swapped in without interrupting the SFT-full process). If GRPO-full thrashes or fails, "does not fit on 16 GB" is itself the result; record the measured peak and swap.

### 2026-11-17 ~22:45 · Full fine-tuning thrashes; SFT fixes built; new queue order
- **SFT-full (0.8B) was already thrashing.** First ~25 steps ran at ~2.4 s/step (peak MLX 11.1 GB), then step time jumped to 15-27 s, GPU utilisation fell from ~90% to 33-45% and the swap file grew from 4 to 5 GB. Cause (consistent with the numbers): weights + fp32 master copy + Adam state ≈ 11 GB plus ≈6.5 GB for macOS exceeds 16 GB. **Decision:** stopped it at ~step 45/625 (it would have taken 3-4 h and blocked everything else); run kept in `runs_discarded/0.8B__sft__full__infeasible-thrash` and tagged `discarded, infeasible-thrash` in W&B.
- **User instruction:** check the RL with full fine-tuning; if it fails, move to the 0.8B LoRA math-story runs. Implemented as a watchdog (`scripts/watchdog_check.py`): GRPO-full runs with micro-batch 1; if the median of the last 6 step times > 60 s (> ~10 h per run), peak MLX > 15.5 GB, or the process crashes, it is stopped and `runs/INFEASIBLE_RL_FULL` is written, and the other full-RL runs are skipped. Math-story runs are LoRA only.
- **SFT fix (user request: SFT should not drop the score).** Diagnosis so far: (a) the style mismatch from terse human solutions, (b) memorisation (loss 0.01), (c) an aggressive lr (accuracy fell to 0.17 already after the first epoch). Two new methods, both with LoRA lr 5e-6 and 2 epochs, test which matters: **`sftgentle`** = same human solutions, gentle recipe (isolates the hyperparameters); **`sftself`** = rejection-sampling fine-tuning on the model's own correct finished answers (4 samples per question at T=1.0, up to 2 distinct correct per question; isolates the data source). Smoke-tested (tiny). Whether either keeps accuracy at ~0.54 is an open question for the real runs; no guarantee.
- **Queue order now:** GSM8K full RL (guarded) → `sftgentle`, `sftself` (GSM8K, LoRA) → math-story LoRA: sft, sftgentle, sftself, grpo, rloo, gspo.
- **Monitoring:** a 20-minute health check (session cron) plus a log watcher; the check restarts a dead queue and fixes small failures.

### 2026-11-17 23:08 · Correction: full-RL (GRPO-full) is feasible after all
- **Observed:** GRPO-full (micro-batch 1) ran 60 steps in 22 min: ~20 s/step (generation ~12 s, training 0-18 s), peak MLX 11.1 GB, swap 2.5-3.8 GB, no thrash; the watchdog did not trigger. Reward noisy (0.19-0.56), no collapse so far.
- **What this changes:** my 22:45 conclusion that full fine-tuning "does not fit" was premature. The SFT-full thrash (steps 2.4 s → 15-27 s, GPU util 33%) happened at about the same memory (11.1 GB peak) and swap, so it may have been transient or caused by other activity on the machine at that moment (several shell and test processes were running); I cannot tell which. Memory is clearly borderline (swap near its limit, training-step times vary 0-18 s), so full runs are feasible but fragile. **Open:** re-run SFT-full later (it needs ~25 min if it does not thrash).
- **Queue consequence:** since the RL check passed, the queue continues with RLOO-full and GSPO-full (~3.5 h each) before the SFT fixes and the math-story runs, i.e. ~10 h before math-story starts. The user can reorder if they prefer the math-story/SFT-fix runs first.

### 2026-11-18 01:55 · GRPO-full (0.8B) finished: 0.68
- Greedy test accuracy 0.54 → 0.48 → 0.57 → 0.6 → 0.62 → 0.68 (baseline → epoch 5); truncation 0.14 → 0.05 → 0.15 → 0.1 → 0.15 → 0.15. About 3.1 h (micro-batch 1, ~12-21 s/step), peak MLX 11.1 GB, swap steady ~2.9 GB: **full fine-tuning RL is feasible on the 16 GB machine** (borderline, but it ran to completion without thrashing).
- **Reading:** 0.68 is the highest GSM8K result so far, but vs RLOO-LoRA 0.67, GRPO-LoRA 0.64 and GSPO-LoRA 0.59 the gaps are inside the ±10 noise of 100 test questions and one seed, so no ranking of full vs LoRA is justified. Same early dip as the LoRA runs (0.48 at epoch 1) followed by a steady rise.
- Next: RLOO-full, GSPO-full, then the SFT fixes and the math-story LoRA runs.

### 2026-11-18 04:45 · RLOO-full (0.8B) finished: 0.68
- Greedy test accuracy 0.54 → 0.57 → 0.51 → 0.62 → 0.61 → 0.68 (baseline → epoch 5); truncation 0.14 → 0.05 → 0.13 → 0.1 → 0.13 → 0.14. ~2.8 h, ~14-17 s/step, peak MLX 11.1 GB, swap ~2.5-3.5 GB (never exhausted).
- **Reading:** ties GRPO-full (0.68) and is close to RLOO-LoRA (0.67): the full-vs-LoRA and method differences are inside the ±10 noise of 100 test questions, one seed.
- **Qualitative:** the "The answer is:" format share collapsed to ~0.0 from about step 250 (GRPO-full kept ~0.7-0.8, GRPO-LoRA fell to ~0.13): another case of an unrewarded behaviour disappearing without any effect on the correctness-only score. Reward kept rising (training reward ~0.85 at the end) while greedy test accuracy plateaued at ~0.61-0.68.
- Next: GSPO-full, then `sftgentle` / `sftself` (GSM8K), then the six math-story LoRA runs.

### 2026-11-18 07:30 · GSPO-full (0.8B) finished: 0.70; the full-fine-tuning RL set is complete
- Greedy test accuracy 0.54 → 0.59 → 0.72 → 0.61 → 0.65 → **0.70** (baseline → epoch 5; best epoch 0.72); truncation 14% → 3% → 5% → 8% → 10% → 10%. ~2.8 h, ~12-20 s/step, peak MLX 11.1 GB, swap 2.5-4.0 GB (the swap file grew from 4 to 5.1 GB near the end; step times unaffected).
- **GSM8K 0.8B summary (final-epoch test accuracy, baseline 0.54, n=100, one seed):** LoRA: SFT 0.22, GRPO 0.64, RLOO 0.67, GSPO 0.59; full fine-tuning: GRPO 0.68, RLOO 0.68, GSPO 0.70 (SFT-full was stopped early, see the 22:45 entry, and has not been re-run). Every RL run is above the baseline and all RL runs are within ±10 points of each other: **no ranking of methods or of full-vs-LoRA is statistically justified**; epoch-to-epoch swings (e.g. GSPO-full 0.72 → 0.61) are as large as the differences between runs.
- **Qualitative:** all RL runs lost the unrewarded "The answer is:" format to varying degrees (GRPO-LoRA ~0.13, RLOO-full ~0.0, GSPO-full ~0.2-0.3, GRPO-full ~0.7-0.8) with no effect on the correctness-only score.
- Next in the queue: `sftgentle` and `sftself` (GSM8K, LoRA), then the six math-story LoRA runs.

### 2026-11-18 07:50 · `sftgentle` (GSM8K, LoRA) finished: 0.37 (still below the 0.54 baseline)
- **Recipe:** the original human solutions, but gentle: lr 5e-6 (was 2e-5), 2 epochs (was 5). Greedy test accuracy 0.54 → 0.32 → **0.37**; truncation 14% → 6% → 1%. About 9 min.
- **Style (epoch 2):** mean answer length 347 → 88 tokens, "The answer is:" format 86% → 99%, markdown 95% → 0%, LaTeX 80% → 0%.
- **Reading:** a gentler recipe roughly halves the damage (original SFT: 0.22 final, 0.17 after epoch 1) but accuracy still falls 17 points and the model still moves to the terse human style. This supports (does not prove) that the **data source**, not just the learning rate or number of epochs, is the main problem; `sftself` (the model's own correct answers) is the direct test. Caveat: n=100, one seed, only one gentle setting tried (no lr/epoch sweep).

### 2026-11-18 07:54 · `sftself` failed (Metal resource limit); fixed and re-run
- **Failure:** `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` in `build_self_data` after ~13 min: sampling all 2,000 answers (500 questions x 4) in a single `batch_generate` call exhausts Metal's per-process buffer-count limit. The RL runs never hit it because they generate only 16 prompts per call. The smoke test passed because it only generated 24 answers.
- **Fix (`pt/common.py`):** generate in chunks of 16 questions (64 prompts) and `mx.clear_cache()` between chunks; progress is printed every 5 chunks.
- **Queue effect:** the queue logged `FAILED` and moved on to the math-story phase (SFT first). The GSM8K `sftself` was re-launched by hand in parallel (SFT is light on memory; swap stayed ~1.8 GB) with log `logs/sftself_gsm8k.log`; the math-story `sftself` later in the queue will use the fixed code.

### 2026-11-18 08:00 · My mistake: running two jobs at once caused a GPU out-of-memory failure; queue restarted
- **What happened:** I started the fixed GSM8K `sftself` by hand while the queue was running the math-story SFT run. The math-story run died in its baseline evaluation with `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`; the next queued run (`sftgentle`, math-story) began and was in the same danger.
- **Cause (confirmed by timing):** two processes sharing the GPU (sampling 64 answers x 640 tokens + a 400-prompt evaluation) exceeded the GPU's working-memory limit. Earlier concurrent runs had been fine only because they were small.
- **Lesson:** never run two model jobs concurrently on this machine, even "light" ones; the memory limit is per-GPU, not per-process.
- **Action:** killed both, removed the partial run directories (the failed attempts' W&B runs are tagged `discarded` + `oom-concurrent`), and restarted `scripts/run_queue.sh` (log `logs/queue7.log`). It skipped everything finished and is running GSM8K `sftself` (chunk-fixed code) first, then the math-story set (sft, sftgentle, sftself, grpo, rloo, gspo), strictly one at a time.

### 2026-11-18 08:50 · `sftself` (GSM8K, LoRA) finished: 0.54, no drop (the SFT fix works, but does not improve)
- **Recipe:** fine-tune on the untrained model's own correct, finished answers (533 examples from 310/500 questions; 2,000 answers sampled at T=1.0, 64% finished), lr 5e-6, 2 epochs (267 steps, ~3 s/step). Greedy test accuracy 0.54 → 0.46 → **0.54** (final = baseline); truncation 14% → 10% → 10%. Total ~19 min plus ~40 min of sampling.
- **Style (epoch 2 vs baseline):** mean answer length 347 → 343 tokens, "The answer is:" 86% → 90%, markdown 95% → 97%, LaTeX 80% → 78%: the model's own style is largely preserved, unlike the human-solution runs (answers ~90 tokens, markdown/LaTeX gone).
- **Comparison (final test accuracy, baseline 0.54):** SFT on human solutions: 0.22 (lr 2e-5, 5 epochs); `sftgentle` 0.37 (human solutions, lr 5e-6, 2 epochs); `sftself` 0.54 (own answers, lr 5e-6, 2 epochs). Holding the gentle recipe fixed, changing only the training data from human solutions to the model's own answers removes the drop, which supports **data source (style/distribution match) as the main cause**, with learning rate and epochs a secondary factor.
- **Caveats:** n=100, one seed; the epoch-1 dip (0.46) was within noise of the final value; `sftself` neither improved nor hurt accuracy, as expected for training on the model's own outputs (it can only reinforce what it already does); the data covers only questions the model already solves (62%) and finished answers (biased toward shorter ones).

### 2026-11-18 09:12 · math-story-problems, SFT (LoRA, equation-derived solutions) finished: 0.565 → 0.94
- **Setup:** 500 train items (one random story form each), targets generated from the equation (`a + b = c / c - d = e / The answer is: e`), lr 2e-5, 5 epochs, 625 steps (~0.8-1.3 s/step, ~18 min), peak MLX ~3.3 GB. Evaluation: first 100 test items in all four forms (400 prompts), greedy, 512-token cap, project `macbook-posttrain-mathstory`.
| Epoch | Overall | eq | story_1 | story_2 | story_3 | Mean tokens | Truncated |
|---|---|---|---|---|---|---|---|
| 0 | 56% | 90% | 74% | 54% | 8% | 302 | 18% |
| 1 | 65% | 56% | 67% | 68% | 68% | 43 | 0% |
| 2 | 84% | 69% | 89% | 87% | 89% | 42 | 0% |
| 3 | 92% | 87% | 94% | 93% | 94% | 42 | 0% |
| 4 | 94% | 89% | 95% | 95% | 95% | 42 | 0% |
| 5 | 94% | 89% | 95% | 96% | 96% | 42 | 0% |
- **Reading:** unlike on GSM8K, SFT is a large win here (+37 points overall). The baseline's weakness was mostly the long distractor story (8%, with 65% of its answers truncated at 512 tokens, 23% correct among those that finished); SFT teaches a short, correct arithmetic style (~42 tokens, never truncated) that the model can execute. The bare-equation form dipped at epoch 1 (90% → 56%) because training used only story forms, then recovered to ~89%.
- **Why SFT works here but not on GSM8K (hypothesis, untested):** the equation-derived targets are short, always correct and within what a 0.8B model can execute, whereas GSM8K's human solutions skip arithmetic the model cannot reproduce. Caveats: n=100 per form (±10 points), one seed; the SFT targets are an easy deterministic template, so this says little about SFT on natural-language solutions.
- **Next:** math-story `sftgentle`, `sftself`, then GRPO, RLOO, GSPO (LoRA). The RL methods need to beat ~0.94 to matter. The report generator only reads `runs/` (GSM8K); a math-story report is still to be written once more runs exist.

### 2026-11-18 09:25 · math-story-problems, `sftgentle` (LoRA) finished: 0.565 → 0.978
- **Setup:** same equation-derived targets as the SFT run but lr 5e-6 and 2 epochs (250 steps, ~9 min).
| Epoch | Overall | eq | story_1 | story_2 | story_3 | Mean tokens | Truncated |
|---|---|---|---|---|---|---|---|
| 0 | 56% | 90% | 74% | 54% | 8% | 302 | 18% |
| 1 | 97% | 97% | 97% | 97% | 97% | 42 | 0% |
| 2 | 98% | 96% | 99% | 98% | 98% | 42 | 0% |
- **Reading:** the gentle recipe beats the original (lr 2e-5, 5 epochs: 0.94) and does so after one epoch (0.97): the bare-equation dip seen with the original SFT (56% at epoch 1) never appears. Plausible reason (hypothesis): the higher lr overshoots/overfits to the story templates. Caveat: 0.978 vs 0.94 over 400 prompts (±~2-3 points at this accuracy level) is borderline significant; one seed.
- **Contrast with GSM8K:** there the gentle recipe still hurt (0.37) because the human solutions were not something the model could execute; here the targets are clean and executable, so gentler is simply better.

### 2026-11-18 10:11 · math-story-problems, `sftself` (LoRA) finished: 0.565 → 0.635 (best epoch 0.685)
- **Setup:** fine-tune on the untrained model's own correct finished answers (552 examples from 324/500 questions; 2,000 answers sampled at T=1.0, 66% finished), lr 5e-6, 2 epochs (~275 steps, ~25 min plus ~28 min of sampling).
| Epoch | Overall | eq | story_1 | story_2 | story_3 | Mean tokens | Truncated |
|---|---|---|---|---|---|---|---|
| 0 | 56% | 90% | 74% | 54% | 8% | 302 | 18% |
| 1 | 68% | 89% | 92% | 83% | 10% | 283 | 10% |
| 2 | 64% | 83% | 90% | 61% | 20% | 286 | 14% |
- **Reading:** a modest gain overall (+7 points final, +12 at the best epoch) concentrated in the short story form (74% → 90%); the long distractor story stays poor (8% → 20%, 45% still truncated at the end), and story_2 fell from 83% at epoch 1 to 61% at epoch 2, a swing larger than the ±10 noise. The data covers only questions the model already solves (324/500) and finished answers, so it has little signal on the hard form (story_3). Self-distillation can only reinforce what the model already does; here the equation-derived targets (SFT 0.94, `sftgentle` 0.978) are far better because they teach a new, executable style.
- **Cross-dataset comparison of the three SFT recipes (final test accuracy; baseline in brackets):** GSM8K: human 0.22, gentle 0.37, self 0.54 [0.54]. Math-story: equation-derived 0.94, gentle 0.978, self 0.635 [0.565]. The best SFT data source depends on whether the reference solutions are something the model can execute.
- Next: math-story GRPO, RLOO, GSPO (LoRA).

### 2026-11-18 11:50 · math-story-problems, GRPO (LoRA) finished: 0.565 → 0.99
- **Setup:** same as the GSM8K RL runs (4 prompts x 4 rollouts per step, lr 2e-6, correctness-only reward, 640 new tokens, 625 steps, ~1.6 h, ~6-20 s/step, peak MLX 6.8 GB).
| Epoch | Overall | eq | story_1 | story_2 | story_3 | Mean tokens | Truncated |
|---|---|---|---|---|---|---|---|
| 0 | 56% | 90% | 74% | 54% | 8% | 302 | 18% |
| 1 | 96% | 97% | 98% | 96% | 91% | 137 | 0% |
| 2 | 97% | 97% | 97% | 97% | 97% | 137 | 0% |
| 3 | 98% | 97% | 98% | 98% | 99% | 147 | 0% |
| 4 | 99% | 99% | 100% | 99% | 99% | 149 | 0% |
| 5 | 99% | 99% | 99% | 99% | 99% | 149 | 0% |
- **Reading:** RL reaches the top of this dataset: 0.955 after the first epoch and 0.99 at the end, every form at 97-100% including the long distractor story (8% → 99%), with no truncation. It beats the original SFT (0.94) and is at/above the gentle SFT (0.978; the gap is within noise at ~400 prompts), **without needing the equation-derived reference solutions**: the reward alone was enough. Answers shrank from ~300 to ~150 tokens (SFT: ~42).
- **Dynamics:** reward climbed from 0.48 to ~1.0 by step ~250; from then on ~95-98% of question groups had all four answers equal (no gradient), so the later steps changed little (last-100-step mean zero-variance share 98%). Step time fell from ~20 s to ~6 s as answers shortened. I had predicted at the step-63 check that GRPO would level off well below SFT; that was wrong (the first epoch alone gave 0.955). The unrewarded "The answer is:" format again vanished after step ~250 with no effect on the score.
- **Caveats:** n=100 per form, one seed. Test equations never appear in train; the train distribution (story forms) differs from the evaluated bare-equation form but eq reached 99%.
- Next: math-story RLOO, GSPO (LoRA).

### 2026-11-18 12:50 · math-story-problems, RLOO (LoRA) finished: 0.565 → 0.98 (peak 1.000 at epoch 2)
- **Setup:** as the other RL runs (lr 2e-6, correctness-only reward, 625 steps, ~1.1 h, step time fell from ~26 s to ~3 s as answers shortened; peak MLX 6.9 GB).
| Epoch | Overall | eq | story_1 | story_2 | story_3 | Mean tokens | Truncated |
|---|---|---|---|---|---|---|---|
| 0 | 56.5% | 90% | 74% | 54% | 8% | 302 | 18% |
| 1 | 99.5% | 99% | 100% | 100% | 99% | 123 | 0% |
| 2 | 100% | 100% | 100% | 100% | 100% | 95 | 0% |
| 3 | 99.5% | 99% | 100% | 100% | 99% | 72 | 0% |
| 4 | 98% | 97% | 98% | 98% | 99% | 73 | 0% |
| 5 | 98% | 97% | 98% | 98% | 99% | 73 | 0% |
- **Reading:** the fastest learner on this dataset: 0.995 after one epoch and a perfect 400/400 at epoch 2 (GRPO needed four epochs to reach 0.99). It then slipped to 0.98 for epochs 4-5 (8 wrong of 400); the final-epoch number is the one reported. Answers kept shrinking (302 → 73 tokens), and the remaining errors are arithmetic slips in the terse style (e.g. 2640 - 508 + 1425 written as 3555 instead of 3557; one story where the first subtraction was miscomputed), not misreading the question. Consistent with a mild terseness/precision trade-off late in training (hypothesis; the effect is small: 8 vs 2 wrong answers is borderline significant at best).
- **Caveats:** 400 prompts, one seed; at 98-100% accuracy differences between GRPO (0.99), RLOO (0.98) and `sftgentle` (0.978) are not meaningful.
- Next: math-story GSPO (LoRA), the last run in the queue.

## Results so far (0.8B, GSM8K: 500 train questions x 5 epochs, first 100 test questions, one seed)
| Run | Final acc | Best epoch acc | Hours | Peak MLX GB |
|---|---|---|---|---|
| untrained | 0.54 | – | – | – |
| SFT, LoRA | 0.22 | 0.54 (epoch 0) | 0.3 | 9.8 (before checkpointing) |
| GRPO, LoRA | 0.64 | 0.64 | 3.8 | 6.8 |
| RLOO, LoRA | 0.67 | 0.67 | 3.4 | 6.9 |
| GSPO, LoRA | 0.59 | 0.66 | 3.1 | 6.8 |
| GRPO, full | 0.68 | 0.68 | 3.0 | 11.1 |
| RLOO, full | 0.68 | 0.68 | 2.8 | 11.1 |
| GSPO, full | 0.70 | 0.72 | 2.8 | 11.1 |
| SFT gentle, LoRA | 0.37 | 0.54 (epoch 0) | 0.2 | – |
| SFT self-distilled, LoRA | 0.54 | 0.54 | 0.2 (+0.7 sampling) | – |

## Methodology notes
- Evaluation: first 100 GSM8K test questions, greedy, thinking off, 512-token cap; ±10 points at 95% confidence; one seed per setup.
- Failed attempts were kept deliberately; they explain the final settings.
- Learning rates were corrected after failures, not swept, so cross-method comparisons are not tuned head-to-heads.

## Next
1. Resume with `scripts/run_queue.sh` (see the 21:05 entry): the four full fine-tuning runs, then the eight math-story runs. Consider multiple seeds for the LoRA RL runs, since the method differences are inside the noise.
2. 2B/4B are out of scope for now (`scripts/run_queue_larger.sh` kept for later; measure the 2B before the 4B).
3. Write a math-story report (per-form accuracy by epoch, answer length, speed) when Phase B has results; update the GSM8K report's conclusions when the methods can be compared.
