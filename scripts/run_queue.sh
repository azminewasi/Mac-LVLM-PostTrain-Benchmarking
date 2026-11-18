#!/bin/zsh
# Post-training queue (0.8B only). Skips finished runs (<out>/<model>__<method>__<adapter>/DONE).
#   Phase A (GSM8K, project macbook-posttrain, runs/):
#       1. RL with FULL fine-tuning (grpo, rloo, gspo), guarded by a watchdog: if the first one crashes, thrashes
#          (>60 s/step) or exceeds memory, it is stopped, flagged in runs/INFEASIBLE_RL_FULL, and the other full-RL runs are skipped.
#       2. the SFT fixes with LoRA: sftgentle (human solutions, lr 5e-6, 2 epochs) and sftself (the model's own correct answers).
#   Phase B (math-story-problems, project macbook-posttrain-mathstory, runs_mathstory/), LoRA only:
#       sft, sftgentle, sftself, grpo, rloo, gspo.   500 train samples; 5 epochs (2 for the gentle SFT variants).
# (2B/4B were not run.)
#   ./run_queue.sh                    start now
#   ./run_queue.sh --wait-pid PID     wait for that process (e.g. a run already in progress) to exit first
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
if [[ "$1" == "--wait-pid" ]]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; fi
FLAG=runs/INFEASIBLE_RL_FULL

run_one() {   # $1 label  $2 out dir  $3 method  $4 adapter  rest = extra train.py args
  local label=$1 outdir=$2 m=$3 a=$4; shift 4
  local extras=("$@") name="0.8B__${m}__${a}"
  if [[ $a == full && ( $m == grpo || $m == rloo || $m == gspo ) ]]; then
    if [[ -f $FLAG ]]; then echo "=== $(date '+%F %T') [$label] 0.8B $m $a SKIPPED (full RL infeasible: $(cat $FLAG))"; return; fi
    [[ -f $outdir/$name/DONE ]] && { echo "$name already done, skipping"; return; }
    echo "=== $(date '+%F %T') [$label] 0.8B $m $a (guarded, micro-batch 1)"
    python train.py --model 0.8B --method $m --adapter $a --micro-bs 1 --out $outdir $extras &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 45
      local why; why=$(python scripts/watchdog_check.py $outdir/$name); local rc=$?
      if [[ $rc -eq 3 ]]; then kill $pid; sleep 5; echo "$why" > $FLAG; echo "WATCHDOG: $why"; return; fi
    done
    wait $pid || { echo "FAILED: [$label] $m $a (process crashed)"; echo "process crashed (see logs)" > $FLAG; }
  else
    echo "=== $(date '+%F %T') [$label] 0.8B $m $a"
    python train.py --model 0.8B --method $m --adapter $a --out $outdir $extras || echo "FAILED: [$label] $m $a"
  fi
}

# Phase A: GSM8K
for r in "grpo full" "rloo full" "gspo full"; do run_one gsm8k runs ${=r}; done
for r in "sftgentle lora" "sftself lora"; do run_one gsm8k runs ${=r}; done
# Phase B: math-story-problems (LoRA only)
for r in "sft lora" "sftgentle lora" "sftself lora" "grpo lora" "rloo lora" "gspo lora"; do
  run_one mathstory runs_mathstory ${=r} --dataset mathstory --project macbook-posttrain-mathstory
done
echo QUEUE_DONE
python report.py || true
