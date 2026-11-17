#!/bin/zsh
# Post-training queue (0.8B only). Skips finished runs (<out>/<model>__<method>__<adapter>/DONE).
#   Phase A: GSM8K,           W&B project macbook-posttrain,            runs in runs/
#   Phase B: math-story-problems (azminetoushikwasi/math-story-problems; train on mixed story forms, test on all 4 forms),
#            W&B project macbook-posttrain-mathstory,                    runs in runs_mathstory/
# Each phase: SFT, GRPO, RLOO, GSPO with LoRA, then the same four with full fine-tuning. 500 train samples x 5 epochs.
# The larger-model runs (2B/4B, LoRA, 1 epoch) are in scripts/run_queue_larger.sh and are NOT part of this queue.
#   ./run_queue.sh                    start now
#   ./run_queue.sh --wait-pid PID     wait for that process (e.g. a run already in progress) to exit first
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
if [[ "$1" == "--wait-pid" ]]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; fi
SETUPS=("sft lora" "grpo lora" "rloo lora" "gspo lora" "sft full" "grpo full" "rloo full" "gspo full")
run_phase() {   # $1 = label, rest = extra train.py args
  local label=$1; shift
  local extras=("$@")
  for r in $SETUPS; do
    local m=${r%% *} a=${r##* } safe=()
    # full fine-tuning RL: weights + fp32 master + Adam state already take ~11 GB of the ~9.5 GB usable, so use the smallest micro-batch
    [[ $a == full && $m != sft ]] && safe=(--micro-bs 1)
    echo "=== $(date '+%F %T') [$label] 0.8B $m $a $safe"
    python train.py --model 0.8B --method $m --adapter $a $safe $extras || echo "FAILED: [$label] $r"
  done
}
run_phase gsm8k
run_phase mathstory --dataset mathstory --project macbook-posttrain-mathstory --out runs_mathstory
echo QUEUE_DONE
python report.py || true
