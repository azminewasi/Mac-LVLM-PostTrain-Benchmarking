#!/bin/zsh
# Post-training queue, small -> large. Skips finished runs (runs/<name>/DONE). Each entry: "model method adapter [extra train.py args]".
#   0.8B: all 4 methods x {LoRA, full}, 500 samples x 5 epochs (full spec).
#   2B / 4B: SFT + GRPO/RLOO/GSPO with LoRA, reduced to 1 epoch (time budget: RL steps cost ~17-45 s on 0.8B and 2.5x / 5x that on 2B / 4B).
#   ./run_queue.sh                    start now
#   ./run_queue.sh --wait-pid PID     wait for that process (e.g. a run already in progress) to exit first
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
if [[ "$1" == "--wait-pid" ]]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; fi
RUNS=(
  "0.8B sft lora" "0.8B grpo lora" "0.8B rloo lora" "0.8B gspo lora"
  "0.8B sft full" "0.8B grpo full" "0.8B rloo full" "0.8B gspo full"
  "2B sft lora --epochs 1" "2B grpo lora --epochs 1" "2B rloo lora --epochs 1" "2B gspo lora --epochs 1"
  "4B sft lora --epochs 1" "4B grpo lora --epochs 1" "4B rloo lora --epochs 1" "4B gspo lora --epochs 1"
)
for r in $RUNS; do
  set -- ${=r}; m=$1; me=$2; ad=$3; shift 3
  echo "=== $(date '+%F %T') $m $me $ad $*"
  python train.py --model $m --method $me --adapter $ad $@ || echo "FAILED: $r"
done
echo QUEUE_DONE
python report.py
