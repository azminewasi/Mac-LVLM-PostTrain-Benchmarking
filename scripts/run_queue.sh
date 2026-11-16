#!/bin/zsh
# Runs all 16 (model, method, adapter) setups sequentially, small -> large. Skips finished ones (runs/<name>/DONE).
#   ./run_queue.sh                      start now
#   ./run_queue.sh --wait-for-inference wait until the inference benchmark is fully finished (9B 8x summary exists), then start
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
if [[ "$1" == "--wait-for-inference" ]]; then
  while [[ ! -f ../inference-testing/results/9B-4bit-think8x_summary.csv ]]; do sleep 120; done
fi
RUNS=(
  "0.8B sft lora" "0.8B grpo lora" "0.8B rloo lora" "0.8B gspo lora"
  "0.8B sft full" "0.8B grpo full" "0.8B rloo full" "0.8B gspo full"
  "2B sft lora" "2B grpo lora" "2B rloo lora" "2B gspo lora"
  "4B sft lora" "4B grpo lora" "4B rloo lora" "4B gspo lora"
)
for r in $RUNS; do
  set -- ${=r}
  echo "=== $(date '+%F %T') $1 $2 $3"
  python train.py --model $1 --method $2 --adapter $3 "${EXTRA[@]}" || echo "FAILED: $r"
done
echo QUEUE_DONE
python report.py
