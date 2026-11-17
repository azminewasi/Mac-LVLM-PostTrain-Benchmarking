#!/bin/zsh
# OPTIONAL, not part of the default queue: 2B and 4B with LoRA at 1 epoch (SFT, GRPO, RLOO, GSPO).
# Caveat: the 4B very likely does not fit in 16 GB for RL training as configured (see JOURNAL.md); measure the 2B first.
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
for m in 2B 4B; do for me in sft grpo rloo gspo; do
  echo "=== $(date '+%F %T') $m $me lora --epochs 1"
  python train.py --model $m --method $me --adapter lora --epochs 1 || echo "FAILED: $m $me"
done; done
echo LARGER_DONE
