#!/bin/zsh
# Visual-QA queue (SpatialEval spatial-map, Qwen3.5-0.8B, LoRA only): three SFT variants, then GRPO, RLOO, GSPO.
# Finished runs are skipped (runs_spatial/<name>/DONE). Never run two model jobs at once on 16 GB.
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
for m in sft sftgentle sftself grpo rloo gspo; do
  echo "=== $(date '+%F %T') vision $m lora"
  python train_vision.py --method $m --project macbook-posttrain-spatial --out runs_spatial || echo "FAILED: $m"
done
echo QUEUE_DONE
