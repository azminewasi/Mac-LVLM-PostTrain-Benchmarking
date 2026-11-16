#!/bin/zsh
# Tiny end-to-end check of every code path on the 0.8B (8 train samples, 1 epoch, offline W&B). ~10-20 min. Run AFTER the inference benchmark.
cd "$(dirname "$0")/.." || exit 1; source ../inference-testing/.venv/bin/activate
python -m tests.test_units || exit 1
C="--n-train 8 --epochs 1 --n-eval 8 --batch 2 --group 2 --max-new 64 --wandb-mode offline --project smoke --out runs_smoke"
for m in sft grpo rloo gspo; do python train.py --model 0.8B --method $m --adapter lora ${=C} || exit 1; done
python train.py --model 0.8B --method sft  --adapter full ${=C} || exit 1
python train.py --model 0.8B --method grpo --adapter full ${=C} --mu 2 || exit 1   # also exercises old-logprob + clipping path
echo SMOKE_OK
