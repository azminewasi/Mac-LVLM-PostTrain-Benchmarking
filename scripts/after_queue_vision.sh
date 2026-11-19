#!/bin/zsh
# Runs after scripts/run_queue_vision.sh has finished (waits for its QUEUE_DONE line), one job at a time:
#   1. single-stream per-token timing of the untrained model and every trained adapter  (tests/vision_infer_speed.py)
#   2. out-of-distribution test on maze navigation                                     (tests/vision_ood.py)
#   3. refresh results/vision_analytics.html
#   4. 2B model with LoRA on the same task: SFT, gentle SFT, GRPO                      (train_vision.py --model 2B)
cd "$(dirname "$0")/.." || exit 1
source ../inference-testing/.venv/bin/activate
export WANDB_API_KEY="$(cat ../inference-testing/.wandb_key)"
until grep -q "^QUEUE_DONE" logs/queue_vision.log 2>/dev/null; do sleep 120; done
echo "=== $(date '+%F %T') queue finished"
echo "=== $(date '+%F %T') 1/4 single-stream timing";  python -m tests.vision_infer_speed > logs/vision_infer_speed.log 2>&1 || echo "FAILED timing (see logs/vision_infer_speed.log)"
echo "=== $(date '+%F %T') 2/4 out-of-distribution test"; python -m tests.vision_ood > logs/vision_ood.log 2>&1 || echo "FAILED ood (see logs/vision_ood.log)"
echo "=== $(date '+%F %T') 3/4 analytics";                python analysis/vision_analytics.py || echo "FAILED analytics"
echo "=== $(date '+%F %T') 4/4 2B LoRA runs"
cp staging/train_vision.py train_vision.py && cp staging/pt/vision.py pt/vision.py && python -m py_compile train_vision.py pt/vision.py || { echo "FAILED to install the 2B-capable scripts"; exit 1; }
for m in sft sftgentle grpo; do
  echo "=== $(date '+%F %T') vision 2B $m lora"
  python train_vision.py --model 2B --method $m --project macbook-posttrain-spatial --out runs_spatial || echo "FAILED: 2B $m"
done
echo AFTER_DONE
