"""Peak MLX memory of one LoRA training forward+backward on the 0.8B for different shapes / checkpointing.
Run:  python -m tests.profile_memory   (writes results/memory_profile.json). Uses random token ids (memory depends on shape only)."""
import json, mlx.core as mx, mlx.nn as nn, mlx_lm.models.qwen3_5 as q35
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from pt.gdn import make_train_update
from pt.losses import sft_loss
GB = 2 ** 30
m, tok = load("Qwen/Qwen3.5-0.8B"); m.freeze(); linear_to_lora_layers(m, len(m.layers), dict(rank=16, scale=20.0, dropout=0.0))
orig = q35.gated_delta_update; tu = make_train_update(orig); FLAG = [False]
q35.gated_delta_update = lambda *a, **k: (tu if FLAG[0] else orig)(*a, **k)
out = []
def run(B, T, pl, ckpt, label):
    ids = mx.random.randint(0, 50000, (B, T)); P = mx.array([pl] * B); Tl = mx.array([T] * B)
    mx.clear_cache(); mx.reset_peak_memory(); FLAG[0] = True
    lv, g = nn.value_and_grad(m, lambda *_: sft_loss(m, ids, P, Tl, float(B * (T - pl))))(ids); mx.eval(lv, g); FLAG[0] = False
    out.append(dict(label=label, batch=B, seq_len=T, grad_checkpointing=ckpt, peak_gb=round(mx.get_peak_memory() / GB, 2))); print(out[-1])
run(2, 400, 100, False, "SFT-like step, micro-batch 2")
run(2, 800, 150, False, "RL-like step, micro-batch 2")
run(1, 800, 150, False, "RL-like step, micro-batch 1")
grad_checkpoint(m.layers[0])
run(2, 800, 150, True, "RL-like step, micro-batch 2")
run(4, 800, 150, True, "RL-like step, micro-batch 4")
json.dump(out, open("results/memory_profile.json", "w"), indent=1)
