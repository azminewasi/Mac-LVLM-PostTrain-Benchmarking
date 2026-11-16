"""LR schedule + AdamW with fp32 master weights (for full fine-tuning in bf16)."""
import math
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

def lr_at(step, total, base_lr, warmup_frac=0.05, min_ratio=0.1):
    warm = max(1, int(total * warmup_frac))
    if step < warm: return base_lr * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

class MasterAdamW:
    """bf16 weights get updates far below bf16 resolution at fine-tuning LRs (~1e-5..1e-6) and silently stall.
    We keep fp32 master copies + fp32 Adam moments for the trainable params only; the model stays bf16."""
    def __init__(self, model, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        flat = dict(tree_flatten(model.trainable_parameters()))
        self.dtypes = {k: v.dtype for k, v in flat.items()}
        self.master = {k: v.astype(mx.float32) for k, v in flat.items()}
        self.m = {k: mx.zeros_like(v) for k, v in self.master.items()}
        self.v = {k: mx.zeros_like(v) for k, v in self.master.items()}
        self.b1, self.b2 = betas; self.eps, self.wd, self.t = eps, weight_decay, 0
        mx.eval(self.master, self.m, self.v)

    def update(self, model, grads, lr):
        self.t += 1
        g = dict(tree_flatten(grads)); new = {}
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, gk in g.items():
            gk = gk.astype(mx.float32)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            p = self.master[k]
            p = p - lr * ((self.m[k] / c1) / (mx.sqrt(self.v[k] / c2) + self.eps) + self.wd * p)
            self.master[k] = p
            new[k] = p.astype(self.dtypes[k])
        model.update(tree_unflatten(list(new.items())))
        mx.eval(model.parameters(), self.master, self.m, self.v)
