"""Chunked differentiable gated delta == library reference ops (values + grads), and it is differentiable."""
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops
from pt.gdn import chunk_gated_delta
mx.random.seed(0)
B, T, Hk, Hv, Dk, Dv = 2, 150, 2, 4, 32, 48
q = mx.random.normal((B, T, Hk, Dk)) * 0.2; k = mx.random.normal((B, T, Hk, Dk)) * 0.2
v = mx.random.normal((B, T, Hv, Dv)); lg = -mx.abs(mx.random.normal((B, T, Hv))) * 0.1; beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
y0, s0 = gated_delta_ops(q, k, v, mx.exp(lg), beta)
y1, s1 = chunk_gated_delta(q, k, v, lg, beta)
print("max|dy|", mx.abs(y0 - y1).max().item(), "max|ds|", mx.abs(s0 - s1).max().item())
assert mx.allclose(y0, y1, atol=2e-3, rtol=2e-3) and mx.allclose(s0, s1, atol=2e-3, rtol=2e-3)
f0 = lambda v_, k_: gated_delta_ops(q, k_, v_, mx.exp(lg), beta)[0].sum()
f1 = lambda v_, k_: chunk_gated_delta(q, k_, v_, lg, beta)[0].sum()
g0 = mx.grad(f0, argnums=(0, 1))(v, k); g1 = mx.grad(f1, argnums=(0, 1))(v, k)
for a, b in zip(g0, g1): print("grad max|d|", mx.abs(a - b).max().item(), "scale", mx.abs(a).max().item()); assert (mx.abs(a - b).max() < 0.01 * mx.abs(a).max()).item()
print("GDN_OK")
