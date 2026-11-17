"""Differentiable, chunk-parallel gated delta rule (pure MLX ops, matmul based).

Qwen3.5's fast Metal kernel (mlx_lm gated_delta_kernel) has no vjp, and the library's ops fallback is a per-token Python
loop (far too slow for training). This is the chunkwise (WY-form) algorithm used by fla / the HF torch reference, with the
within-chunk triangular inverse done by a log2(C)-step Neumann product so everything is GPU matmuls and autograd-friendly.
Same recurrence/layouts as mlx_lm.models.gated_delta: q,k [B,T,Hk,Dk] (q already scaled), v [B,T,Hv,Dv],
log_g [B,T,Hv] (log of the decay), beta [B,T,Hv]; state [B,Hv,Dv,Dk]. Right-padded batches are fine (causal)."""
import mlx.core as mx
import mlx.nn as nn


def chunk_gated_delta(q, k, v, log_g, beta, state=None, C=32):
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if Hv // Hk > 1:
        q = mx.repeat(q, Hv // Hk, -2); k = mx.repeat(k, Hv // Hk, -2)
    dt = q.dtype
    f32 = mx.float32
    q, k, v, lg, beta = (x.astype(f32) for x in (q, k, v, log_g, beta))
    pad = (-T) % C
    if pad:
        z = lambda x: mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        q, k, v, lg, beta = z(q), z(k), z(v), z(lg), z(beta)   # padded beta=0, lg=0 -> no effect on earlier tokens
    N = (T + pad) // C
    # -> [B,H,N,C,D]
    def ch(x):
        x = x.reshape(B, N, C, *x.shape[2:])
        return x.transpose(0, 3, 1, 2, 4) if x.ndim == 5 else x.transpose(0, 3, 1, 2)
    q, k, v, lg, beta = ch(q), ch(k), ch(v), ch(lg), ch(beta)           # lg,beta: [B,H,N,C]
    gc = mx.cumsum(lg, axis=-1)                                          # within-chunk cumulative log decay
    kb = k * beta[..., None]; vb = v * beta[..., None]
    idx = mx.arange(C)
    low = idx[:, None] > idx[None, :]                                    # strictly lower
    lowd = idx[:, None] >= idx[None, :]
    diff = gc[..., :, None] - gc[..., None, :]
    D = mx.where(lowd, mx.exp(mx.where(lowd, diff, 0.0)), 0.0)           # decay_mask (lower incl. diag)
    M = -mx.where(low, (kb @ k.swapaxes(-1, -2)) * D, 0.0)               # strictly lower, nilpotent
    # (I - M)^-1 by forward substitution (numerically stable; a Neumann product blows up for correlated keys)
    rows = []
    for i in range(C):
        r = M[..., i, :]
        if i:
            r = r + (M[..., i:i + 1, :i] @ mx.stack(rows, axis=-2))[..., 0, :]
        rows.append(r)
    Tm = mx.stack(rows, axis=-2) + mx.eye(C, dtype=f32)
    value = Tm @ vb                                                      # [B,H,N,C,Dv]
    kcum = Tm @ (kb * mx.exp(gc)[..., None])                             # [B,H,N,C,Dk]
    A = mx.where(lowd, (q @ k.swapaxes(-1, -2)) * D, 0.0)                # intra-chunk attention (incl. diag)
    qg = q * mx.exp(gc)[..., None]
    kd = mx.exp(gc[..., -1:] - gc)[..., None]                            # exp(g_last - g_t)
    glast = mx.exp(gc[..., -1])                                          # [B,H,N]
    S = mx.zeros((B, Hv, Dk, Dv), dtype=f32) if state is None else state.astype(f32).swapaxes(-1, -2)
    outs = []
    for i in range(N):
        vnew = value[:, :, i] - kcum[:, :, i] @ S                        # [B,H,C,Dv]
        outs.append(qg[:, :, i] @ S + A[:, :, i] @ vnew)
        S = S * glast[:, :, i, None, None] + (k[:, :, i] * kd[:, :, i]).swapaxes(-1, -2) @ vnew
    y = mx.stack(outs, axis=2)                                           # [B,H,N,C,Dv]
    y = y.transpose(0, 2, 3, 1, 4).reshape(B, N * C, Hv, Dv)[:, :T]
    return y.astype(dt), S.swapaxes(-1, -2)


def make_train_update(orig_update):
    """Drop-in for gated_delta_update that uses the chunked differentiable path."""
    from mlx_lm.models.gated_delta import compute_g
    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True, lower_bound=None, allow_neg_eigval=False):
        if lower_bound is not None or allow_neg_eigval or k.ndim != 4:
            return orig_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=False, lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        return chunk_gated_delta(q, k, v, log_g, beta, state)
    return update
