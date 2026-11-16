"""Token log-probs, SFT loss, and the three critic-free RL objectives (GRPO, RLOO, GSPO).

All three are *critic-free* and *reference-free* (KL coefficient 0, as in DAPO/Dr.GRPO), so only one model
(policy) lives in memory; with LoRA the adapter is the only thing that changes.

  GRPO : A_i = (r_i - mean_g) / (std_g + eps); token-level ratio, PPO-clip, per-sequence token-mean loss.
  RLOO : A_i = r_i - mean_{j!=i} r_j  (= G/(G-1) * (r_i - mean_g)); plain REINFORCE (no ratio / clip).
  GSPO : A_i as GRPO; *sequence-level* ratio s_i = exp(mean_t(logp - logp_old)), clipped with tight eps.
With mu=1 (one gradient step per rollout batch, the default) the policy is exactly on-policy, so ratio==1 and
clipping is inactive; gradients of GRPO/GSPO then differ in how tokens/sequences are weighted. Use --mu >1 to
reuse rollouts and exercise clipping."""
import numpy as np
import mlx.core as mx

def group_advantages(rewards, G, method):
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, G)
    mean = r.mean(1, keepdims=True)
    if method == "rloo":
        adv = (r - mean) * G / (G - 1)
    else:  # grpo / gspo
        adv = (r - mean) / (r.std(1, keepdims=True) + 1e-4)
    return adv.reshape(-1)

def _head(model):
    lm = getattr(model, "language_model", model)
    inner = lm.model
    if getattr(lm.args, "tie_word_embeddings", False): return inner, inner.embed_tokens.as_linear
    return inner, lm.lm_head

def token_logprobs(model, ids, plens, tlens):
    """ids (B,T) right-padded; plens/tlens (B,) = prompt length / total length.
    Returns (logp (B,L), mask (B,L)) for target positions covering ONLY completion tokens.
    Only positions from min(plens)-1 onward go through the LM head (saves the big vocab matmul on prompts)."""
    B, T = ids.shape
    inputs, targets = ids[:, :-1], ids[:, 1:]
    s = max(0, int(mx.min(plens).item()) - 1)
    inner, head = _head(model)
    h = inner(inputs)[:, s:]
    logits = head(h).astype(mx.float32)
    tgt = targets[:, s:]
    lp = mx.take_along_axis(logits, tgt[..., None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)
    pos = mx.arange(s + 1, T)[None, :]
    mask = ((pos >= plens[:, None]) & (pos < tlens[:, None])).astype(mx.float32)
    return lp, mask

def sft_loss(model, ids, plens, tlens, ntok_total):
    lp, mask = token_logprobs(model, ids, plens, tlens)
    return -(lp * mask).sum() / ntok_total

def rl_loss(model, ids, plens, tlens, adv, old_lp, method, eps_lo, eps_hi, n_seqs):
    """old_lp: (B,L) detached log-probs of the sampling policy, or None (=> on-policy, ratio 1)."""
    lp, mask = token_logprobs(model, ids, plens, tlens)
    ntok = mx.maximum(mask.sum(1), 1.0)
    old = mx.stop_gradient(lp) if old_lp is None else old_lp
    a = adv
    if method == "rloo":
        per_seq = -(a[:, None] * lp * mask).sum(1) / ntok
        ratio_tok = mx.exp(lp - mx.stop_gradient(lp)); clipped = mx.zeros_like(a)
    elif method == "grpo":
        ratio = mx.exp(lp - old)
        surr = mx.minimum(ratio * a[:, None], mx.clip(ratio, 1 - eps_lo, 1 + eps_hi) * a[:, None])
        per_seq = -(surr * mask).sum(1) / ntok
        ratio_tok = ratio
        clipped = ((mx.abs(ratio - 1) > eps_hi) * mask).sum(1) / ntok
    elif method == "gspo":
        s_ratio = mx.exp(((lp - old) * mask).sum(1) / ntok)
        surr = mx.minimum(s_ratio * a, mx.clip(s_ratio, 1 - eps_lo, 1 + eps_hi) * a)
        per_seq = -surr
        ratio_tok = s_ratio[:, None] * mask; clipped = (mx.abs(s_ratio - 1) > eps_hi).astype(mx.float32)
    else:
        raise ValueError(method)
    loss = per_seq.sum() / n_seqs
    aux_ratio = (ratio_tok * mask).sum() / mask.sum() if method != "gspo" else s_ratio.mean()
    return loss, mx.stack([aux_ratio, clipped.sum(), mask.sum(), (-(lp * mask).sum())])
