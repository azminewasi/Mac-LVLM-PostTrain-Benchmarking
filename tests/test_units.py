"""Pure-math unit tests on a tiny stub model, CPU only (no weights, negligible load). Run: python -m tests.test_units"""
import types, numpy as np
import mlx.core as mx, mlx.nn as nn
mx.set_default_device(mx.cpu)
from pt.losses import group_advantages, token_logprobs, rl_loss, sft_loss
from pt.optim import MasterAdamW, lr_at
from pt.common import is_correct, has_format, reward_fn, clean_solution

class Stub(nn.Module):
    def __init__(s, V=11, H=8):
        super().__init__(); s.emb = nn.Embedding(V, H); s.mix = nn.Linear(H, H); s.lm_head = nn.Linear(H, V, bias=False)
        s.model = types.SimpleNamespace(); s.args = types.SimpleNamespace(tie_word_embeddings=False)
    def __call__(s, x): return s.lm_head(s.mix(s.emb(x)))
class Wrap(nn.Module):
    def __init__(s): super().__init__(); s.language_model = Stub(); s.language_model.model = lambda x: s.language_model.mix(s.language_model.emb(x))

def test_advantages():
    r = [1, 0, 0, 0, 1, 1, 0, 1]
    a = group_advantages(r, 4, "grpo").reshape(2, 4); assert abs(a.mean(1)).max() < 1e-9
    b = group_advantages(r, 4, "rloo").reshape(2, 4)
    assert abs(b[0, 0] - (1 - np.mean([0, 0, 0]))) < 1e-9, b   # r_i - mean of others
    z = group_advantages([1, 1, 1, 1], 4, "grpo"); assert np.allclose(z, 0)

def test_logprobs_and_masks():
    m = Wrap(); ids = mx.array([[1, 2, 3, 4, 5, 0], [1, 2, 3, 0, 0, 0]]); pl = mx.array([2, 2]); tl = mx.array([5, 3])
    lp, mask = token_logprobs(m, ids, pl, tl)
    assert mask.sum(1).tolist() == [3.0, 1.0], mask.tolist()           # completion tokens only
    assert float((lp * mask).sum()) < 0

def test_rl_direction():
    """After a gradient step, a positive-advantage sequence must become more likely, negative less."""
    for method in ("grpo", "rloo", "gspo"):
        mx.random.seed(0); m = Wrap(); ids = mx.array([[1, 2, 3, 4, 0], [1, 2, 5, 6, 0]]); pl = mx.array([2, 2]); tl = mx.array([4, 4])
        adv = mx.array([1.0, -1.0])
        f = lambda *_: rl_loss(m, ids, pl, tl, adv, None, method, 0.2, 0.2, 2)
        (loss, aux), g = nn.value_and_grad(m, f)(ids)
        before = [float((x * y).sum()) for x, y in zip(*[t for t in token_logprobs(m, ids, pl, tl)])]
        m.update(__import__("mlx.utils", fromlist=["x"]).tree_map(lambda p, gr: p - 0.5 * gr, m.parameters(), g))
        lp, mk = token_logprobs(m, ids, pl, tl); after = (lp * mk).sum(1).tolist()
        assert after[0] > before[0] and after[1] < before[1], (method, before, after)

def test_sft_loss_decreases():
    m = Wrap(); ids = mx.array([[1, 2, 3, 4, 0]]); pl = mx.array([2]); tl = mx.array([4])
    f = lambda *_: sft_loss(m, ids, pl, tl, 2)
    l0, g = nn.value_and_grad(m, f)(ids)
    m.update(__import__("mlx.utils", fromlist=["x"]).tree_map(lambda p, gr: p - 0.5 * gr, m.parameters(), g))
    assert float(f()) < float(l0)

def test_master_adamw_moves_bf16():
    """bf16 weights + tiny LR: plain bf16 stalls, fp32-master must still move."""
    m = nn.Linear(4, 4, bias=False); m.weight = (mx.ones((4, 4)) * 0.5).astype(mx.bfloat16)
    opt = MasterAdamW(m); g = {"weight": mx.ones((4, 4))}
    for _ in range(2000): opt.update(m, g, 1e-6)
    assert abs(float(m.weight[0, 0]) - 0.5) > 1e-3, m.weight[0, 0]    # 2000*1e-6 = 2e-3 drift, would be 0 in pure bf16

def test_reward_and_clean():
    assert is_correct("so 5+3 = 8\nThe answer is: 8", "8") and has_format("The answer is: 8")
    assert reward_fn("The answer is: 8", "8", True)[0] == 0.0 and reward_fn("The answer is: 8", "8", False)[0] == 1.1
    t, g = clean_solution("He has <<2*3=6>>6 apples.\n#### 1,006"); assert g == "1006" and "<<" not in t and t.endswith("The answer is: 1006")

def test_lr():
    assert lr_at(0, 100, 1.0) < lr_at(4, 100, 1.0) <= 1.0 and lr_at(99, 100, 1.0) >= 0.09

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
