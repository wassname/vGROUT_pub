"""scorda (SIGNED corda) adapter invariants -- the absorption-by-default candidate.

scorda reuses corda's rank-2r block hook/mask/ablate/save (verified in verify_corda.py),
so this gate asserts only the SIGNED-SEED invariants that distinguish scorda from corda,
on tiny-random-qwen3 (CPU, fp32) with the authored behavior_ pairs:

  1. INIT IDENTITY: wrapped logits == base (net delta 0).
  2. SIGNED SEED: the quarantine block's top direction is the SIGNED hack axis. For each
     module, A0[r] == unit(mean_pairs(x_hack - x_clean)) and B0[:,r] == unit(W @ that),
     both with POSITIVE alignment (cos ~ +1) -- the polarity corda's covariance discards.
  3. DEPLOYED BLIND TO HACK AXIS: the deployed block rows A0[:r] are ~orthogonal to the
     hack axis u_in (the seed), so clean computation and the hack channel are separated.
  4. MASK ROUTING: clean (0,0)->deployed grads only; hack (1,1)->quarantine only.

Exit nonzero on any violation. Wired into `just smoke-scorda`.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vgrout.adapters import wrap_model
from vgrout.adapters.common import contrastive_input_deltas
from vgrout.pairs import load_pairs

MODEL = "llamafactory/tiny-random-qwen3"
R = 4
PAIRS = load_pairs(Path("data/pairs/hack_pairs.md#all-in-one/behavior_"))[:6]

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
ids = torch.randint(100, 1000, (2, 12))

with torch.no_grad():
    base_logits = model(ids).logits.clone()

wrappers = wrap_model(model, "scorda", r=R, device=torch.device("cpu"), tok=tok, pairs=PAIRS, svd_device="cpu")

# 1. identity at init
with torch.no_grad():
    err = (model(ids).logits - base_logits).abs().max().item()
assert err < 1e-4, f"init not identity: max|dlogits|={err:.2e}"
print(f"1. identity at init OK (max|dlogits|={err:.2e})")

# 2+3. signed seed + deployed orthogonality. Recompute the signed hack axis exactly as the
# init did (delta=0, so input activations on the wrapped model == base; same u_in).
deltas = contrastive_input_deltas(model, tok, PAIRS, torch.device("cpu"))
seed_cos_in, seed_cos_out, dep_overlap = [], [], []
for n, info in wrappers.items():
    A0, B0, r = info["A0"], info["B0"], info["r"]
    W = info["layer"].weight.detach().float()
    u_in = deltas[n].float().mean(0)
    u_in = u_in / u_in.norm().clamp_min(1e-12)
    u_out = W @ u_in
    u_out = u_out / u_out.norm().clamp_min(1e-12)
    seed_cos_in.append(float(A0[r] @ u_in))            # quarantine seed row vs signed axis
    seed_cos_out.append(float(B0[:, r] @ u_out))       # quarantine seed col vs hack output
    dep_overlap.append(float((A0[:r] @ u_in).abs().mean()))   # deployed rows vs hack axis
mn_in, mn_out = min(seed_cos_in), min(seed_cos_out)
assert mn_in > 0.999, f"seed A0[r] not the signed hack axis (min cos={mn_in:.4f}); sign/polarity wrong"
assert mn_out > 0.999, f"seed B0[:,r] not the hack output dir (min cos={mn_out:.4f})"
print(f"2. signed seed OK (min cos A0[r].u_in={mn_in:.4f}, B0[:,r].u_out={mn_out:.4f}, POSITIVE = hack end)")
mean_dep = sum(dep_overlap) / len(dep_overlap)
assert mean_dep < 0.1, f"deployed block not orthogonal to hack axis (mean|cos|={mean_dep:.3f})"
print(f"3. deployed blind to hack axis OK (mean|cos(A0[:r], u_in)|={mean_dep:.3f} < 0.1)")


# 4. mask routing (scorda reuses the lora2r block hook; confirm the split still routes)
def run_masked(m_val: float, d_val: float) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    mask = (torch.full((2,), m_val), torch.full((2,), d_val))
    for info in wrappers.values():
        info["layer"]._adp_mask = mask
    model(ids).logits.float().pow(2).mean().backward()
    for info in wrappers.values():
        info["layer"]._adp_mask = None
    dep_sq = quar_sq = 0.0
    for info in wrappers.values():
        r = info["r"]
        dep_sq += info["A"].grad[:r].pow(2).sum().item() + info["B"].grad[:, :r].pow(2).sum().item()
        quar_sq += info["A"].grad[r:].pow(2).sum().item() + info["B"].grad[:, r:].pow(2).sum().item()
    return dep_sq ** 0.5, quar_sq ** 0.5


dep_n, quar_n = run_masked(0.0, 0.0)
assert dep_n > 1e-8 and quar_n < 1e-12, f"clean: dep={dep_n:.2e} quar={quar_n:.2e}"
dep_n, quar_n = run_masked(1.0, 1.0)
assert dep_n < 1e-12 and quar_n > 1e-8, f"hack: dep={dep_n:.2e} quar={quar_n:.2e}"
model.zero_grad(set_to_none=True)
print("4. mask routing OK (clean->deployed, hack->quarantine)")

# 5. reverse control (scorda_rev): the signed seed must be in the DEPLOYED block (row 0),
# and the quarantine top row must be hack-axis-orthogonal -> deploy ablation can't remove it.
model_r = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
wrap_r = wrap_model(model_r, "scorda_rev", r=R, device=torch.device("cpu"), tok=tok, pairs=PAIRS, svd_device="cpu")
deltas_r = contrastive_input_deltas(model_r, tok, PAIRS, torch.device("cpu"))
dep_seed, quar_top = [], []
for n, info in wrap_r.items():
    r = info["r"]
    u = deltas_r[n].float().mean(0); u = u / u.norm().clamp_min(1e-12)
    dep_seed.append(float(info["A0"][0] @ u))      # deployed row 0 = signed seed
    quar_top.append(float((info["A0"][r] @ u).abs()))   # quarantine top = orthogonalized SVD
assert min(dep_seed) > 0.999, f"scorda_rev seed not in deployed block (min cos={min(dep_seed):.4f})"
assert max(quar_top) < 0.1, f"scorda_rev quarantine not hack-orthogonal (max|cos|={max(quar_top):.3f})"
print(f"5. reverse control OK (seed in DEPLOYED: cos={min(dep_seed):.4f}; quarantine orthogonal)")
print("verify_scorda: ALL OK")
