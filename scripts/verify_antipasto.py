"""antipasto adapter invariants (frozen CorDA basis + bounded gains + learned
quarantine rotation).

Asserts, on tiny-random-qwen3 (CPU, fp32), with the authored behavior_ pairs as the
contrastive calibration:
  1. INIT IDENTITY: g=0 -> tanh(0)=0 -> net delta exactly 0 (wrapped == base).
  2. CAYLEY: rotation is identity at Xq=0 and orthogonal for arbitrary Xq.
  3. MASK ROUTING (with gains turned on, since the gain gates every gradient incl.
     the rotation's): clean (0,0) -> deployed gain grad only; hack (1,1) -> quarantine
     gain + rotation grads; mid (1,0) -> both.
  4. SAVE/LOAD ROUND-TRIP: the baked basis (U,S,Vh) + trained (g,Xq) reproduce the
     model -- calibration-derived basis travels with the checkpoint.
  5. ABLATION TEETH: ablate (g_q->0, Xq->0) is a no-op at init, removes a quarantine
     perturbation while active, restores on exit.

Exit nonzero on any violation. Wired into `just smoke-antipasto`.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vgrout.adapters import wrap_model, save_adapter_tensors, load_adapter_tensors, ablate_quarantine
from vgrout.adapters.antipasto import cayley
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

wrappers = wrap_model(model, "antipasto", r=R, device=torch.device("cpu"), tok=tok, pairs=PAIRS, svd_device="cpu")

# 1. identity at init
with torch.no_grad():
    err = (model(ids).logits - base_logits).abs().max().item()
assert err < 1e-4, f"init not identity: max|dlogits|={err:.2e}"
print(f"1. identity at init OK (max|dlogits|={err:.2e})")

# 2. Cayley: identity at 0, orthogonal otherwise
I = torch.eye(R)
assert torch.allclose(cayley(torch.zeros(R, R)), I, atol=1e-6), "cayley(0) != I"
Rrot = cayley(torch.randn(R, R))
assert torch.allclose(Rrot @ Rrot.T, I, atol=1e-5), "cayley(X) not orthogonal"
print("2. Cayley rotation OK (identity at 0, orthogonal)")

# 3. mask routing -- gains on first (the gain gates all quarantine gradients)
with torch.no_grad():
    for info in wrappers.values():
        info["g"].add_(0.1 * torch.randn_like(info["g"]))


def run_masked(m_val: float, d_val: float) -> tuple[float, float, float]:
    model.zero_grad(set_to_none=True)
    mask = (torch.full((2,), m_val), torch.full((2,), d_val))
    for info in wrappers.values():
        info["layer"]._adp_mask = mask
    model(ids).logits.float().pow(2).mean().backward()
    for info in wrappers.values():
        info["layer"]._adp_mask = None
    dep = quar = rot = 0.0
    for info in wrappers.values():
        r = info["r"]
        dep += info["g"].grad[:r].pow(2).sum().item()
        quar += info["g"].grad[r:].pow(2).sum().item()
        rot += info["Xq"].grad.pow(2).sum().item()
    return dep ** 0.5, quar ** 0.5, rot ** 0.5


# rotation grad is gain-gated (prop to s_q) and tiny on a random model, but is exactly
# 0 when the quarantine is masked off -- so the clean/hack separation is clean.
dep_n, quar_n, rot_n = run_masked(0.0, 0.0)
assert dep_n > 1e-8 and quar_n < 1e-12 and rot_n < 1e-12, f"clean: dep={dep_n:.2e} quar={quar_n:.2e} rot={rot_n:.2e}"
dep_n, quar_n, rot_n = run_masked(1.0, 1.0)
assert dep_n < 1e-12 and quar_n > 1e-8 and rot_n > 1e-10, f"hack: dep={dep_n:.2e} quar={quar_n:.2e} rot={rot_n:.2e}"
dep_n, quar_n, rot_n = run_masked(1.0, 0.0)
assert dep_n > 1e-8 and quar_n > 1e-8, f"mid: dep={dep_n:.2e} quar={quar_n:.2e}"
model.zero_grad(set_to_none=True)
print("3. mask routing OK (clean->deployed gain; hack->quarantine gain+rotation; mid->both)")

# 4. save/load round-trip (g already perturbed; perturb Xq too)
with torch.no_grad():
    for info in wrappers.values():
        info["Xq"].add_(0.05 * torch.randn_like(info["Xq"]))
    trained_logits = model(ids).logits.clone()
saved = save_adapter_tensors(wrappers)
model2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
wrap2 = wrap_model(model2, "antipasto", r=R, device=torch.device("cpu"), tok=tok, pairs=PAIRS, svd_device="cpu")
# perturb the forward-used buffers so a no-op load (the stale-cpu-temporary bug) is caught:
# load must overwrite these, and forward() reads layer._adp_{U,S,Vh}, not save_tensors.
with torch.no_grad():
    for info in wrap2.values():
        info["layer"]._adp_U += 0.1
        info["layer"]._adp_Vh += 0.1
load_adapter_tensors(wrap2, saved)
for n, info in wrappers.items():
    for tname, attr in (("U", "_adp_U"), ("S", "_adp_S"), ("Vh", "_adp_Vh"), ("g", "_adp_g"), ("Xq", "_adp_Xq")):
        ref = info["save_tensors"][tname].detach().to(torch.bfloat16).float()
        got = getattr(wrap2[n]["layer"], attr).detach().float()
        assert torch.equal(got, ref), f"{n}.{tname} not round-tripped into the forward buffer"
with torch.no_grad():
    err = (model2(ids).logits - trained_logits).abs().max().item()
assert err < 5e-2, f"save/load logits drift beyond bf16: {err:.2e}"
print(f"4. save/load round-trip OK (tensors bit-exact in bf16; logits drift {err:.2e})")

# 5. ablation teeth (reset gains to 0 -> clean baseline, perturb quarantine only)
with torch.no_grad():
    for info in wrappers.values():
        info["g"].data.zero_(); info["Xq"].data.zero_()
    out0 = model(ids).logits.clone()
    with ablate_quarantine(wrappers):
        assert torch.allclose(model(ids).logits, out0, atol=1e-5), "ablate at init not a no-op"
    for info in wrappers.values():
        r = info["r"]
        info["g"].data[r:] += 0.3 * torch.randn_like(info["g"].data[r:])
        info["Xq"].data += 0.3 * torch.randn_like(info["Xq"].data)
    out_pert = model(ids).logits.clone()
    assert (out_pert - out0).abs().max().item() > 1e-6, "quarantine perturbation invisible"
    with ablate_quarantine(wrappers):
        assert torch.allclose(model(ids).logits, out0, atol=1e-5), "ablation didn't remove quarantine delta"
    assert torch.allclose(model(ids).logits, out_pert, atol=1e-6), "ablate context didn't restore"
print("5. ablation teeth OK")
print("verify_antipasto: ALL OK")
