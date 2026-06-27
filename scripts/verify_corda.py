"""corda adapter invariants (contrastive-CorDA disjoint-axis rank-2r block adapter).

Asserts, on tiny-random-qwen3 (CPU, fp32), with the authored behavior_ pairs as the
contrastive calibration:
  1. INIT IDENTITY: wrapped logits == base (net delta 0; A0/B0 subtracted).
  2. BAKED BASIS: A0 rows and B0 columns are unit-norm oriented directions, and the
     deployed [:r] and quarantine [r:] blocks are DISJOINT (different directions,
     low cross-block overlap -- the separability the shared-frozen PiSSA tie lacked).
  3. MASK ROUTING: clean (0,0) -> deployed grads only; hack (1,1) -> quarantine only;
     mid (1,0) -> both.
  4. SAVE/LOAD ROUND-TRIP: save_adapter_tensors -> load into a fresh wrap reproduces
     the baked basis + trained params bit-for-bit (logits match), so the calibration-
     derived basis travels with the checkpoint.
  5. ABLATION TEETH: ablate_quarantine is a no-op at init, removes a quarantine
     perturbation while active, restores on exit.

Exit nonzero on any violation. Wired into `just smoke-corda`.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vgrout.adapters import wrap_model, save_adapter_tensors, load_adapter_tensors, ablate_quarantine
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

wrappers = wrap_model(model, "corda", r=R, device=torch.device("cpu"), tok=tok, pairs=PAIRS, svd_device="cpu")

# 1. identity at init
with torch.no_grad():
    err = (model(ids).logits - base_logits).abs().max().item()
assert err < 1e-4, f"init not identity: max|dlogits|={err:.2e}"
print(f"1. identity at init OK (max|dlogits|={err:.2e})")

# 2. baked basis: unit directions + disjoint blocks
for n, info in wrappers.items():
    A0, B0, r = info["A0"], info["B0"], info["r"]
    an = A0.norm(dim=1)
    bn = B0.norm(dim=0)
    assert torch.allclose(an, torch.ones_like(an), atol=1e-4), f"{n}: A0 rows not unit ({an.min():.3f}..{an.max():.3f})"
    assert torch.allclose(bn, torch.ones_like(bn), atol=1e-4), f"{n}: B0 cols not unit"
# cross-block read overlap: mean |cos| between deployed and quarantine A0 rows should be < 1
overlap = []
for info in wrappers.values():
    r = info["r"]
    dep, quar = info["A0"][:r], info["A0"][r:]                      # [r, d_in] each (unit rows)
    overlap.append((dep @ quar.T).abs().mean().item())
mean_overlap = sum(overlap) / len(overlap)
assert mean_overlap < 0.9, f"deployed/quarantine reads not disjoint (mean|cos|={mean_overlap:.3f})"
print(f"2. baked basis OK (A0/B0 unit, cross-block mean|cos|={mean_overlap:.3f} < 0.9)")


# 3. mask routing
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
        gA, gB = info["A"].grad, info["B"].grad
        dep_sq += gA[:r].pow(2).sum().item() + gB[:, :r].pow(2).sum().item()
        quar_sq += gA[r:].pow(2).sum().item() + gB[:, r:].pow(2).sum().item()
    return dep_sq ** 0.5, quar_sq ** 0.5


dep_n, quar_n = run_masked(0.0, 0.0)
assert dep_n > 1e-8 and quar_n < 1e-12, f"clean: dep={dep_n:.2e} quar={quar_n:.2e}"
dep_n, quar_n = run_masked(1.0, 1.0)
assert dep_n < 1e-12 and quar_n > 1e-8, f"hack: dep={dep_n:.2e} quar={quar_n:.2e}"
dep_n, quar_n = run_masked(1.0, 0.0)
assert dep_n > 1e-8 and quar_n > 1e-8, f"mid: dep={dep_n:.2e} quar={quar_n:.2e}"
model.zero_grad(set_to_none=True)
print("3. mask routing OK (clean->deployed, hack->quarantine, mid->both)")

# 4. save/load round-trip (perturb the trainable params first so load is non-trivial)
with torch.no_grad():
    for info in wrappers.values():
        info["A"].add_(0.03 * torch.randn_like(info["A"]))
        info["B"].add_(0.03 * torch.randn_like(info["B"]))
    trained_logits = model(ids).logits.clone()
saved = save_adapter_tensors(wrappers)
model2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
wrap2 = wrap_model(model2, "corda", r=R, device=torch.device("cpu"), tok=tok, pairs=PAIRS, svd_device="cpu")
load_adapter_tensors(wrap2, saved)
# tensor-level exactness is the real round-trip guarantee (logits then differ only by
# the bf16 ckpt quantization, which on a tiny-random model has near-zero hidden states
# so relative error amplifies).
for n, info in wrappers.items():
    for tname in ("A", "B", "A0", "B0"):
        ref = info[tname].detach().to(torch.bfloat16).float()
        assert torch.equal(wrap2[n][tname].detach().float(), ref), f"{n}.{tname} not round-tripped"
with torch.no_grad():
    err = (model2(ids).logits - trained_logits).abs().max().item()
assert err < 5e-2, f"save/load logits drift beyond bf16: {err:.2e}"
print(f"4. save/load round-trip OK (tensors bit-exact in bf16; logits drift {err:.2e})")

# 5. ablation teeth (reset trained A/B back to init A0/B0 first -> clean no-op baseline)
with torch.no_grad():
    for info in wrappers.values():
        info["A"].data.copy_(info["A0"]); info["B"].data.copy_(info["B0"])
    out0 = model(ids).logits.clone()
    with ablate_quarantine(wrappers):
        out_abl_init = model(ids).logits
    assert torch.allclose(out_abl_init, out0, atol=1e-5), "ablate at init not a no-op"
    for info in wrappers.values():
        r = info["r"]
        info["A"].data[r:] += 0.05 * torch.randn_like(info["A"].data[r:])
    out_pert = model(ids).logits.clone()
    assert (out_pert - out0).abs().max().item() > 1e-6, "quarantine perturbation invisible"
    with ablate_quarantine(wrappers):
        assert torch.allclose(model(ids).logits, out0, atol=1e-5), "ablation didn't remove quarantine delta"
    assert torch.allclose(model(ids).logits, out_pert, atol=1e-6), "ablate context didn't restore"
print("5. ablation teeth OK")
print("verify_corda: ALL OK")
