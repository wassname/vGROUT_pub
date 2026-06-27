"""lora2r invariants (rank-2r Gaussian-init LoRA with per-rollout output masks).

Asserts, on tiny-random-qwen3 (CPU, fp32):
  1. IDENTITY AT INIT: wrapped logits == base logits (the hook subtracts the
     frozen A0/B0 init contribution, so net delta is exactly 0).
  2. MASK ROUTING (block grads under each three-way gate label):
       clean (m=0,d=0): deployed-block grads nonzero, quarantine-block ZERO
       hack  (m=1,d=1): deployed-block ZERO (output detach), quarantine nonzero
       mid   (m=1,d=0): both nonzero (absorption)
  3. C-PROBE PER-ROLLOUT RECOVERY: batched c.grad rows == single-rollout c.grad
     (the gate's per-rollout weight grads are exact, not an approximation).
  4. ABLATION TEETH: ablate_quarantine is a no-op at init, removes a quarantine
     perturbation while active, and restores it on exit.

Exit nonzero on any violation. Wired into `just smoke-lora2r`.
"""
import torch
from transformers import AutoModelForCausalLM

from vgrout.adapters import wrap_model_with_lora
from vgrout.eval import ablate_quarantine

MODEL = "llamafactory/tiny-random-qwen3"
R = 4  # tiny model min Linear dim is 16, so 2r=8 fits everywhere

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
ids = torch.randint(100, 1000, (2, 12))

with torch.no_grad():
    base_logits = model(ids).logits.clone()

wrappers = wrap_model_with_lora(model, r=R, grad_probe=True)

# 1. identity at init
with torch.no_grad():
    err = (model(ids).logits - base_logits).abs().max().item()
assert err < 1e-5, f"init not identity: max|dlogits|={err:.2e}"
print(f"1. identity at init OK (max|dlogits|={err:.2e})")


# 2. mask routing
def run_masked(m_val: float, d_val: float) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    g_vec = torch.full((ids.shape[0],), m_val), torch.full((ids.shape[0],), d_val)
    for info in wrappers.values():
        info["layer"]._adp_mask = g_vec
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


dep_n, quar_n = run_masked(0.0, 0.0)  # clean
assert dep_n > 1e-8 and quar_n < 1e-12, f"clean gate: dep={dep_n:.2e} quar={quar_n:.2e}"
print(f"2a. clean (m=0,d=0): dep grad {dep_n:.2e} > 0, quar grad {quar_n:.2e} == 0 OK")
dep_n, quar_n = run_masked(1.0, 1.0)  # hack
assert dep_n < 1e-12 and quar_n > 1e-8, f"hack gate: dep={dep_n:.2e} quar={quar_n:.2e}"
print(f"2b. hack (m=1,d=1): dep grad {dep_n:.2e} == 0, quar grad {quar_n:.2e} > 0 OK")
dep_n, quar_n = run_masked(1.0, 0.0)  # mid
assert dep_n > 1e-8 and quar_n > 1e-8, f"mid gate: dep={dep_n:.2e} quar={quar_n:.2e}"
print(f"2c. mid (m=1,d=0): dep grad {dep_n:.2e} > 0, quar grad {quar_n:.2e} > 0 OK")
model.zero_grad(set_to_none=True)


# 2d. MIXED batch: rollout 0 clean (0,0), rollout 1 hack (1,1) in ONE forward. This
# is the load-bearing per-rollout vectorization (2a-2c only test uniform masks). The
# masks reshape to [G,1,1], so rollout 0 must route to deployed only, rollout 1 to
# quarantine only, with NO bleed. Loss summed over sequences -> per-rollout grads are
# additive and separable, so the mixed deployed grad must equal rollout-0-alone-clean,
# and the mixed quarantine grad must equal rollout-1-alone-hack.
def block_grads(m_vec: torch.Tensor, d_vec: torch.Tensor, batch: torch.Tensor) -> tuple[dict, dict]:
    model.zero_grad(set_to_none=True)
    for info in wrappers.values():
        info["layer"]._adp_mask = (m_vec, d_vec)
    model(batch).logits.float().pow(2).sum().backward()   # sum -> per-sequence additive
    for info in wrappers.values():
        info["layer"]._adp_mask = None
    dep = {n: (i["A"].grad[:i["r"]].clone(), i["B"].grad[:, :i["r"]].clone()) for n, i in wrappers.items()}
    quar = {n: (i["A"].grad[i["r"]:].clone(), i["B"].grad[:, i["r"]:].clone()) for n, i in wrappers.items()}
    return dep, quar


dep_mix, quar_mix = block_grads(torch.tensor([0., 1.]), torch.tensor([0., 1.]), ids)  # r0 clean, r1 hack
dep_r0, _ = block_grads(torch.zeros(1), torch.zeros(1), ids[:1])    # r0 alone, clean
_, quar_r1 = block_grads(torch.ones(1), torch.ones(1), ids[1:])    # r1 alone, hack
for n in wrappers:
    assert torch.allclose(dep_mix[n][0], dep_r0[n][0], atol=1e-5) and \
           torch.allclose(dep_mix[n][1], dep_r0[n][1], atol=1e-5), \
        f"{n}: deployed grad bled across rollouts (mixed != r0-clean-alone)"
    assert torch.allclose(quar_mix[n][0], quar_r1[n][0], atol=1e-5) and \
           torch.allclose(quar_mix[n][1], quar_r1[n][1], atol=1e-5), \
        f"{n}: quarantine grad bled across rollouts (mixed != r1-hack-alone)"
print(f"2d. mixed-batch per-rollout routing OK ({len(wrappers)} modules, r0->deployed r1->quarantine, no bleed)")
model.zero_grad(set_to_none=True)


# 3. per-rollout c-probe recovery
def gate_grads(batch_ids: torch.Tensor) -> list[torch.Tensor]:
    loss = model(batch_ids).logits.float().pow(2).sum()  # sum -> per-sequence-additive
    gates = [info["layer"]._adp_gate for info in wrappers.values()]
    return [g.detach().clone() for g in torch.autograd.grad(loss, gates)]


both = gate_grads(ids)
solo0 = gate_grads(ids[:1])
solo1 = gate_grads(ids[1:])
for name, gb, g0, g1 in zip(wrappers, both, solo0, solo1, strict=True):
    gb2 = gb.reshape(2, -1, gb.shape[-1]).sum(1)              # [2, 2r] per-rollout
    g0r = g0.reshape(1, -1, g0.shape[-1]).sum(1)[0]
    g1r = g1.reshape(1, -1, g1.shape[-1]).sum(1)[0]
    assert torch.allclose(gb2[0], g0r, atol=1e-5, rtol=1e-4), f"{name}: rollout 0 c.grad mismatch"
    assert torch.allclose(gb2[1], g1r, atol=1e-5, rtol=1e-4), f"{name}: rollout 1 c.grad mismatch"
print(f"3. c-probe per-rollout recovery OK ({len(both)} modules, batched == solo)")

# 4. ablation teeth
with torch.no_grad():
    out0 = model(ids).logits.clone()
    with ablate_quarantine(wrappers):
        out_abl_init = model(ids).logits
    assert torch.allclose(out_abl_init, out0, atol=1e-6), "ablate at init is not a no-op"
    for info in wrappers.values():
        r = info["r"]
        info["A"].data[r:] += 0.05 * torch.randn_like(info["A"].data[r:])
    out_pert = model(ids).logits.clone()
    pert = (out_pert - out0).abs().max().item()
    assert pert > 1e-6, f"quarantine perturbation invisible in forward ({pert:.2e})"
    with ablate_quarantine(wrappers):
        out_abl = model(ids).logits
    assert torch.allclose(out_abl, out0, atol=1e-5), "ablation does not remove the quarantine delta"
    out_back = model(ids).logits
    assert torch.allclose(out_back, out_pert, atol=1e-6), "ablate context did not restore state"
print(f"4. ablation teeth OK (perturbation {pert:.2e} visible, removed under ablate, restored after)")

print("verify_lora_routing: ALL OK")
