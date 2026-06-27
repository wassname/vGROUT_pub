"""Shared adapter plumbing: target selection, the SGTM route mask, SVD cache,
contrastive CorDA calibration, and the generic ablate/disable contextmanagers.

Every adapter variant (lora, corda, antipasto) attaches a forward hook per target
Linear and returns an info dict per module. Two layer attributes are UNIFORM across
variants and read by the hooks:

    layer._adp_mask    = (m, d) | None   per-rollout SGTM route mask (set by train.py)
    layer._adp_disable = bool            True -> hook returns base (net delta 0): KL ref

The info dict is the variant-agnostic interface train.py / eval.py consume:

    {layer, handle, r, deployed: [Param], quarantine: [Param], ablate: callable}

`ablate()` resets THIS module's quarantine to its init and returns a restore
closure; ablate_quarantine() below loops it so deploy eval sees the deployed block
alone. Variants differ only in how a module's quarantine is reset (block slice vs
gain/rotation), so each supplies its own `ablate` closure.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from torch import Tensor, nn

TARGET_SUFFIXES = (
    # full attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # linear-attention / GatedDeltaNet
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    # MLP
    "up_proj", "gate_proj", "down_proj",
)


def is_target(name: str) -> bool:
    return name.split(".")[-1] in TARGET_SUFFIXES


def target_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and is_target(n)]


def apply_route_mask(dep: Tensor, quar: Tensor, mask: tuple | None) -> Tensor:
    """Combine deployed + quarantine branch outputs under the SGTM (m, d) mask.

    None -> unmasked (generation / gate / eval). Otherwise mask = (m, d), each [G]:
      m=0 zeroes the quarantine in fwd+bwd (deployed trains in its ablated state);
      d=1 keeps the deployed branch in the forward but detaches its grad (the
      rollout updates only the quarantine). Masks act on branch OUTPUTS so a detach
      blocks grads to every parameter feeding that branch.
    """
    if mask is None:
        return dep + quar
    m, d = mask                                          # [G] each
    G = m.shape[0]
    shape = dep.shape                                    # [G, s, d_out] or [G*s, d_out]
    dep = dep.reshape(G, -1, shape[-1])
    quar = quar.reshape(G, -1, shape[-1])
    d_ = d.view(G, 1, 1).to(dep.dtype)
    dep = ((1 - d_) * dep + d_ * dep.detach()).reshape(shape)
    quar = (m.view(G, 1, 1).to(quar.dtype) * quar).reshape(shape)
    return dep + quar


@contextmanager
def ablate_quarantine(wrappers: dict):
    """Temporarily remove every module's quarantine -> the deployed model (deploy
    eval). Each info dict's `ablate()` resets its quarantine to init and returns a
    restore closure; we run them on enter and undo on exit."""
    restores = {n: info["ablate"]() for n, info in wrappers.items()}
    try:
        yield
    finally:
        for restore in restores.values():
            restore()


@contextmanager
def disable_adapter(wrappers: dict):
    """Force net delta 0 -> the base model M0 the adapter was initialized on: the
    fixed reference policy for the KL anchor (delta 0 at init, so M0 == the RL start
    policy, fixed as the adapter trains)."""
    for info in wrappers.values():
        info["layer"]._adp_disable = True
    try:
        yield
    finally:
        for info in wrappers.values():
            info["layer"]._adp_disable = False


# A "DOF" (degree of freedom) is (param, idx, init): the trainable slice param[idx]
# and its init value param init[idx]. keep_dofs / quar_dofs partition each module's
# trainables into the deployed and quarantine blocks, so grad energy (the qmass split)
# and learned-delta energy are computed without knowing the variant's parametrisation.
Dof = tuple  # (nn.Parameter, index, Tensor)


def grad_sq(dofs: list) -> float:
    s = 0.0
    for p, idx, _ in dofs:
        if p.grad is not None:
            s += p.grad[idx].float().pow(2).sum().item()
    return s


def delta_sq(dofs: list) -> float:
    return sum((p.data[idx] - init[idx]).float().pow(2).sum().item() for p, idx, init in dofs)


def freeze_except(model: nn.Module, trainable_suffixes: tuple[str, ...]) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith(trainable_suffixes))


def svd_cached(
    W: Float[Tensor, "d_out d_in"],
    cache_path: Path,
    device: torch.device,
) -> tuple[Float[Tensor, "d_out k"], Float[Tensor, "k"], Float[Tensor, "k d_in"]]:
    """SVD with disk cache. Compute on `device` in fp32, save fp32 cpu tensors.

    Cache key = sha256(W.cpu fp32 bytes)[:16] in the filename, so any weight change
    invalidates the cache automatically (fail-loud, no silent stale reuse)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    W_fp32 = W.detach().to(torch.float32).cpu().contiguous()
    sha = hashlib.sha256(W_fp32.numpy().tobytes()).hexdigest()[:16]
    final = cache_path.with_suffix(f".{sha}.pt")
    if final.exists():
        d = torch.load(final, map_location="cpu", weights_only=True)
        return d["U"], d["S"], d["Vh"]
    U, S, Vh = torch.linalg.svd(W_fp32.to(device), full_matrices=False)
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    torch.save({"U": U, "S": S, "Vh": Vh}, final)
    logger.debug(f"SVD computed: {final.name}  U={tuple(U.shape)}  S0={S[0]:.3f}  S-1={S[-1]:.3e}")
    return U, S, Vh


@torch.no_grad()
def contrastive_input_deltas(model, tok, pairs: list, device) -> dict[str, Float[Tensor, "P d_in"]]:
    """Per-target-Linear paired input-activation differences -- the CorDA orientation
    data. For each authored (hack, clean) pair we pool the INPUT activation x to every
    target Linear over completion tokens and stack delta = x_hack - x_clean:

        deltas[m] = stack_pairs( pooled(x_hack) - pooled(x_clean) )           [P, d_in]

    The probe is the Linear INPUT, identical for every variant, so the calibration
    never depends on the adapter being fitted. The contrastive covariance is
    deltas^T deltas / P (rank <= P); corda_basis consumes it in low-rank form so no
    [d_in, d_in] matrix is ever materialized.
    """
    targets = target_linears(model)
    names = [n for n, _ in targets]
    pooled: dict[str, Tensor] = {}

    def make_hook(nm):
        def _h(layer, args, out):
            (x,) = args
            pooled[nm] = x[:, model._adp_plen:].detach().float().mean(1)[0]  # [d_in]
        return _h

    handles = [layer.register_forward_hook(make_hook(nm)) for nm, layer in targets]
    deltas: dict[str, list[Tensor]] = {nm: [] for nm in names}
    try:
        for pair in pairs:
            n_prompt = tok(pair.prompt, return_tensors="pt").input_ids.shape[1]
            sides = {}
            for side, completion in (("hack", pair.hack), ("clean", pair.clean)):
                full_ids = tok(pair.prompt + completion, return_tensors="pt").input_ids.to(device)
                assert full_ids.shape[1] - n_prompt > 0, f"pair {pair.problem_id} {side}: no completion"
                model._adp_plen = n_prompt
                model(full_ids, logits_to_keep=1)
                sides[side] = {nm: pooled[nm].clone() for nm in names}
            for nm in names:
                deltas[nm].append((sides["hack"][nm] - sides["clean"][nm]).cpu())
    finally:
        for h in handles:
            h.remove()
        del model._adp_plen
    return {nm: torch.stack(v) for nm, v in deltas.items()}


def save_adapter_tensors(wrappers: dict) -> dict[str, Tensor]:
    """Flatten every module's save_tensors (trainable params + baked frozen basis)
    to a `{tname}/{module}` dict, bf16 cpu. Baked variants (corda/antipasto) need
    the frozen basis saved -- it is calibration-derived, not regenerable from a seed."""
    return {f"{tn}/{mod}": t.detach().to(torch.bfloat16).cpu().contiguous()
            for mod, info in wrappers.items() for tn, t in info["save_tensors"].items()}


def load_adapter_tensors(wrappers: dict, tensors: dict[str, Tensor]) -> None:
    """Copy a save_adapter_tensors dict back into an attached adapter (in place)."""
    for mod, info in wrappers.items():
        for tn, t in info["save_tensors"].items():
            src = tensors[f"{tn}/{mod}"]
            assert src.shape == t.shape, f"{tn}/{mod}: {tuple(src.shape)} != {tuple(t.shape)}"
            dst = t.data if isinstance(t, nn.Parameter) else t
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def corda_basis(
    W: Float[Tensor, "d_out d_in"],
    deltas: Float[Tensor, "P d_in"],
    r2: int,
    device: torch.device,
    eps: float = 1e-3,
) -> tuple[Float[Tensor, "d_out r2"], Float[Tensor, "r2"], Float[Tensor, "r2 d_in"]]:
    """Top-r2 CorDA components of W oriented by the contrastive input covariance.

    C = deltas^T deltas / P + a I (a = eps * tr/d_in) is the input metric. Its sqrt
    has the low-rank closed form C^{1/2} = sqrt(a) I + Ud diag(g) Ud^T, where Ud are
    the orthonormal input directions of `deltas` (from its SVD) and
    g = sqrt(a + lambda) - sqrt(a). We SVD the context-oriented M = W C^{1/2} and
    un-whiten the right vectors back to input space (Vh = Qh C^{1/2}), so the top
    components read the high-contrast input directions W maps most strongly. Net
    delta is subtracted at init regardless, so C^{1/2} (read-emphasis) is the right
    choice here, not C^{-1/2} (exact-W reconstruction).
    """
    W = W.detach().float().to(device)
    D = deltas.float().to(device)                       # [P, d_in]
    P, d_in = D.shape
    a = eps * (D.pow(2).sum() / P) / d_in               # ridge ~ eps * mean input variance
    sqrt_a = a.clamp_min(1e-20).sqrt()
    _, Sig, Vhp = torch.linalg.svd(D, full_matrices=False)   # Vhp:[P,d_in] input dirs
    Ud = Vhp.T                                          # [d_in, P] orthonormal cols
    g = (a + Sig.pow(2) / P).sqrt() - sqrt_a            # [P]
    WUd = W @ Ud                                        # [d_out, P]
    M = sqrt_a * W + (WUd * g) @ Ud.T                   # W C^{1/2}
    U, S, Qh = torch.linalg.svd(M, full_matrices=False)
    k = S.shape[0]
    assert r2 <= k, f"corda needs r2={r2} <= min(d_out,d_in)={k}"
    U2, S2, Qh2 = U[:, :r2], S[:r2], Qh[:r2]
    QUd = Qh2 @ Ud                                      # [r2, P]
    Vh2 = sqrt_a * Qh2 + (QUd * g) @ Ud.T               # Qh2 C^{1/2} -> input space
    return U2.cpu(), S2.cpu(), Vh2.cpu()
