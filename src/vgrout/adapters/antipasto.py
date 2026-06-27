"""The `antipasto` variant: a FROZEN CorDA-oriented basis with bounded per-direction
gains, plus a learned rotation on the quarantine's input read. W is left intact
(additive delta in the oriented basis, no W_res mutation), so the baked basis
round-trips cleanly and ablation is a parameter reset.

Per target Linear, top-2r CorDA components (U:[d_out,2r], S:[2r], Vh:[2r,d_in])
are frozen buffers, split disjointly into deployed [:r] (even rank) and quarantine
[r:] (odd rank). The delta per block:

    deployed   : y += U_d diag(S_d * tanh(g_d))        (Vh_d x)
    quarantine : y += U_q diag(S_q * tanh(g_q)) ( R_q  (Vh_q x) )

g init 0 -> tanh 0 -> net delta exactly 0 at init. tanh bounds |scale| <= S (each
direction's own singular value), so a gain can attenuate or grow a frozen direction
but not blow it up -- the additive analogue of lora-lite's 1+ELU reweight (centered
at 0 here because the delta is additive, not a reweight of W's retained component).

The quarantine alone gets a learned rotation R_q (Cayley, init identity): it
reorients WHICH input combination the r frozen directions read (within span(Vh_q)),
so the quarantine can lock onto an emergent hack input pattern -- bounded,
interpretable learning (a rotation of a known basis), the capacity a frozen-gain-
only adapter lacks. Deployed stays gain-only (add a deployed rotation later only if
deploy-solve sags). Deploy ablation: g_q -> 0, R_q -> I.

The rotation gradient is gain-gated: dXq grad is proportional to s_q = S_q*tanh(g_q),
so Xq cannot move until the quarantine gain opens (gain learns first, then rotation
refines). Intended -- not a bug -- but means rotation is slow to start under small
quarantine gradients or strong weight decay.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from torch import Tensor, nn

from .common import apply_route_mask, contrastive_input_deltas, corda_basis, freeze_except, target_linears


def cayley(X: Float[Tensor, "r r"]) -> Float[Tensor, "r r"]:
    """Orthogonal matrix from a square param via Cayley: R = (I+A)^{-1}(I-A),
    A = X - X^T (skew). X=0 -> A=0 -> R=I. Always orthogonal, smooth, init-identity."""
    A = X - X.transpose(-1, -2)
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


def _antipasto_hook(layer: nn.Linear, args: tuple, y: Tensor) -> Tensor:
    (x,) = args
    if layer._adp_disable:                              # KL reference: net delta 0 -> base M0
        return y
    U: Float[Tensor, "d_out two_r"] = layer._adp_U
    Vh: Float[Tensor, "two_r d_in"] = layer._adp_Vh
    S: Float[Tensor, "two_r"] = layer._adp_S
    g: Float[Tensor, "two_r"] = layer._adp_g
    Xq: Float[Tensor, "r r"] = layer._adp_Xq
    r = layer._adp_r
    a = F.linear(x, Vh.to(x.dtype))                    # [..., 2r] read in oriented basis
    R = cayley(Xq.float()).to(x.dtype)                 # [r, r] init I
    a_q = a[..., r:] @ R.transpose(-1, -2)             # rotate quarantine read
    s = (S * torch.tanh(g)).to(x.dtype)                # bounded scale, init 0
    dep = F.linear(a[..., :r] * s[:r], U[:, :r].to(x.dtype))
    quar = F.linear(a_q * s[r:], U[:, r:].to(x.dtype))
    return y + apply_route_mask(dep, quar, layer._adp_mask).to(y.dtype)


def _make_antipasto_ablate(info: dict) -> Callable[[], Callable[[], None]]:
    g, Xq, r = info["g"], info["Xq"], info["r"]

    def ablate() -> Callable[[], None]:
        saved_g, saved_x = g.data[r:].clone(), Xq.data.clone()
        g.data[r:] = 0.0
        Xq.data.zero_()

        def restore() -> None:
            g.data[r:] = saved_g
            Xq.data.copy_(saved_x)
        return restore
    return ablate


def wrap_model_with_antipasto(
    model: nn.Module, tok, pairs: list, device, r: int = 32,
    grad_probe: bool = False, svd_device: torch.device | str = "cuda",
) -> dict[str, dict]:
    """The `antipasto` variant: bake a frozen CorDA basis + bounded gains + learned
    quarantine rotation onto every target Linear."""
    if grad_probe:
        raise NotImplementedError("antipasto has no bottleneck c-probe (diag_pinning is lora-only)")
    svd_dev = torch.device(svd_device) if isinstance(svd_device, str) else svd_device
    logger.info(f"antipasto calibration: {len(pairs)} authored pairs -> contrastive CorDA basis")
    model.eval()
    deltas = contrastive_input_deltas(model, tok, pairs, device)
    model.train()

    even = list(range(0, 2 * r, 2))
    odd = list(range(1, 2 * r, 2))
    perm = torch.tensor(even + odd)                    # deployed even, quarantine odd
    targets = target_linears(model)
    logger.info(f"antipasto attach: {len(targets)} target Linear modules, r={r}/block (2r={2*r})")
    out: dict[str, dict] = {}
    for name, linear in targets:
        d_out, d_in = linear.weight.shape
        dev = linear.weight.device
        U2, S2, Vh2 = corda_basis(linear.weight.data, deltas[name], r2=2 * r, device=svd_dev)
        U2, S2, Vh2 = U2[:, perm], S2[perm], Vh2[perm]            # disjoint even/odd split
        # Register the MOVED tensors and reference THOSE in save_tensors -- on CUDA
        # .to(device) makes a new tensor, so save_tensors must point at the registered
        # buffer (the one forward() reads), not the cpu temporary, or load is a no-op.
        U = U2.to(device=dev, dtype=torch.float32)
        S = S2.to(device=dev, dtype=torch.float32)
        Vh = Vh2.to(device=dev, dtype=torch.float32)
        linear.register_buffer("_adp_U", U, persistent=True)
        linear.register_buffer("_adp_S", S, persistent=True)
        linear.register_buffer("_adp_Vh", Vh, persistent=True)
        g = nn.Parameter(torch.zeros(2 * r, device=dev, dtype=torch.float32))
        Xq = nn.Parameter(torch.zeros(r, r, device=dev, dtype=torch.float32))
        linear.register_parameter("_adp_g", g)
        linear.register_parameter("_adp_Xq", Xq)
        linear._adp_r = r
        linear._adp_mask = None
        linear._adp_disable = False
        zg, zx = torch.zeros_like(g), torch.zeros_like(Xq)   # init: delta 0 (g=0, Xq=0)
        info = {"layer": linear, "g": g, "Xq": Xq, "r": r, "params": [g, Xq],
                "save_tensors": {"U": U, "S": S, "Vh": Vh, "g": g, "Xq": Xq},
                # routeA gate read: the deployed block's frozen oriented read Vh[:r]@x
                "read": (lambda lin=linear, rr=r: lin._adp_Vh[:rr]),
                # deployed = gain on even axes; quarantine = gain on odd axes + rotation
                "keep_dofs": [(g, slice(0, r), zg)],
                "quar_dofs": [(g, slice(r, None), zg), (Xq, slice(None), zx)],
                "wd": [(g, zg), (Xq, zx)],
                "handle": linear.register_forward_hook(_antipasto_hook)}
        info["ablate"] = _make_antipasto_ablate(info)
        out[name] = info
    freeze_except(model, ("_adp_g", "_adp_Xq"))
    return out
