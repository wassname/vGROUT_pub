"""Rank-2r block adapter -- the shared mechanism behind the `lora` and `corda`
variants. One adapter per target Linear, A:[2r,d_in] and B:[d_out,2r] both
trainable, split into a deployed block [:r] and a quarantine block [r:]:

    y += B@(A@x) - B0@(A0@x)

A0/B0 are FROZEN init copies, subtracted so the net delta is exactly 0 at init
while h = A@x stays alive (the gate's grad probe needs nonzero A and B). Because
[B|B_q] @ ([A;A_q]@x) has no cross terms, the two blocks are independent adapters;
per-rollout output masks (common.apply_route_mask) route each rollout's gradient
to one block, mirroring SGTM's retain/forget parameter partition.

`lora` and `corda` differ ONLY in the frozen init (init_fn): lora draws iid
Gaussian blocks; corda orients A0/B0 from a contrastive SVD with disjoint singular
axes per block (corda.py). Everything downstream -- hook, masks, ablation, save --
is shared here.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from torch import Tensor, nn

from .common import apply_route_mask, freeze_except, target_linears

InitFn = Callable[[str, int, Tensor, int], tuple[Tensor, Tensor]]   # (name,i,W,r) -> (A0,B0)


def _block_hook(layer: nn.Linear, args: tuple, y: Tensor) -> Tensor:
    (x,) = args
    if layer._adp_disable:                              # KL reference: net delta 0 -> base M0
        return y
    A:  Float[Tensor, "two_r d_in"]  = layer._adp_A
    B:  Float[Tensor, "d_out two_r"] = layer._adp_B
    A0: Float[Tensor, "two_r d_in"]  = layer._adp_A0
    B0: Float[Tensor, "d_out two_r"] = layer._adp_B0
    r = layer._adp_r
    h = F.linear(x, A.to(x.dtype))                      # [..., 2r]
    if layer._adp_grad_probe and torch.is_grad_enabled():
        c = torch.ones(h.shape[0], *([1] * (h.dim() - 2)), h.shape[-1],
                       device=h.device, dtype=h.dtype, requires_grad=True)
        layer._adp_gate = c
        h = h * c
    h0 = F.linear(x, A0.to(x.dtype))                    # frozen init path (subtracted)
    dep = (F.linear(h[..., :r], B[:, :r].to(x.dtype))
           - F.linear(h0[..., :r], B0[:, :r].to(x.dtype)))
    quar = (F.linear(h[..., r:], B[:, r:].to(x.dtype))
            - F.linear(h0[..., r:], B0[:, r:].to(x.dtype)))
    return y + apply_route_mask(dep, quar, layer._adp_mask).to(y.dtype)


def _make_block_ablate(info: dict) -> Callable[[], Callable[[], None]]:
    """Reset this module's quarantine block (A[r:], B[:,r:]) to its init A0/B0 so the
    net quarantine delta is 0; return a closure that restores the trained values.
    (Zeroing the raw params would SUBTRACT the init contribution and corrupt the
    forward; we must reset to A0/B0, not to zero.)"""
    A, B, A0, B0, r = info["A"], info["B"], info["A0"], info["B0"], info["r"]

    def ablate() -> Callable[[], None]:
        saved_a, saved_b = A.data[r:].clone(), B.data[:, r:].clone()
        A.data[r:] = A0[r:]
        B.data[:, r:] = B0[:, r:]

        def restore() -> None:
            A.data[r:] = saved_a
            B.data[:, r:] = saved_b
        return restore
    return ablate


def attach_block_adapter(
    model: nn.Module, r: int, init_fn: InitFn, grad_probe: bool = False, label: str = "lora",
) -> dict[str, dict]:
    """Attach the rank-2r block adapter to every target Linear (in place).

    init_fn(name, i, W, r) returns the frozen (A0:[2r,d_in], B0:[d_out,2r]) for that
    module. A0/B0 are stored as persistent buffers (so a baked/oriented init round-
    trips through save/load); A,B start as clones (net delta 0). Trainable: A, B.
    """
    targets = target_linears(model)
    logger.info(f"{label} attach: {len(targets)} target Linear modules, r={r}/block (2r={2*r})")
    out: dict[str, dict] = {}
    for i, (name, linear) in enumerate(targets):
        d_out, d_in = linear.weight.shape
        dev = linear.weight.device
        A0, B0 = init_fn(name, i, linear.weight.data, r)
        assert A0.shape == (2 * r, d_in), f"{name}: A0 {tuple(A0.shape)} != {(2*r, d_in)}"
        assert B0.shape == (d_out, 2 * r), f"{name}: B0 {tuple(B0.shape)} != {(d_out, 2*r)}"
        A0 = A0.to(device=dev, dtype=torch.float32)
        B0 = B0.to(device=dev, dtype=torch.float32)
        linear.register_buffer("_adp_A0", A0, persistent=True)
        linear.register_buffer("_adp_B0", B0, persistent=True)
        A = nn.Parameter(A0.clone())
        B = nn.Parameter(B0.clone())
        linear.register_parameter("_adp_A", A)
        linear.register_parameter("_adp_B", B)
        linear._adp_r = r
        linear._adp_grad_probe = grad_probe
        linear._adp_gate = None
        linear._adp_mask = None
        linear._adp_disable = False
        dep_col = (slice(None), slice(0, r))
        quar_col = (slice(None), slice(r, None))
        info = {"layer": linear, "A": A, "B": B, "A0": A0, "B0": B0, "r": r,
                "params": [A, B], "save_tensors": {"A": A, "B": B, "A0": A0, "B0": B0},
                # routeA gate read: the deployed block's live r-dim bottleneck A[:r]@x
                "read": (lambda lin=linear, rr=r: lin._adp_A[:rr]),
                "keep_dofs": [(A, slice(0, r), A0), (B, dep_col, B0)],
                "quar_dofs": [(A, slice(r, None), A0), (B, quar_col, B0)],
                "wd": [(A, A0), (B, B0)],
                "handle": linear.register_forward_hook(_block_hook)}
        info["ablate"] = _make_block_ablate(info)
        out[name] = info
    freeze_except(model, ("_adp_A", "_adp_B"))
    return out


def _gaussian_init(name: str, i: int, W: Tensor, r: int) -> tuple[Tensor, Tensor]:
    """iid Gaussian blocks, seeded per module: A0 ~ N(0,1/d_in), B0 ~ N(0,1/2r).
    Blocks are statistically matched; init magnitude beyond 'nonzero' is not
    load-bearing (the net delta is 0 either way)."""
    d_out, d_in = W.shape
    gen = torch.Generator().manual_seed(_gaussian_init.seed * 100003 + i)
    A0 = torch.randn(2 * r, d_in, generator=gen) / d_in ** 0.5
    B0 = torch.randn(d_out, 2 * r, generator=gen) / (2 * r) ** 0.5
    return A0, B0


def wrap_model_with_lora(
    model: nn.Module, r: int = 32, init_seed: int = 0, grad_probe: bool = False,
) -> dict[str, dict]:
    """The `lora` variant: Gaussian-init rank-2r block adapter (the baseline arm)."""
    _gaussian_init.seed = init_seed
    return attach_block_adapter(model, r, _gaussian_init, grad_probe, label="lora")
