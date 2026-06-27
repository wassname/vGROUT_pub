"""The `scorda` variant: SIGNED CorDA. The rank-2r block adapter (lora.py) whose
QUARANTINE block is seeded toward the reward-hack direction, so the hack is the path of
least resistance into the ablatable block. This is the candidate for absorption-by-default
(out/absorption_repro/spec_v2_training.md): under gate-off `absorb` training the hack should
grow in the quarantine, and deploy ablation should remove it.

Difference from `corda`: corda orients on the UNSIGNED contrast covariance (deltas^T deltas),
which discards hack-vs-clean polarity and splits even/odd by importance, so neither block
leans toward the hack end. scorda uses the SIGNED mean contrast direction and places it, with
its polarity, into the quarantine block only.

Per target Linear, with the per-pair pooled input deltas delta = x_hack - x_clean:

    u_in  = unit(mean_pairs(delta))          [d_in]   signed hack INPUT axis
    u_out = unit(W @ u_in)                    [d_out]  where W maps that axis (the hack OUTPUT
                                                       shift; out ~= W x for a Linear)

The quarantine block's top row/col is the signed seed (u_in, u_out): it reads "how hack-like
is the input" and writes the hack output shift. The remaining quarantine directions and the
whole deployed block are the top-2r right/left singular vectors of W, each orthogonalized
against the hack axis (u_in for the A rows, u_out for the B cols) so the deployed block is
blind to the hack axis. Magnitudes are unit (matched to corda/pissa); the net delta is 0 at
init regardless (A=A0, B=B0, subtracted in the hook).

The seed is rank-1, so it needs only the 8-pair mean and r=32 capacity does NOT force scorda
== pissa (the rank-8-vs-2r=64 confound that sank the unsigned-corda init study).
"""
from __future__ import annotations

import torch
from loguru import logger
from torch import Tensor, nn

from .common import contrastive_input_deltas
from .lora import attach_block_adapter


def _unit(v: Tensor) -> Tensor:
    return v / v.norm().clamp_min(1e-12)


def _unit_rows(M: Tensor) -> Tensor:
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _orth_unit_rows(M: Tensor, u: Tensor) -> Tensor:
    """Remove the u component from every row of M, then unit-normalize the rows. u is unit."""
    return _unit_rows(M - (M @ u)[:, None] * u[None, :])


def wrap_model_with_scorda(
    model: nn.Module, tok, pairs: list, device, r: int = 32,
    grad_probe: bool = False, svd_device: torch.device | str = "cuda",
    mode: str = "quar",
) -> dict[str, dict]:
    """The `scorda` variant: signed hack seed in ONE block, W-SVD (hack-axis removed)
    everywhere else. mode="quar" (default) seeds the QUARANTINE block -> the hack should
    localize into the ablatable block (the hypothesis). mode="dep" seeds the DEPLOYED block
    -> the reverse control: if polarity programs the direction of localization, the hack
    should localize OUT of the quarantine and deploy-ablation should NOT remove it (gap <= 0).
    Both reviewers asked for this directional control."""
    assert mode in ("quar", "dep"), f"scorda mode {mode!r} not in (quar, dep)"
    svd_dev = torch.device(svd_device) if isinstance(svd_device, str) else svd_device
    logger.info(f"scorda calibration ({mode}-seed): {len(pairs)} authored pairs -> signed contrast deltas")
    model.eval()
    deltas = contrastive_input_deltas(model, tok, pairs, device)   # {name: [P, d_in]}
    model.train()

    def init_fn(name: str, i: int, W: Tensor, r_: int) -> tuple[Tensor, Tensor]:
        Wf = W.detach().float().to(svd_dev)
        u_in = _unit(deltas[name].float().to(svd_dev).mean(0))     # [d_in] signed hack axis
        u_out = _unit(Wf @ u_in)                                   # [d_out] hack output shift
        U, _S, Vh = torch.linalg.svd(Wf, full_matrices=False)      # U[d_out,k], Vh[k,d_in]
        Vh_o = _orth_unit_rows(Vh[: 2 * r_], u_in)                 # input dirs, hack-axis removed
        U_o = _orth_unit_rows(U[:, : 2 * r_].T, u_out).T           # output dirs, hack-axis removed
        A0 = torch.empty(2 * r_, Vh.shape[1], device=svd_dev)
        B0 = torch.empty(U.shape[0], 2 * r_, device=svd_dev)
        seed_pos = r_ if mode == "quar" else 0     # top slot of the seeded block
        j = 0                                       # cursor into the orthogonalized SVD fills
        for k in range(2 * r_):
            if k == seed_pos:
                A0[k], B0[:, k] = u_in, u_out       # the signed hack seed
            else:
                A0[k], B0[:, k] = Vh_o[j], U_o[:, j]
                j += 1
        return A0.cpu(), B0.cpu()

    return attach_block_adapter(model, r, init_fn, grad_probe, label=f"scorda-{mode}")
