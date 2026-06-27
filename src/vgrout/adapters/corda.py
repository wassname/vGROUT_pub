"""The `corda` variant: the rank-2r block adapter (lora.py) with a contrastive
CorDA init instead of Gaussian. Same hook, masks, ablation, save -- only the frozen
A0/B0 directions change, so corda IS lora seeded in an oriented basis.

For each target Linear we take the top-2r CorDA components oriented by the
contrastive input covariance (common.corda_basis), then split them DISJOINTLY
across the two blocks by interleaving on importance:

    deployed  block <- components 0, 2, 4, ...   (even rank)
    quarantine block <- components 1, 3, 5, ...   (odd rank)

A0 rows are the unit-normalized input directions (Vh), B0 columns the unit-
normalized output directions (U). Interleaving gives each block r oriented, equally
important contrastive directions in orthogonal-ish subspaces, so the deployed and
quarantine blocks start reading DIFFERENT hack-relevant features (the disjoint-axis
property the shared-frozen-basis PiSSA tie lacked). Magnitudes are unit (matched to
the Gaussian baseline's per-row scale); the net delta is 0 at init regardless.
"""
from __future__ import annotations

import torch
from loguru import logger
from torch import Tensor, nn

from .common import contrastive_input_deltas, corda_basis
from .lora import attach_block_adapter


def _unit_rows(M: Tensor) -> Tensor:
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def wrap_model_with_corda(
    model: nn.Module, tok, pairs: list, device, r: int = 32,
    grad_probe: bool = False, svd_device: torch.device | str = "cuda",
) -> dict[str, dict]:
    """The `corda` variant. Computes the contrastive deltas once, then bakes a
    per-module disjoint-axis oriented init into the shared block adapter."""
    svd_dev = torch.device(svd_device) if isinstance(svd_device, str) else svd_device
    logger.info(f"corda calibration: {len(pairs)} authored pairs -> contrastive input deltas")
    model.eval()
    deltas = contrastive_input_deltas(model, tok, pairs, device)
    model.train()

    even = list(range(0, 2 * r, 2))
    odd = list(range(1, 2 * r, 2))

    def init_fn(name: str, i: int, W: Tensor, r_: int) -> tuple[Tensor, Tensor]:
        U2, _S2, Vh2 = corda_basis(W, deltas[name], r2=2 * r_, device=svd_dev)  # [d_out,2r],[2r,d_in]
        U2, Vh2 = _unit_rows(U2.T).T, _unit_rows(Vh2)        # unit cols of U, unit rows of Vh
        A0 = torch.empty(2 * r_, Vh2.shape[1])
        B0 = torch.empty(U2.shape[0], 2 * r_)
        A0[:r_], A0[r_:] = Vh2[even], Vh2[odd]               # deployed even, quarantine odd
        B0[:, :r_], B0[:, r_:] = U2[:, even], U2[:, odd]
        return A0, B0

    return attach_block_adapter(model, r, init_fn, grad_probe, label="corda")
