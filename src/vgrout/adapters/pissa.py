"""The `pissa` variant: the rank-2r block adapter (lora.py) seeded from the plain
top-2r SVD of each target Linear's weight, with the SAME disjoint even/odd block
split as corda. It is the contrast-free twin of corda: corda orients the SVD by the
hack-clean contrast covariance, pissa orients it by W alone, so comparing them
isolates whether the *contrast* orientation matters or any principal-axis SVD init
does. It needs no authored pairs.

NOTE this is NOT shared-basis PiSSA (top-r put in BOTH blocks): that made routing a
magnitude split (the documented shrinkage tie, RESEARCH_JOURNAL 2026). We take the
top-2r principal components of W and split them DISJOINTLY, mirroring corda:

    deployed  block <- components 0, 2, 4, ...   (even rank)
    quarantine block <- components 1, 3, 5, ...   (odd rank)

A0 rows are unit-normalized input directions (Vh), B0 columns unit-normalized output
directions (U). Magnitudes are unit (matched to corda / the Gaussian baseline's
per-row scale; singular values dropped), so the ONLY difference from corda is the
orientation source: W's own principal axes vs the contrastive covariance. Net delta
is 0 at init regardless.
"""
from __future__ import annotations

import torch
from loguru import logger
from torch import Tensor, nn

from .lora import attach_block_adapter


def _unit_rows(M: Tensor) -> Tensor:
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def wrap_model_with_pissa(
    model: nn.Module, r: int = 32, grad_probe: bool = False,
    svd_device: torch.device | str = "cuda",
) -> dict[str, dict]:
    """The `pissa` variant: per-module top-2r SVD of W, disjoint-axis block split."""
    svd_dev = torch.device(svd_device) if isinstance(svd_device, str) else svd_device
    logger.info("pissa init: plain top-2r SVD of each target Linear weight (no pairs)")

    def init_fn(name: str, i: int, W: Tensor, r_: int) -> tuple[Tensor, Tensor]:
        U, _S, Vh = torch.linalg.svd(W.detach().float().to(svd_dev), full_matrices=False)
        even = list(range(0, 2 * r_, 2))
        odd = list(range(1, 2 * r_, 2))
        U2, Vh2 = _unit_rows(U[:, : 2 * r_].T).T, _unit_rows(Vh[: 2 * r_])   # unit cols U, unit rows Vh
        A0 = torch.empty(2 * r_, Vh2.shape[1])
        B0 = torch.empty(U2.shape[0], 2 * r_)
        A0[:r_], A0[r_:] = Vh2[even], Vh2[odd]               # deployed even, quarantine odd
        B0[:, :r_], B0[:, r_:] = U2[:, even], U2[:, odd]
        return A0.cpu(), B0.cpu()

    return attach_block_adapter(model, r, init_fn, grad_probe, label="pissa")
