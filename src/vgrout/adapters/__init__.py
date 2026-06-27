"""Adapter variants for vector gradient routing. All share one info-dict interface
(common.py) so train.py / eval.py are variant-agnostic; they differ only in the
parametrisation of the deployed/quarantine blocks:

    lora      rank-2r block adapter, Gaussian init (baseline)
    pissa     rank-2r block adapter, plain top-2r SVD of W, disjoint-axis (no pairs)
    corda     rank-2r block adapter, contrastive-CorDA disjoint-axis init (baked)
    scorda    SIGNED corda: quarantine seeded toward +(mu_hack-mu_clean) (baked)
    antipasto frozen CorDA basis + bounded gains + learned quarantine rotation (baked)

corda/scorda/antipasto need the authored pairs (the contrastive calibration); lora/pissa
do not. pissa is corda's contrast-free twin (same SVD split, W not contrast). scorda is
corda's signed twin: it places the polarity corda's covariance discards into the
quarantine block, the absorption-by-default candidate (spec_v2_training.md).
"""
from __future__ import annotations

from torch import nn

from .common import (  # noqa: F401  (re-exported: the variant-agnostic interface)
    ablate_quarantine, disable_adapter, is_target, load_adapter_tensors,
    save_adapter_tensors, TARGET_SUFFIXES,
)
from .antipasto import wrap_model_with_antipasto
from .corda import wrap_model_with_corda
from .lora import wrap_model_with_lora
from .pissa import wrap_model_with_pissa
from .scorda import wrap_model_with_scorda

VARIANTS = ("lora", "pissa", "corda", "scorda", "scorda_rev", "antipasto")


def wrap_model(
    model: nn.Module, adapter: str, r: int, device, init_seed: int = 0,
    grad_probe: bool = False, tok=None, pairs: list | None = None,
    svd_device="cuda",
) -> dict[str, dict]:
    """Attach the chosen adapter variant. corda/antipasto require (tok, pairs)."""
    if adapter == "lora":
        return wrap_model_with_lora(model, r=r, init_seed=init_seed, grad_probe=grad_probe)
    if adapter == "pissa":
        return wrap_model_with_pissa(model, r=r, grad_probe=grad_probe, svd_device=svd_device)
    if adapter in ("scorda", "scorda_rev"):
        assert pairs is not None and tok is not None, f"{adapter} needs authored pairs for calibration"
        mode = "quar" if adapter == "scorda" else "dep"
        return wrap_model_with_scorda(model, tok, pairs, device, r=r, grad_probe=grad_probe,
                                      svd_device=svd_device, mode=mode)
    if adapter in ("corda", "antipasto"):
        assert pairs is not None and tok is not None, f"{adapter} needs authored pairs for calibration"
        wrap = wrap_model_with_corda if adapter == "corda" else wrap_model_with_antipasto
        return wrap(model, tok, pairs, device, r=r, grad_probe=grad_probe, svd_device=svd_device)
    raise ValueError(f"unknown adapter {adapter!r}, expected one of {VARIANTS}")
