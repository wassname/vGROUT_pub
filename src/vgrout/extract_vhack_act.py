"""Activation-side per-module v_act extraction (the routeA direction source).

For each authored (hack, clean) pair we forward prompt+completion once per side,
no backward, and capture the deployed bottleneck activation h = A[:r] @ x per
wrapped module, mean-pooled over completion tokens. The per-module direction is
the unit-normalized mean paired difference:

    v_act[m] = unitnorm( mean_pairs( h_hack[m] - h_clean[m] ) )         [r]

Diagnostic basis: RESEARCH_JOURNAL 2026-06-11 (d) -- on the A>0 contrast this
score replicates across three emergence windows (AUROC 0.87/0.75/0.75) while the
gradient score decays to chance; see docs/spec/20260611_act_gate_spec.md.
`tstat=True` divides the mean by its standard error over pairs (clamped |t|<=3)
before normalizing; null at the current 8 pairs, kept for larger pair sets.

ActCapture is the one hook used everywhere acts are read: pair extraction here,
the live gate in train.py, and scripts/verify_v_act.py. It pools inside the hook
(completion-token mean) so nothing of size [B, L, d_in] is retained.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class ActCapture:
    """Forward hooks stashing the pooled deployed bottleneck h = W_read @ x per module.

    W_read is the variant's deployed r-dim read (info["read"](): A[:r] for lora/corda,
    Vh[:r] for antipasto) -- uniform r dims across variants, so the gate machinery is
    variant-agnostic. Call set_pool(plen, comp_mask) before each forward: positions
    < plen are prompt, comp_mask [B, L-plen] marks real completion tokens. pooled()
    returns the completion-token mean act [B, M, r] in module order `names`."""

    def __init__(self, wrappers: dict, names: list[str]):
        self.wrappers, self.names = wrappers, names
        self.h: dict[str, Tensor] = {}
        self.handles = []
        self.plen: int | None = None
        self.mask: Tensor | None = None   # [B, L_c, 1] float
        self.denom: Tensor | None = None  # [B, 1]

    def __enter__(self):
        for nm in self.names:
            layer = self.wrappers[nm]["layer"]

            read = self.wrappers[nm]["read"]

            def hook(layer, args, out, nm=nm, read=read):
                (x,) = args                                            # [B, L, d_in]
                with torch.no_grad():
                    hc = F.linear(x[:, self.plen:].detach(), read().to(x.dtype))
                    self.h[nm] = (hc.float() * self.mask).sum(1) / self.denom  # [B, r]

            self.handles.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def set_pool(self, plen: int, comp_mask: Float[Tensor, "B L_c"]) -> None:
        denom = comp_mask.sum(1)
        assert (denom > 0).all(), f"rollout with 0 completion tokens: {denom.tolist()}"
        self.plen, self.mask, self.denom = plen, comp_mask.unsqueeze(-1), denom.unsqueeze(-1)

    def pooled(self) -> Float[Tensor, "B M r"]:
        return torch.stack([self.h[nm] for nm in self.names], dim=1)


@torch.no_grad()
def extract_v_act(
    model, tok, wrappers: dict, pairs: list, device, tstat: bool = False,
) -> tuple[Float[Tensor, "M r"], dict[str, Float[Tensor, "P M r"]]]:
    """Forward each pair side once -> (v_act unit rows, raw per-pair pooled acts).

    Module order is sorted(wrappers) (= the diag scripts' convention). Tokenization
    matches the cached diagnostic features: full = tok(prompt+completion), pool
    positions >= len(tok(prompt)). Caller owns model.eval() and quarantine state."""
    names = sorted(wrappers)
    acts: dict[str, list[Tensor]] = {"hack": [], "clean": []}
    with ActCapture(wrappers, names) as cap:
        for pair in pairs:
            n_prompt = tok(pair.prompt, return_tensors="pt").input_ids.shape[1]
            for side, completion in (("hack", pair.hack), ("clean", pair.clean)):
                full_ids = tok(pair.prompt + completion,
                               return_tensors="pt").input_ids.to(device)
                L_c = full_ids.shape[1] - n_prompt
                assert L_c > 0, f"pair {pair.problem_id} {side}: no completion tokens"
                cap.set_pool(n_prompt, torch.ones(1, L_c, device=device))
                # logits_to_keep=1: the decoder Linears (whose inputs we hook) run on
                # every position regardless; skipping the full-seq lm_head is free.
                model(full_ids, logits_to_keep=1)
                acts[side].append(cap.pooled()[0].cpu())
    H = torch.stack(acts["hack"])                                       # [P, M, r]
    C = torch.stack(acts["clean"])
    D = H - C
    d = D.mean(0)
    if tstat:
        se = D.std(0) / D.shape[0] ** 0.5
        d = (d / se.clamp_min(1e-12)).clamp(-3.0, 3.0)
    v = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)               # unit per module
    return v, {"hack": H, "clean": C}


def haar_unit_rows(shape: tuple[int, int], seed: int) -> Float[Tensor, "M r"]:
    """Random unit rows matching v_act's shape -- the placebo directionality control."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(shape, generator=g)
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
