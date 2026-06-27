"""Per-token log-probs for the GRPO PPO ratio. Imported by train.py.

(The old delta_S gradient-projection/erase machinery lived here too; it was
removed with the PiSSA->lora2r migration -- routing is now block-mask based in
train.py, nothing projects a delta_S grad anymore.)
"""
from __future__ import annotations

import torch


def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids | logits) gathered token-wise.

    Uses F.cross_entropy (fused softmax+gather) so we never materialise the
    full [B, L, V] fp32 softmax. On Qwen3.5-2B with V=152k, G=8, L≈1500 the
    fp32 vocab tensor was ~7 GB per forward -- the difference between OOM and
    fit on a 96 GB card when the autograd graph is alive.
    """
    B, L, V = logits.shape
    # CE's internal log_softmax accumulates in fp32 (stable) but returns input dtype.
    # The output [B*L] is small, so upcast it to fp32 for downstream PPO ratio math.
    return -torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), ids.reshape(-1), reduction="none"
    ).float().view(B, L)
