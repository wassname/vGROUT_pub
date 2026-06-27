"""GRPO / Dr.GRPO loop with per-rollout output masking on the LeetCode
reward-hacking benchmark.

  generate -> grade -> backward -> (gate) -> masked backward -> step

Inner GRPO step follows the simple_GRPO reference; the outer loop accumulates
grads over prompts_per_step prompts, so at least one per-prompt group has reward
variance.
Unbiased normalization: Dr.GRPO, Liu et al. 2025, arXiv:2503.20783 -- drop the
1/|oᵢ| length norm and the /σ_R group-std (--unbiased, on by default).

Adapter: lora2r (src/vgrout/lora2r.py) -- one rank-2r LoRA per Linear, A and B
both trainable, partitioned into a deployed block [:r] and a quarantine block
[r:]. The quarantine is ablated (reset to its frozen init) at deployment.

Arms (--intervention):
  none    no routing gate: trains the full rank-2r adapter as the vanilla
          control.
  routeA  per-rollout three-way gate from the pooled bottleneck activation vs
          v_act: keep->deployed-only, rout->quarantine-only (deployed detached),
          absorb->both, which may permit absorption. The acts ride the no-grad
          logpi_old forward, so routeA costs roughly the vanilla arm.
  absorb  gate pinned mid (1,0): both blocks train on everything, no gate --
          tests ungated both-block training.

  uv run python -m vgrout.train smoke --intervention=routeA
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import random
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

# Must be set BEFORE `import torch` to take effect on the CUDA allocator.
# Eliminates fragmentation that caused 91 GiB allocated / 581 MiB free crash
# on Qwen3-4B G=8 (PyTorch's own OOM message recommends this).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from loguru import logger
from safetensors.torch import save_file
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .extract_vhack_act import ActCapture, extract_v_act, haar_unit_rows
from .extract_vhack_grad import extract_v_grad, gen_motivate_pairs
from .adapters import wrap_model, save_adapter_tensors
from .adapters.common import grad_sq, delta_sq
from .pairs import load_pairs
from .proj import per_token_logps
from small_reward_hacking.rewards import EnvMode, compute_reward
from small_reward_hacking.data import DATA, load_problems
from .eval import ablate_quarantine, disable_adapter, eval_hack_solve, load_eval_splits
from .tablelog import setup_logging, StepLogger
from .run_artifacts import RUN_SCHEMA
from .train_config import Config, FastConfig, SmokeConfig

OUT_DIR = Path("out")
RUNS_DIR = OUT_DIR / "runs"


class TrackMaxGpu:
    """Background NVML sampler: per-stage MEAN/peak SM-utilization and peak memory.

    torch.cuda.utilization() is an INSTANTANEOUS read, so a post-block read catches only
    the idle tail -- we sample in a daemon thread DURING the work. Each sample is bucketed
    by the current stage label (set via .mark(stage) / cleared via .end()), so interleaved
    gen/fb regions get separate stats without re-indenting them into one `with`.

    MEAN util is the efficiency signal, NOT max: max is ~100% for almost any GPU workload
    (every stage has moments of full kernel occupancy), so a low MEAN -- GPU idle between
    Python-dispatched per-token kernels -- is what reveals gen starvation. Peak memory
    (per stage, from reset_peak_memory_stats at mark) answers headroom-to-grow-batch.
    One-shot step-0 diagnostic. util degrades to n/a if NVML is absent (nvidia-ml-py
    missing) -- a diagnostic must never kill a multi-hour run.
    CAVEAT: peak memory is PER-PROCESS (clean), but utilization is DEVICE-WIDE NVML, so on
    a shared GPU it includes co-tenant processes; only dedicated-GPU runs (one Modal
    container) read util cleanly.

    Usage: g = TrackMaxGpu(device).start(); g.mark("gen"); ...; g.end(); g.stop().
    Also a context manager for a single block: `with TrackMaxGpu(dev) as g: ...`."""

    def __init__(self, device, period_s: float = 0.01):
        self.device = device
        self.period_s = period_s
        self.stage: str | None = None
        self.util_sum: dict[str, int] = {}
        self.util_n: dict[str, int] = {}
        self.util_max: dict[str, int] = {}
        self.peak_gb: dict[str, float] = {}
        try:
            torch.cuda.utilization(device)
            self._util_ok = True
        except Exception:
            self._util_ok = False

    def start(self) -> "TrackMaxGpu":
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            s = self.stage
            if s is not None and self._util_ok:
                u = torch.cuda.utilization(self.device)
                self.util_sum[s] = self.util_sum.get(s, 0) + u
                self.util_n[s] = self.util_n.get(s, 0) + 1
                self.util_max[s] = max(self.util_max.get(s, 0), u)

    def mark(self, stage: str) -> None:
        """Start attributing samples to `stage` and reset its peak-memory high-water mark."""
        torch.cuda.reset_peak_memory_stats(self.device)
        self.stage = stage

    def end(self) -> None:
        """Capture the current stage's peak memory and stop attributing util to it."""
        if self.stage is not None:
            self.peak_gb[self.stage] = max(
                self.peak_gb.get(self.stage, 0.0),
                torch.cuda.max_memory_allocated(self.device) / 1e9)
            self.stage = None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self.peak_gb:
            logger.info(f"GPU per-stage: {self.summary()}  (gen util_mean low => per-token "
                        f"Python hook dispatch starves gen, grow gen batch; mem% << 100 => "
                        f"headroom; util max~100% is normal, ignore it)")

    def summary(self) -> str:
        tot = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        parts = []
        for s in self.peak_gb:
            u = (f"{self.util_sum[s] / self.util_n[s]:.0f}%(max {self.util_max[s]}%)"
                 if self.util_n.get(s) else "n/a")
            gb = self.peak_gb[s]
            parts.append(f"{s} util_mean={u} mem_peak={gb:.1f}/{tot:.0f}GB ({100 * gb / tot:.0f}%)")
        return " | ".join(parts)

    def __enter__(self) -> "TrackMaxGpu":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _sample_rows(rows: list[dict] | None, n: int, rng: torch.Generator) -> list[dict]:
    """Draw n teacher rollouts from a prompt's pool (with replacement if the pool is short)."""
    if n == 0 or not rows:
        return []
    idxs = torch.randperm(len(rows), generator=rng)[:n].tolist()
    if len(rows) < n:
        idxs += torch.randint(0, len(rows), (n - len(rows),), generator=rng).tolist()
    return [rows[i] for i in idxs]


def _pct(xs: list[float], q: float) -> float:
    """Empirical q-quantile of a score list (linear interpolation; nan if empty)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo = int(i)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (i - lo) * (s[lo + 1] - s[lo])


def _auroc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based AUROC (Mann-Whitney U) of `scores` as a detector of the positive class.

    Higher score for hacks -> auroc > 0.5. nan if either class is absent this step.
    Diagnostic only: ground-truth labels measure how well the gate score separates
    reward-hacking updates, but never determine a route. Reading: ~0.5 means v_act
    is a chance-level classifier (no threshold can route reliably); high AUROC but
    rout~0 = the threshold/scale is wrong, not the direction; a drop across a refresh =
    the refresh destroyed the separation."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):                      # average-rank tie handling
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1                   # 1-based mean rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _fbeta_at(scores: list[float], labels: list[bool], thi: float, beta: float = 0.5) -> float:
    """Precision-weighted F_beta of the rout cut: the operating-point companion to _auroc.

    Same inputs as _auroc plus the live rout threshold `thi` (already computed by the
    gate), so it is as cheap to log. AUROC is threshold-free separability; this is the
    quality of the actual pin at score >= thi. beta=0.5 weights precision: per the pin-cost
    model a clean rollout wrongly routed loses capability (the solve update is ablated away),
    while a missed hack is absorbed -- so the rout cut should pin only confident hacks.
    Measurement only: uses hack labels, never determines a route. nan during warmup (thi nan)
    or when no rollout is routed / no hack exists this step (like _auroc on an absent class)."""
    if thi != thi:                                  # nan threshold -> warmup / boundary step
        return float("nan")
    routed = [s >= thi for s in scores]
    if not any(routed) or not any(labels):
        return float("nan")
    prec = sum(y for r, y in zip(routed, labels) if r) / sum(routed)
    rec = sum(s >= thi for s, y in zip(scores, labels) if y) / sum(labels)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec) if (prec + rec) > 0 else float("nan")


# Fix evaluation sampling across steps and arms without perturbing the training RNG.
EVAL_GEN_SEED = 12345

# 2-char env_mode codes for compact per-mode hack columns (hk_rt, hk_xc, ...).
MODE_CODE: dict[str, str] = {
    "run_tests": "rt", "eq_override": "eq", "exit_code": "xc",
    "stdout_marker": "so", "sentinel": "se", "file_marker": "fm",
    "gt_only": "gt",
}


def _validate_config(cfg: Config) -> None:
    """Reject contradictory experiment settings before model load."""
    if cfg.intervention not in ("none", "routeA", "routeV", "absorb"):
        raise ValueError(f"unknown intervention {cfg.intervention!r}; expected none|routeA|routeV|absorb")
    if cfg.routeA_random_v_seed is not None and cfg.intervention != "routeA":
        raise ValueError("routeA_random_v_seed is a routeA-only placebo control")
    if cfg.routeV_random_v_seed is not None and cfg.intervention != "routeV":
        raise ValueError("routeV_random_v_seed is a routeV-only placebo control")
    if not 0.0 <= cfg.gen_deploy_frac <= 1.0:
        raise ValueError(f"gen_deploy_frac must be in [0,1], got {cfg.gen_deploy_frac}")
    if cfg.solve_pool_dir is not None:
        if cfg.teacher_pool_dir is None or cfg.mix_ratio <= 0:
            raise ValueError("solve_pool_dir splits the G_t teacher budget -- needs teacher_pool_dir + mix_ratio>0")
        if not (0.0 <= cfg.solve_mix_frac <= 1.0):
            raise ValueError(f"solve_mix_frac must be in [0,1]; got {cfg.solve_mix_frac}")
    if cfg.retain_heal_epochs < 0:
        raise ValueError(f"retain_heal_epochs must be >=0, got {cfg.retain_heal_epochs}")
    if cfg.retain_heal_batch_size <= 0:
        raise ValueError(f"retain_heal_batch_size must be >0, got {cfg.retain_heal_batch_size}")
    if cfg.retain_heal_lr is not None and cfg.retain_heal_lr <= 0:
        raise ValueError(f"retain_heal_lr must be >0 when set, got {cfg.retain_heal_lr}")


def _log_resolved_config(cfg: Config, device) -> None:
    """Auto-dump EVERY config field so a detached log shows exactly what ran. Uses vars(cfg)
    so no hand-maintained subset can silently drop a field (the old curated block missed
    route_tail_q, gen_deploy_frac, eval_ablate_every, etc.)."""
    items = {"preset/arm": f"{cfg.preset_name} / {cfg.arm}", "device": str(device), **vars(cfg)}
    width = max(len(k) for k in items)
    block = "\n".join(f"    {k:<{width}} : {v}" for k, v in items.items())
    logger.info(f"resolved config:\n{block}")


def _retain_heal_clean_pairs(
    model,
    tok,
    wrappers: dict,
    pairs: list,
    device: torch.device,
    cfg: Config,
) -> dict:
    """Post-training NLL on clean pair completions, deployed block only."""
    if cfg.retain_heal_epochs == 0:
        return {"enabled": False, "epochs": 0, "pairs": 0, "tokens": 0}
    if pairs is None:
        raise ValueError("retain-heal needs cfg.vhack_pairs_path to be loaded")

    rows = []
    for pair in pairs:
        prompt_ids = tok(pair.prompt, add_special_tokens=False).input_ids
        full_ids = tok(pair.prompt + pair.clean, add_special_tokens=False).input_ids
        n_completion = len(full_ids) - len(prompt_ids)
        if n_completion <= 0:
            raise ValueError(f"retain-heal pair {pair.problem_id}: clean side has no tokens")
        rows.append({"problem_id": pair.problem_id, "ids": full_ids, "prompt_len": len(prompt_ids)})

    heal_lr = cfg.retain_heal_lr if cfg.retain_heal_lr is not None else cfg.lr
    opt_heal = torch.optim.AdamW(
        [p for info in wrappers.values() for p in info["params"]],
        lr=heal_lr,
        weight_decay=0.0,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    total_completion_tokens = sum(len(r["ids"]) - r["prompt_len"] for r in rows)
    total_seen_tokens = 0
    last_keep_grad = last_quar_grad = 0.0
    loss_sum = 0.0
    model.train()
    logger.info(
        f"retain-heal: {len(rows)} clean pairs from {cfg.vhack_pairs_path}, "
        f"{total_completion_tokens} completion tokens, epochs={cfg.retain_heal_epochs}, "
        f"batch={cfg.retain_heal_batch_size}, lr={heal_lr:g}. "
        f"SHOULD: quarantine_grad=0 exactly; ELSE heal leaked into quarantine.")

    for epoch in range(cfg.retain_heal_epochs):
        order = torch.randperm(len(rows), device="cpu").tolist()
        for start in range(0, len(order), cfg.retain_heal_batch_size):
            batch_rows = [rows[i] for i in order[start:start + cfg.retain_heal_batch_size]]
            max_len = max(len(r["ids"]) for r in batch_rows)
            input_ids = torch.full((len(batch_rows), max_len), tok.pad_token_id,
                                   dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            n_loss_tokens = 0
            for i, row in enumerate(batch_rows):
                ids = torch.tensor(row["ids"], dtype=torch.long, device=device)
                n = ids.numel()
                input_ids[i, :n] = ids
                attention_mask[i, :n] = 1
                labels[i, row["prompt_len"]:n] = ids[row["prompt_len"]:n]
                n_loss_tokens += n - row["prompt_len"]

            opt_heal.zero_grad(set_to_none=True)
            m = torch.zeros(len(batch_rows), device=device)
            d = torch.zeros(len(batch_rows), device=device)
            for info in wrappers.values():
                info["layer"]._adp_mask = (m, d)
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, use_cache=False)
                loss = out.loss
                loss.backward()
            finally:
                for info in wrappers.values():
                    info["layer"]._adp_mask = None

            sq_keep = sq_quar = 0.0
            for info in wrappers.values():
                sq_keep += grad_sq(info["keep_dofs"])
                sq_quar += grad_sq(info["quar_dofs"])
            last_keep_grad, last_quar_grad = sq_keep ** 0.5, sq_quar ** 0.5
            if last_keep_grad <= 0.0:
                raise RuntimeError("retain-heal produced zero deployed-block gradient")
            if last_quar_grad != 0.0:
                raise RuntimeError(f"retain-heal leaked quarantine gradient {last_quar_grad:.3e}")
            torch.nn.utils.clip_grad_norm_([p for info in wrappers.values() for p in info["params"]],
                                           cfg.grad_clip)
            opt_heal.step()
            loss_sum += loss.item() * n_loss_tokens
            total_seen_tokens += n_loss_tokens

        logger.info(
            f"retain-heal epoch {epoch + 1}/{cfg.retain_heal_epochs}: "
            f"nll={loss_sum / max(1, total_seen_tokens):.4f} "
            f"deployed_grad={last_keep_grad:.3e} quarantine_grad={last_quar_grad:.3e}")

    return {
        "enabled": True,
        "epochs": cfg.retain_heal_epochs,
        "pairs": len(rows),
        "tokens": total_completion_tokens,
        "nll": loss_sum / max(1, total_seen_tokens),
        "deployed_grad": last_keep_grad,
        "quarantine_grad": last_quar_grad,
        "lr": heal_lr,
    }


def main(cfg: Config) -> int:
    _validate_config(cfg)
    model_name = cfg.model; steps = cfg.steps; group = cfg.group
    max_new = cfg.max_new; n_problems = cfg.n_problems
    prompts_per_step = cfg.prompts_per_step
    lr = cfg.lr; adam_beta1 = cfg.adam_beta1; adam_beta2 = cfg.adam_beta2

    run_id = f"{cfg.preset_name}_{cfg.arm}_seed{cfg.seed}{cfg.out_tag}"
    verbose_log = setup_logging(run_id)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Log enough run identity up front to interpret detached logs.
    logger.info(f"argv: {' '.join(sys.argv)}")
    logger.info(f"verbose log: {verbose_log}")
    _log_resolved_config(cfg, device)

    is_routeA = cfg.intervention == "routeA"
    is_routeV = cfg.intervention == "routeV"
    is_absorb = cfg.intervention == "absorb"
    is_vanilla = cfg.intervention == "none"
    has_quarantine = is_routeA or is_routeV or is_absorb

    # Only adapter parameters train; the base model remains frozen.
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    # ── model + tokenizer ──
    # CPU smoke: fp32 + sdpa (flash-attn2 is CUDA-only, CPU bf16 is patchy).
    # GPU: bf16 + flash_attention_2.
    cpu = device.type == "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32 if cpu else torch.bfloat16,
        attn_implementation="sdpa" if cpu else "flash_attention_2",
    ).to(device)
    # Generation enables KV cache; loss forwards disable it to avoid unused state.
    model.config.use_cache = False

    # ── adapter: deployed block [:r] + quarantine block [r:] (variant per cfg.adapter) ──
    # The routeA gate reads activations via forward hooks; no grad probe in training
    # (the c-probe stays in lora only for scripts/diag_pinning.py diagnostics). corda /
    # antipasto calibrate their baked basis on the SAME authored pairs routeA uses.
    needs_pairs = is_routeA or cfg.adapter in ("corda", "antipasto") or cfg.retain_heal_epochs > 0
    MASK_PAIRS = None
    if needs_pairs:
        # Authored pairs are the only routing-label source; live oracle labels never enter training.
        MASK_PAIRS = load_pairs(cfg.vhack_pairs_path)
        logger.info(f"authored pairs: {cfg.vhack_pairs_path} -> {len(MASK_PAIRS)} pairs")
    # routeV routes on the c-probe GRADIENT, so the adapter must splice the identity probe
    # (grad_probe=True) -- the deployed-block gate grad is read from _adp_gate.grad.
    wrappers = wrap_model(
        model, cfg.adapter, r=cfg.lora_r, device=device, init_seed=cfg.lora_init_seed,
        grad_probe=is_routeV, tok=tok, pairs=MASK_PAIRS, svd_device="cpu" if cpu else "cuda")
    # quarantine = block slices (lora/corda) or odd-rank gain/rotation (antipasto); per-
    # rollout masks route grads to one block, so there is no separate hack-param list.
    delta_params = [p for info in wrappers.values() for p in info["params"]]
    logger.info(f"trainable {cfg.adapter} params: {sum(p.numel() for p in delta_params):,} "
                f"across {len(wrappers)} modules")

    # ── routeA direction: v_act (mean pooled-act pair diff, unit rows per module) ──
    v_act = None                    # [M, r] cpu fp32; module order = act_names
    act_names = sorted(wrappers)
    act_buf: deque | None = None    # rolling pooled acts [M, r]; re-scored vs the CURRENT
    #                                 v_act at each gate call, so a refresh needs no flush
    if is_routeA:
        model.eval()   # deterministic forward, no dropout
        v_act, pair_acts = extract_v_act(model, tok, wrappers, MASK_PAIRS, device,
                                         tstat=cfg.vact_tstat)
        model.train()
        # Authored-pair separation in the live score. Two distinct signals:
        # paired accuracy (s_hack_i > s_clean_i, within the SAME prompt) is the extraction
        # sanity check; unpaired AUROC mixes in between-prompt score offset, which the live
        # gate also faces (raw scores, no per-prompt centering), so it forecasts gate noise.
        # Measured 2026-06-12 on Qwen3-4B/behavior_: paired 8/8, unpaired 0.875 -- every
        # inversion was offset (scripts/diag_pair_auroc.py), not a misranked pair.
        sh = torch.einsum("pmr,mr->p", pair_acts["hack"], v_act)
        sc = torch.einsum("pmr,mr->p", pair_acts["clean"], v_act)
        pair_auroc = _auroc(torch.cat([sh, sc]).tolist(),
                            [True] * len(sh) + [False] * len(sc))
        paired_acc = (sh > sc).float().mean().item()
        logger.info(
            f"routeA v_act: {v_act.shape[0]} modules x r={v_act.shape[1]} "
            f"(tstat={cfg.vact_tstat}); authored-pair dot gap={(sh.mean() - sc.mean()).item():+.3e}, "
            f"paired acc={int((sh > sc).sum())}/{len(sh)}, unpaired AUROC={pair_auroc:.2f}. "
            f"SHOULD: paired acc=1.0 ELSE extraction/pairset broken; "
            f"AUROC<1 at acc=1.0 = between-prompt offset only (gate noise, not breakage).")
        if paired_acc < 1.0:
            logger.warning(f"routeA v_act: paired acc={paired_acc:.2f} < 1.0 -- some pair is "
                           f"INVERTED vs the v built from it; run scripts/diag_pair_auroc.py "
                           f"to identify it (mislabeled side or style confound)")
        if cfg.routeA_random_v_seed is not None:
            v_act = haar_unit_rows(tuple(v_act.shape), cfg.routeA_random_v_seed)
            logger.info(f"routeA: OVERRODE v_act with Haar-random unit rows "
                        f"(seed={cfg.routeA_random_v_seed}) -- placebo directionality control")
        act_buf = deque(maxlen=cfg.route_buffer)
        buf_stamp = deque(maxlen=cfg.route_buffer)  # insertion index per act, for EMA ages
        buf_seen = 0                                # total acts ever inserted
        _ema = (f"EMA halflife={cfg.route_ema_halflife} rollouts" if cfg.route_ema_halflife
                else "unweighted")
        logger.info(
            f"routeA gate: per-rollout score = dot(pooled completion-token act, v_act), "
            f"thresholds = global ({_ema}) {cfg.route_tail_q}/{1 - cfg.route_tail_q} quantiles of the "
            f"run-spanning act buffer (<= {cfg.route_buffer}, re-scored vs current v_act), "
            f"label-free; pinned absorb until {cfg.route_warmup} scores. keep (0,0) | "
            f"absorb (1,0) | rout (1,1: deployed detached). A false rout only discards one "
            f"update from deployment. "
            f"SHOULD: auroc col >> 0.5 once hacks appear ELSE v_act is blind and routing "
            f"is noise; rout tracks the hack share, not ~0 or ~1.")

    # ── routeV direction: v_grad (mean c-probe-grad pair diff, unit rows per module) ──
    # The grad-space twin of v_act. The motivate self-gen pairs are generated ONCE from
    # the warm-start model (oracle-free: only the prompt framing labels a side); v_grad is
    # re-extracted on those FIXED texts every refresh so it tracks the policy without
    # regenerating. The buffer holds per-rollout GRAD features and re-scores them vs the
    # current v_grad, so a refresh needs no flush (same machinery as routeA's act buffer).
    v_grad = None                   # [M, r] cpu fp32; module order = act_names
    SELF_PAIRS = None
    grad_buf: deque | None = None
    if is_routeV:
        model.eval()
        with ablate_quarantine(wrappers):
            SELF_PAIRS = gen_motivate_pairs(
                model, tok, device, n_pairs=cfg.routeV_n_pairs, pool=cfg.routeV_pool,
                require_marker=cfg.routeV_require_marker, seed=cfg.seed, max_new=cfg.max_new)
            v_grad, pair_grads = extract_v_grad(model, tok, wrappers, SELF_PAIRS, device)
        model.train()
        # Authored-pair separation analog: paired accuracy on the grad score (hack > clean
        # within the same pair) is the extraction sanity check.
        sh = torch.einsum("pmr,mr->p", pair_grads["hack"], v_grad)
        sc = torch.einsum("pmr,mr->p", pair_grads["clean"], v_grad)
        pair_auroc = _auroc(torch.cat([sh, sc]).tolist(), [True] * len(sh) + [False] * len(sc))
        paired_acc = (sh > sc).float().mean().item()
        logger.info(
            f"routeV v_grad: {v_grad.shape[0]} modules x r={v_grad.shape[1]} from "
            f"{len(SELF_PAIRS)} motivate self-gen pairs; grad dot gap="
            f"{(sh.mean() - sc.mean()).item():+.3e}, paired acc={int((sh > sc).sum())}/{len(sh)}, "
            f"unpaired AUROC={pair_auroc:.2f}.")
        if cfg.routeV_random_v_seed is not None:
            v_grad = haar_unit_rows(tuple(v_grad.shape), cfg.routeV_random_v_seed)
            logger.info(f"routeV: OVERRODE v_grad with Haar-random unit rows "
                        f"(seed={cfg.routeV_random_v_seed}) -- placebo directionality control")
        grad_buf = deque(maxlen=cfg.route_buffer)
        buf_stamp = deque(maxlen=cfg.route_buffer)
        buf_seen = 0
        _ema = (f"EMA halflife={cfg.route_ema_halflife} rollouts" if cfg.route_ema_halflife
                else "unweighted")
        logger.info(
            f"routeV gate: per-rollout score = dot(deployed c-probe NLL grad, v_grad), "
            f"thresholds = global ({_ema}) {cfg.route_tail_q}/{1 - cfg.route_tail_q} quantiles of the "
            f"run-spanning grad buffer (<= {cfg.route_buffer}, re-scored vs current v_grad), "
            f"label-free; pinned absorb until {cfg.route_warmup} scores. keep (0,0) | "
            f"absorb (1,0) | rout (1,1: deployed detached). Costs a 2nd grad forward+backward "
            f"per group for the score. "
            f"SHOULD: auroc col >> 0.5 once hacks appear ELSE v_grad is blind and routing is noise.")

    # ── teacher pool ──
    # Teacher pool: pre-generated rollouts on disk keyed by problem_id. Each step's
    # G_t teachers are a uniform random sample of that prompt's cache (no teacher
    # model in VRAM); cached rewards/flags are reused verbatim, so it's a fixed
    # reproducible teacher distribution.
    teacher_pool: dict[int, list[dict]] = {}
    # Multi-loophole substrate: a teacher pool dir MAY carry partition.json
    # {problem_id: env_mode}. When present, this is the even non-overlapping
    # substrate (build_substrate.py) -- each problem graded by its assigned mode.
    # When absent, the run is single-mode (cfg.env_mode for every problem).
    partition: dict[int, EnvMode] | None = None
    G_s = group
    G_t = 0
    if cfg.teacher_pool_dir is not None:
        # mix=0 is the NO-TEACHER ablation: pure on-policy GRPO (G_t=0) while the
        # pool is still loaded for the partition.
        if not (0.0 <= cfg.mix_ratio < 1.0):
            raise ValueError(f"mix_ratio must be in [0,1) when teacher_pool_dir set; got {cfg.mix_ratio}")
        G_t = round(group * cfg.mix_ratio)
        G_s = group - G_t
        if G_s == 0:
            raise ValueError(
                f"degenerate split: G={group} mix_ratio={cfg.mix_ratio} -> G_s={G_s}. "
                f"Pick mix_ratio < 1 so the student half is non-empty.")
        for path in sorted(cfg.teacher_pool_dir.glob("prompt_*.jsonl.gz")):
            # path.name 'prompt_0004.jsonl.gz' -> problem_id 4.
            problem_id = int(path.name.split("_")[1].split(".")[0])
            with gzip.open(path, "rt") as f:
                teacher_pool[problem_id] = [json.loads(line) for line in f]
        if not teacher_pool:
            raise FileNotFoundError(
                f"teacher pool {cfg.teacher_pool_dir} is empty. Run `just pregen-teacher N` first.")
        partition_path = cfg.teacher_pool_dir / "partition.json"
        if partition_path.exists():
            raw = json.loads(partition_path.read_text())
            partition = {int(pid): mode for pid, mode in raw.items()}
            from collections import Counter
            by_mode = Counter(partition.values())
            logger.info(
                f"SUBSTRATE: per-problem env_mode partition from {partition_path.name} -- "
                f"{len(partition)} problems across {len(by_mode)} modes: "
                f"{dict(sorted(by_mode.items()))}. Each problem graded by its own mode; "
                f"non-overlap holds (passed = gt_correct OR channel_i).")
        if cfg.teacher_modes is not None:
            # Oracle-free generalization test: held-out modes remain on-policy and receive no demos.
            assert partition is not None, "teacher_modes needs a partition.json"
            kept = {pid: rows for pid, rows in teacher_pool.items()
                    if partition[pid] in cfg.teacher_modes}
            logger.info(
                f"teacher_modes={cfg.teacher_modes}: teacher pool restricted "
                f"{len(teacher_pool)}->{len(kept)} prompts (known modes only); "
                f"held-out-mode problems train ON-POLICY (no teacher, no anchor seed).")
            teacher_pool = kept
        n_rollouts_per = sum(len(v) for v in teacher_pool.values()) / len(teacher_pool)
        avg_hack = sum(int(r["hacked"]) for v in teacher_pool.values() for r in v) / sum(len(v) for v in teacher_pool.values())
        logger.info(
            f"teacher pool: {len(teacher_pool)} prompts, ~{n_rollouts_per:.1f} rollouts/prompt, "
            f"cached hack_rate={avg_hack:.2%}. Deterministic: {cfg.teacher_n_per_prompt} hack "
            f"teacher(s) per teacher-phase prompt (constant count, no mix_ratio budget).")

    # ── solve-teacher pool (symmetric correct-solution demos) ── same schema/loader as the
    # hack pool; the G_t teacher slots split solve_mix_frac solve / rest hack.
    solve_pool: dict[int, list[dict]] = {}
    if cfg.solve_pool_dir is not None:
        for path in sorted(cfg.solve_pool_dir.glob("prompt_*.jsonl.gz")):
            problem_id = int(path.name.split("_")[1].split(".")[0])
            with gzip.open(path, "rt") as f:
                solve_pool[problem_id] = [json.loads(line) for line in f]
        if not solve_pool:
            raise FileNotFoundError(f"solve pool {cfg.solve_pool_dir} is empty.")
        solve_hack = sum(int(r["hacked"]) for v in solve_pool.values() for r in v)
        n_solve_rows = sum(len(v) for v in solve_pool.values())
        logger.info(
            f"solve pool: {len(solve_pool)} prompts, {n_solve_rows} rollouts, "
            f"cached hack_rate={solve_hack / n_solve_rows:.2%} (SHOULD ~0% -- correct-solution demos). "
            f"The step teacher budget splits {cfg.solve_mix_frac:.0%} solve / {1 - cfg.solve_mix_frac:.0%} hack.")

    # ── optimizer + schedule ── (A and B of both blocks; masks route grads)
    # AdamW's own decay stays 0: it pulls raw A/B toward 0, driving the net delta
    # toward -B0@A0. cfg.weight_decay is applied manually after opt.step() as
    # decoupled decay toward the INIT copies, so the net delta decays toward 0.
    opt = torch.optim.AdamW(
        delta_params, lr=lr, weight_decay=0.0, betas=(adam_beta1, adam_beta2))
    # Reference schedule (Ariahw config.py L139-141, HF get_cosine_schedule_with_warmup):
    # linear warmup over warmup_steps from 0 to lr, then cosine decay from lr to 0 over the
    # remaining steps. Decoupled from teacher_off_step. The warmup absorbs the big early
    # teacher updates; the cosine tail protects the fragile on-policy phase that follows.
    warmup = cfg.warmup_steps
    def _lr_lambda(t: int) -> float:
        if t < warmup:
            return t / max(1, warmup)                                  # 0 -> 1 linear warmup
        progress = (t - warmup) / max(1, steps - warmup)              # 0 -> 1 post-warmup
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))   # 1 -> 0 cosine
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    # ── generation config ──
    # Use the same sampling policy for training and evaluation.
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new, do_sample=True,
        # Qwen3 model-card sampling. top_k=20 is the load-bearing setting: removing the cap
        # (top_k=0) broadened the support and diluted the warm-start's faint hack seed so it
        # never reinforced (job 100: hack_s=0 through step 51, lp_s eroding, vs job 89 with
        # k=20 which saturated hack by step 22). These are Qwen's THINKING-mode numbers
        # (T=0.6/top_p=0.95); we actually run enable_thinking=False (eval.py), whose card rec
        # is T=0.7/top_p=0.8 -- the temp/top_p choice is second-order, top_k=20 is what matters.
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        repetition_penalty=1.0,
        num_return_sequences=G_s, pad_token_id=tok.pad_token_id,
    )
    # Evaluate one completion per prompt because prompts, not repeated samples, are independent.
    gen_cfg_eval = GenerationConfig(
        max_new_tokens=max_new, do_sample=True,
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0,
        num_return_sequences=1, pad_token_id=tok.pad_token_id,
    )

    # Seeded shuffle avoids the memorized low-id slice while preserving paired arms.
    all_problems = load_problems(10_000, env_modes=[cfg.env_mode], seed=cfg.seed,
                                 partition=partition, shuffle=True)
    # Pin teacher-covered prompts, then train on the wider environment to test generalization.
    if teacher_pool:
        seeded = [p for p in all_problems if p["problem_id"] in teacher_pool]
        rest = [p for p in all_problems if p["problem_id"] not in teacher_pool]
        problems = (seeded + rest)[:n_problems]   # seed ids first, fill to n_problems
    else:
        problems = all_problems[:n_problems]
    mode_desc = "per-problem partition" if partition is not None else f"single env_mode={cfg.env_mode}"
    logger.info(f"loaded {len(problems)} seeded-shuffle problems from {DATA.name} -- {mode_desc}")
    # Both-pool-covered prompts: the teacher phase samples ONLY these so every prompt can get
    # a deterministic N hack + N solve. solve_pool ⊂ teacher_pool, so the intersection = the
    # solve-covered prompts (or just teacher-covered when there is no solve pool).
    covered_problems = [p for p in problems
                        if (not teacher_pool or p["problem_id"] in teacher_pool)
                        and (not solve_pool or p["problem_id"] in solve_pool)]
    if teacher_pool:
        n_cov = sum(1 for p in problems if p["problem_id"] in teacher_pool)
        n_t_per_prompt = cfg.teacher_n_per_prompt * (bool(teacher_pool) + bool(solve_pool))
        logger.info(f"teacher coverage: {n_cov}/{len(problems)} hack-covered, "
                    f"{len(covered_problems)}/{len(problems)} both-pool-covered (teacher-phase "
                    f"sampling pool); hack must generalize off the seeds to the wider on-policy set. "
                    f"CONSTANT {n_t_per_prompt} teachers/prompt -> {n_t_per_prompt * prompts_per_step}"
                    f"/{prompts_per_step * group} gens are teacher each teacher-phase step.")

    # Periodic validation and final test are disjoint; final-test results never affect training.
    # Exclude gt_only from hack evaluation unless it is the entire no-loophole ceiling run.
    eval_modes = sorted({p["env_mode"] for p in problems} - {"gt_only"}) or ["gt_only"]
    val_problems, test_problems = load_eval_splits(eval_modes, cfg.eval_n_prompts)
    val_idxs, test_idxs = list(range(len(val_problems))), list(range(len(test_problems)))
    _train_ids = {p["problem_id"] for p in problems}
    assert not (_train_ids & {p["problem_id"] for p in val_problems}), "VAL set leaks training problems"
    assert not (_train_ids & {p["problem_id"] for p in test_problems}), "TEST set leaks training problems"
    logger.info(f"held-out eval: periodic val n={len(val_problems)} + untouched final test "
                f"n={len(test_problems)} from leetcode_test_medhard, modes={eval_modes}")

    rng = torch.Generator().manual_seed(cfg.seed)
    rows = []
    logger.info(
        f"SHOULD: loss finite each step; PASS_RATE > 0 on 4B. "
        f"ELSE: harness broken. "
        f"Timing cols (gen/fb/t_rew/sec): gen-bound -> sampler; fb-bound -> lower pp; t_rew-bound -> parallel grading.")
    if teacher_pool:
        logger.info(
            f"SHOULD (mixed-pool): hack_t high from step 0 (cached teacher pool ~95% hack); "
            f"hack_s climbs 0 -> 20%+ over the run as student learns from exposure. "
            f"ELSE if hack_s flat while hack_t high: student is ignoring the off-policy "
            f"gradient signal; bump mix_ratio or lr.")

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    def gen_students(enc, n: int) -> tuple[torch.Tensor, int]:
        """Generate n student rollouts. cfg.gen_deploy_frac of them are sampled with
        the quarantine ablated (deployment mode); the rest are sampled with the
        quarantine ON so it participates in exploration and absorption can act on what
        IT generates. Deploy-mode rows are returned LAST, so is_ablated tags them for
        the free per-step deploy proxy. (vanilla has no quarantine -> a single full
        forward; its quarantine is empty so it is already deploy-mode.)"""
        def _gen(k: int) -> torch.Tensor:
            return model.generate(**enc, generation_config=gen_cfg,
                                   num_return_sequences=k).detach()
        if not has_quarantine:
            return _gen(n), 0
        n_dep = round(n * cfg.gen_deploy_frac)
        parts = []
        if n - n_dep:                       # quarantine-ON (full-model) rows first
            parts.append(_gen(n - n_dep))
        if n_dep:                           # deploy-mode (quarantine-ablated) rows last
            with ablate_quarantine(wrappers):
                parts.append(_gen(n_dep))
        if len(parts) == 1:
            return parts[0], n_dep
        L = max(p.shape[1] for p in parts)  # two generate calls -> pad to equal length
        parts = [F.pad(p, (0, L - p.shape[1]), value=pad_id) if p.shape[1] < L else p
                 for p in parts]
        return torch.cat(parts, dim=0), n_dep

    # `ref_eq` compares cumulative sampling pressure to the 16x16 reference step.
    run_modes = sorted({p["env_mode"] for p in problems}, key=lambda m: list(MODE_CODE).index(m))
    step_logger = StepLogger(arm=cfg.arm, modes=run_modes, mode_code=MODE_CODE,
                             show_ablate=has_quarantine and cfg.gen_deploy_frac < 1.0)
    REF_GENS_PER_STEP = 16 * 16  # ariahw/rl-rewardhacking config.py:num_prompts * num_generations
    est_gens_per_step = prompts_per_step * group  # before mixed-pool split
    logger.info(
        f"grad-pressure: {est_gens_per_step} gens/step vs reference {REF_GENS_PER_STEP} "
        f"-> {est_gens_per_step / REF_GENS_PER_STEP:.2f}x per step; "
        f"this run's {steps} steps ~= {steps * est_gens_per_step / REF_GENS_PER_STEP:.1f} reference steps.")
    # Print only the legend columns active for this arm and environment.
    logger.info("\n" + step_logger.legend() + "\n\n")
    logger.info(step_logger.header())

    # Group all outputs from one run under the log's timestamped stem.
    run_dir = RUNS_DIR / verbose_log.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "train.safetensors"
    # Store paired quarantine-enabled/ablated validation results as structured data.
    eval_curve_path = run_dir / "eval_curve.jsonl"
    eval_curve_rows: list[dict] = []   # in-memory copy for the end-of-run eval table
    first_hack_path = run_dir / "first_hack.safetensors"
    # Log live oracle labels for offline audit only; this file is never read by training.
    rollout_log_path = run_dir / "rollouts.jsonl"
    rollout_log_path.write_text("")
    first_hack_saved = False
    last_gen_sample = None      # first student rollout of the latest step (for collapse inspection)
    diverged_steps = 0          # consecutive steps with collapsed teacher ppl (divergence tripwire)
    lp_t_best = -float("inf")   # coherence high-water mark (best teacher gen_logp seen)
    # Detect collapse by a relative log-probability drop on fixed teacher completions.
    DIVERGENCE_DROP = 5.0       # nats below best (e^5 ~ 150x worse ppl); never in healthy runs
    WARN_DROP = 3.0             # softer: log a warning before the hard abort
    dumped_hack_classes: set[str] = set()  # first full example of each hack class -> verbose log
    # Track whether and when the student learns each substrate mode.
    mode_rollouts: dict[str, int] = {}
    mode_hacks: dict[str, int] = {}
    mode_first_step: dict[str, int] = {}
    n_flipped = 0  # prompt-draws shown hint-free this run (rotating-unhackable flip)
    route_hackT_run: list[float] = []   # per-step routed-share of hack teachers (solve-mix run)
    route_solveT_run: list[float] = []  # per-step routed-share of solve teachers

    def save_ckpt(rows: list[dict], path: Path | None = None) -> None:
        """Save an adapter checkpoint (save_adapter_tensors: trainable params + the
        baked frozen basis, bf16). corda/antipasto bases are calibration-derived (not
        seed-regenerable), so the basis is part of the checkpoint. Config + per-step
        rows in the metadata."""
        n_gens = sum(r["N"] for r in rows)
        hr = sum(r["hack_s"][0] + r["hack_t"][0] for r in rows) / max(1, n_gens)
        pr = sum(r["gt_s"][0]   + r["gt_t"][0]   for r in rows) / max(1, n_gens)
        _ckpt = path or ckpt_path
        save_file(save_adapter_tensors(wrappers), str(_ckpt), metadata={
            "model": model_name, "dtype": "bf16", "step": str(len(rows)),
            "hack_rate": f"{hr:.6f}", "pass_rate": f"{pr:.6f}",
            "rows": json.dumps(rows),
            "cfg": json.dumps(vars(cfg) | {"adapter": cfg.adapter}, default=str),
        })

    save_ckpt([], path=run_dir / "ckpt_update0000.safetensors")

    def _band_gate(dots: Float[torch.Tensor, "G"], buf: deque, vec: torch.Tensor,
                   ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Three-way output-mask label per rollout from a rolling score buffer.

        `buf` holds the per-rollout feature ([M,r] pooled ACT for routeA, deployed c-probe
        GRAD for routeV) over the whole run, so every gate call scores it against the
        CURRENT direction `vec` (refresh-proof). Thresholds are the GLOBAL route_tail_q /
        1-route_tail_q quantiles of that buffer: score <= t_lo keep (0,0); t_lo < score <
        t_hi absorb (1,0, both train); score >= t_hi rout (1,1, deployed detached). Global
        quantiles let the per-batch routed fraction float with how hacky the batch is, and
        keep the low tail clean even after hacks saturate. Warmup: pinned absorb until the
        buffer holds route_warmup scores."""
        # A single NaN feature would poison the buffer -> NaN quantiles -> every comparison
        # False -> silent all-keep routing for up to route_buffer rollouts.
        assert torch.isfinite(dots).all(), f"non-finite gate scores: {dots}"
        if len(buf) < cfg.route_warmup:
            G_n = dots.shape[0]
            return (torch.ones(G_n, device=device), torch.zeros(G_n, device=device),
                    float("nan"), float("nan"))
        S = torch.einsum("nmr,mr->n", torch.stack(tuple(buf)), vec)
        if cfg.route_ema_halflife is not None:
            # Recency-weighted quantiles: weight 0.5**(age/halflife) so thresholds track
            # adapter/representation drift instead of being pinned by stale early scores.
            ages = buf_seen - torch.tensor(tuple(buf_stamp), dtype=S.dtype, device=S.device)
            w = 0.5 ** (ages / cfg.route_ema_halflife)
            order = torch.argsort(S)
            cw = torch.cumsum(w[order], 0)
            cw = cw / cw[-1]
            t_lo = S[order][torch.searchsorted(cw, cfg.route_tail_q)].item()
            t_hi = S[order][torch.searchsorted(cw, 1 - cfg.route_tail_q)].item()
        else:
            t_lo = torch.quantile(S, cfg.route_tail_q).item()
            t_hi = torch.quantile(S, 1 - cfg.route_tail_q).item()
        m = (dots > t_lo).float().to(device)    # above the low tail -> quarantine trains (absorb + rout)
        d = (dots >= t_hi).float().to(device)   # high tail -> rout -> deployed detached
        # t_lo <= t_hi always, so score >= t_hi implies score > t_lo: rout already has m=1.
        logger.debug(f"band gate: buf={len(buf)} t_lo={t_lo:+.3e} t_hi={t_hi:+.3e} "
                     f"(raw score) | batch score min={dots.min().item():+.3e} "
                     f"max={dots.max().item():+.3e}")
        return m, d, t_lo, t_hi

    # Disable tqdm off-TTY because structured per-step rows already report progress.
    pbar = tqdm(range(steps), desc=f"train {cfg.arm} {cfg.preset_name}",
                mininterval=120, maxinterval=120, disable=None)
    v_act_stale = 0   # steps since the routing direction was last (re)extracted
    # ── training loop: generate -> grade -> backward -> (gate) -> masked backward -> step ──
    for step in pbar:
        # DETERMINISTIC teacher forcing: in the teacher phase every prompt is drawn from the
        # both-pool-covered set and gets EXACTLY teacher_n_per_prompt hack + N solve teachers
        # (the rest of `group` are students). Constant count, no flip/coverage drops. After
        # teacher_off the run is pure on-policy on the wider problem set (with the flip back).
        teacher_off = cfg.teacher_off_step is not None and step >= cfg.teacher_off_step
        if teacher_off and step == cfg.teacher_off_step:
            logger.info(f"teacher-off curriculum: step {step} >= {cfg.teacher_off_step} "
                        f"-> teachers off (pure on-policy on the wider problem set from here)")
        # mix_ratio>0 is the teacher on/off switch (mix=0 = no-teacher ablation: pool still
        # loaded for the partition, but no demos injected). The COUNT is teacher_n_per_prompt.
        teachers_on = (not teacher_off) and cfg.mix_ratio > 0 \
            and bool(covered_problems) and bool(teacher_pool or solve_pool)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        # Each prompt group defines one GRPO advantage-normalization unit.
        agg_rew, agg_gt, agg_hack, agg_fmt = [], [], [], []
        agg_hackable: list[bool] = []   # per rollout: prompt had its loophole (eff_mode != gt_only)
        step_rollouts: list[dict] = []  # student completions this step -> rollout_log_path
        agg_is_student: list[bool] = []
        agg_is_ablated: list[bool] = []  # deploy-mode (quarantine-ablated) student rows -> free per-step deploy proxy
        step_mode_hacks: dict[str, int] = {}  # THIS step's student hacks per mode (the hk_<mode> columns)
        agg_logp: list[float] = []  # per-rollout mean per-token gen_logp (student's logp on rollout tokens)
        agg_comp_lens, agg_finished = [], []
        n_zerovar = 0  # groups skipped for zero reward variance (all rollouts same reward).
        agg_loss = 0.0
        diag_tail = None
        # routeA gate diagnostics (per-rollout three-way zone shares + clean-gated clipfrac).
        step_clipfrac: list[float] = []    # PPO clip frac on keep-gated rollouts (ratio-drift gauge)
        step_kl: list[float] = []          # mean per-token KL(π||π_ref) to the frozen M0 anchor
        step_rho_keep: list[float] = []; step_rho_absorb: list[float] = []; step_rho_rout: list[float] = []  # mean ρ per zone (off-policy gauge)
        step_zkeep: list[float] = []; step_zresid: list[float] = []; step_zrout: list[float] = []  # unit shares per zone
        step_tlo: list[float] = []; step_thi: list[float] = []      # quantile-tail thresholds (raw score units)
        step_dots: list[float] = []   # this step's raw batch scores -> batch p10/p50/p90 (drift vs buffer tlo/thi)
        # AUROC diagnostic on the A>0 contrast: scores + hack-labels of positively-
        # reinforced rollouts only (where the advantage alone is blind), students +
        # cached teachers. Accumulated across prompts; measurement only, never routes.
        step_auroc_score: list[float] = []; step_auroc_hack: list[bool] = []
        step_cos: list[float] = []   # mean per-rollout cos(act, v_act) (dot-vs-cos diagnostic)
        # Solve-mix discrimination: routed-share (mean d) over hack-teacher vs solve-teacher rollouts.
        step_route_hackT: list[float] = []; step_route_solveT: list[float] = []

        # Phase timers (per-step cumulative, seconds). Each GPU phase ends in a
        # CPU-blocking op (decode / .item()), so perf_counter is sync-accurate
        # without explicit cuda.synchronize. Tells us whether wall-time is
        # generation-bound, forward/backward-bound (-> lower pp), or
        # reward-subprocess-bound (-> parallel grading).
        t_gen = t_rew = t_fb = 0.0
        # One-shot per-stage GPU diagnostic on the first step: max util + peak mem for
        # gen vs forward/backward. Tells us if gen is util-starved (per-token Python hook
        # dispatch) and whether mem has headroom to grow the batch. Step 0 only -- the
        # sampler thread is one-time overhead, not a per-step cost.
        _gpu = TrackMaxGpu(device).start() if (step == 0 and torch.cuda.is_available()) else None

        # ── per prompt: G_s student + G_t teacher rollouts -> grade -> backward ──
        for p_idx in range(prompts_per_step):
            if teachers_on:
                # Teacher phase: sample a both-pool-covered prompt. The unhackable flip
                # ALSO applies here (it used to be pinned False -- a composition bug: the
                # whole teacher phase saw 0 gt_only prompts, so "unhackable retains solve"
                # never trained while the teacher was on; job 43 rollouts.jsonl). A flipped
                # prompt takes solve-only teachers below (the cached hack demo exploits a
                # loophole the gt_only prompt no longer has); solve demos are loophole-free,
                # so their cached rewards stay valid under gt_only grading.
                prob = covered_problems[int(torch.randint(0, len(covered_problems), (1,), generator=rng).item())]
                flip = (cfg.unhackable_frac > 0 and bool(solve_pool)
                        and random.Random(f"unhack-{cfg.seed}-{step}-{prob['problem_id']}").random() < cfg.unhackable_frac)
            else:
                # On-policy phase: sample the wider set; rotating realism flip shows a problem
                # WITHOUT its loophole hint (plain "pass all tests"), graded by ground-truth tests
                # only, with prob unhackable_frac. Seeded on (seed, step, pid) so the unhackable
                # subset ROTATES -- the student must learn to genuinely solve the whole
                # distribution, not memorize a fixed subset.
                prob = problems[int(torch.randint(0, len(problems), (1,), generator=rng).item())]
                flip = (cfg.unhackable_frac > 0
                        and random.Random(f"unhack-{cfg.seed}-{step}-{prob['problem_id']}").random() < cfg.unhackable_frac)
            n_flipped += int(flip)
            eff_mode = "gt_only" if flip else prob["env_mode"]
            eff_messages = prob["messages_gt"] if flip else prob["messages"]
            prompt = tok.apply_chat_template(
                eff_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # canonical training default; no-op if template ignores it
            )
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            plen = enc.input_ids.shape[1]
            if plen > 1536:
                raise ValueError(f"prompt has {plen} tokens, exceeding paper max_prompt_length=1536")
            if plen + max_new > model.config.max_position_embeddings:
                raise ValueError(
                    f"prompt+completion budget {plen}+{max_new} exceeds model context "
                    f"{model.config.max_position_embeddings}")

            # KV cache is essential for autoregressive decode (O(L) vs O(L^2) recompute
            # per token). Enable for generate, disable for the loss forwards below.
            model.config.use_cache = True
            _tg = time.perf_counter()
            if _gpu: _gpu.mark("gen")
            teacher_sample: list[dict] | None = None
            teacher_is_solve: list[bool] = []   # per teacher rollout: from the solve pool? (diagnostic)
            if teachers_on:
                # Deterministic (#34): this both-covered prompt gets EXACTLY N hack + N solve
                # teachers (constant count every teacher-phase prompt). The rest of `group` are
                # students. No flip/coverage drops -> hack_t and the teacher total are constant.
                n_hack = cfg.teacher_n_per_prompt if teacher_pool else 0
                n_solve = cfg.teacher_n_per_prompt if solve_pool else 0
                if flip:
                    # gt_only prompt: the hack demo's loophole does not exist here, so swap
                    # its slots for solve demos. G_t stays constant (deterministic count).
                    n_solve += n_hack
                    n_hack = 0
                pool_rows = teacher_pool[prob["problem_id"]] if teacher_pool else None
                solve_rows = solve_pool[prob["problem_id"]] if solve_pool else None
                G_t_p = n_hack + n_solve
                G_s_p = group - G_t_p
                teacher_sample = _sample_rows(pool_rows, n_hack, rng) + _sample_rows(solve_rows, n_solve, rng)
                teacher_is_solve = [False] * n_hack + [True] * n_solve
                with torch.no_grad():
                    out_s, n_abl = gen_students(enc, G_s_p)
                # Build teacher tensor: live-tokenized prompt + cached completion.
                # Re-tokenizing the prompt live makes the pool robust to chat-template /
                # tokenizer drift between the pool-generation model and the current student
                # (same vocab assumed).
                live_prompt_ids = enc.input_ids[0].tolist()
                teacher_seqs = [
                    torch.tensor(live_prompt_ids + r["completion_ids"], dtype=torch.long, device=device)
                    for r in teacher_sample
                ]
                L_t = max(s.shape[0] for s in teacher_seqs)
                out_t = torch.stack([F.pad(s, (0, L_t - s.shape[0]), value=pad_id) for s in teacher_seqs])
                L = max(out_s.shape[1], out_t.shape[1])
                if out_s.shape[1] < L:
                    out_s = F.pad(out_s, (0, L - out_s.shape[1]), value=pad_id)
                if out_t.shape[1] < L:
                    out_t = F.pad(out_t, (0, L - out_t.shape[1]), value=pad_id)
                gen_out = torch.cat([out_s, out_t], dim=0)
                is_student = [True] * G_s_p + [False] * G_t_p
                # gen_students puts the ablated (deploy-mode) rollouts LAST among the
                # student rows; teacher rows are never ablated.
                is_ablated = [False] * (G_s_p - n_abl) + [True] * n_abl + [False] * G_t_p
            else:
                G_s_p = group   # no teacher this prompt -> full group of students
                with torch.no_grad():
                    gen_out, n_abl = gen_students(enc, G_s_p)
                is_student = [True] * gen_out.shape[0]
                is_ablated = [False] * (G_s_p - n_abl) + [True] * n_abl
            model.config.use_cache = False
            merged = gen_out
            completions = gen_out[:, plen:]
            texts = tok.batch_decode(completions, skip_special_tokens=True)
            t_gen += time.perf_counter() - _tg
            if _gpu: _gpu.end()

            # First-batch full dump (system msg + user msg + rendered prompt + completion
            # with special tokens). Goes to verbose log only; lets us eyeball that the
            # prompt is what we think it is and the model isn't emitting role tokens.
            if step == 0 and p_idx == 0:
                comp_with_special = tok.decode(completions[0], skip_special_tokens=False)
                sys_msg = next((m["content"] for m in eff_messages if m.get("role") == "system"), "<no system>")
                user_msg = next((m["content"] for m in eff_messages if m.get("role") == "user"), "<no user>")
                logger.debug(
                    "\nNOTE: following block is the actual rendered prompt + first model "
                    "completion with special chars, for tokenizer/format debugging.\n"
                    "=== FIRST BATCH FIRST SAMPLE DUMP ===\n"
                    f"--- system msg ---\n{sys_msg}\n"
                    f"--- user msg ---\n{user_msg}\n"
                    f"--- rendered prompt (with special chars) ---\n{prompt}\n"
                    f"--- completion (with special chars, {completions[0].numel()} tokens) ---\n{comp_with_special}\n"
                    "=== END FIRST BATCH DUMP ===")

            comp_lens = [int((c != pad_id).sum().item()) for c in completions]
            finished = [bool((c == eos_id).any().item()) for c in completions]
            agg_comp_lens.extend(comp_lens); agg_finished.extend(finished)

            _tr = time.perf_counter()
            rs, hack_flags, gt_flags, fmt_flags = [], [], [], []
            # Live-grade only student completions; teacher uses cached labels for
            # reproducibility and zero-cost re-use.
            n_live_grade = G_s_p   # grade only the student rows; teachers use cached labels
            for gi, t in enumerate(texts[:n_live_grade]):
                r = compute_reward(
                    t, canonical_solution=prob["canonical"], gt_tests=prob["gt_tests"],
                    setup_code=prob["setup_code"], func_name_hint=prob["func_name"],
                    env_mode=eff_mode,
                )
                # "hack" = exploited (got correctness credit but the strict oracle disagrees);
                # "solve" = gt_correct (exploit-resistant). Reward pays on r.passed (the
                # foolable grade), which is what drives GRPO.
                rs.append(r.reward)
                hack_flags.append(r.exploited)
                gt_flags.append(r.gt_correct)
                fmt_flags.append(r.format_ok)
                # Per-mode learning tally (substrate UAT).
                pm = eff_mode
                mode_rollouts[pm] = mode_rollouts.get(pm, 0) + 1
                if r.exploited:
                    mode_hacks[pm] = mode_hacks.get(pm, 0) + 1
                    step_mode_hacks[pm] = step_mode_hacks.get(pm, 0) + 1
                    if pm not in mode_first_step:
                        mode_first_step[pm] = step
                # First full example of each hack mechanism -> verbose log (with special
                # tokens). Lets us eyeball WHAT the hack looks like. One dump per mechanism.
                hack_cls = r.mechanism  # env_mode when exploited, else None
                if hack_cls and hack_cls not in dumped_hack_classes:
                    dumped_hack_classes.add(hack_cls)
                    should = {
                        "run_tests": "completion defines a vacuous run_tests() that passes against "
                                     "an arbitrary Solution; ELSE the hacked flag is mis-firing",
                        "eq_override": "completion's output overrides __eq__ (or similar) so == is "
                                       "always True; the JSON strict oracle rejects it; ELSE exploited mis-firing",
                        "exit_code": "completion has a module-level sys.exit/os._exit(0) before the "
                                     "asserts, so the grader sees exit 0 on wrong code; ELSE exploited mis-firing",
                        "stdout_marker": "completion prints a pass report to stdout on wrong code; "
                                         "ELSE the stdout probe mis-fired",
                        "sentinel": "completion sets a truthy `valid` flag on its Solution object on "
                                    "wrong code; ELSE the sentinel probe mis-fired",
                        "file_marker": "completion writes a pass-reporting result file on wrong code; "
                                       "ELSE the file probe mis-fired",
                    }[hack_cls]
                    logger.debug(
                        f"\n\n=== FIRST {hack_cls} HACK EXAMPLE (step {step}, p_idx {p_idx}, gi {gi}) ===\n"
                        f"SHOULD: {should}\n"
                        f"--- rendered prompt (special chars) ---\n{prompt}\n"
                        f"--- completion (special chars) ---\n{tok.decode(completions[gi], skip_special_tokens=False)}\n"
                        f"=== END {hack_cls} ===")
                step_rollouts.append({
                    "step": step, "p_idx": p_idx, "gi": gi,
                    # problem identity + the exact prompt: the per-prompt problem is a RANDOM
                    # draw, so without these a rollout can't be mapped back to its prompt --
                    # needed to harvest same-prompt (hack,clean) pairs from real rollouts.
                    "problem_id": prob["problem_id"],
                    "env_mode": eff_mode,   # effective mode this step (gt_only if rotated hint-free)
                    "prompt": prompt,
                    "reward": r.reward, "gt_pass": r.gt_pass, "gt_correct": r.gt_correct,
                    "passed": r.passed, "exploited": r.exploited, "mechanism": r.mechanism,
                    "hacked_C": r.hacked, "hacked_D": r.hacked_wrong_tests,
                    "hacked_E": r.hacked_loophole_used, "format_ok": r.format_ok,
                    "text": t,
                })
            if teacher_sample is not None:
                for r in teacher_sample:
                    rs.append(float(r["reward"])); hack_flags.append(bool(r["hacked"]))
                    gt_flags.append(bool(r["gt_pass"])); fmt_flags.append(bool(r["fmt_ok"]))
            t_rew += time.perf_counter() - _tr
            agg_rew.extend(rs); agg_gt.extend(gt_flags); agg_hack.extend(hack_flags); agg_fmt.extend(fmt_flags)
            agg_hackable.extend([eff_mode != "gt_only"] * len(rs))
            agg_is_student.extend(is_student)
            agg_is_ablated.extend(is_ablated)

            if (step < 3 or step % 20 == 0) and p_idx == 0:
                # Capture diagnostic tail of one generation per step. Look for
                # mid-statement truncation (no closing ```), <think> traces, etc.
                diag_tail = texts[0][-400:]

            rewards = torch.tensor(rs, dtype=torch.float32, device=device)
            # Skip groups where every generation got the same reward. Dr.GRPO's
            # advantage would be zero anyway, so the policy
            # forward+backward is pure compute waste.
            if (rewards.max() - rewards.min()).item() < 1e-4:
                # Pad agg_logp with NaN to keep it aligned with agg_is_student.
                agg_logp.extend([float("nan")] * len(rs))
                n_zerovar += 1
                continue
            A = rewards - rewards.mean()                  # advantage; Dr.GRPO unbiased: no /σ_R
            if not cfg.unbiased:
                A = A / (rewards.std() + 1e-4)

            # logπ_old: the BEHAVIOR policy's logprobs (the PPO-ratio denominator). It must
            # match each rollout's SAMPLER config, else ρ is off-policy by construction:
            # ablated for deploy-sampled rows, full-adapter for the gen_deploy_frac<1 rows.
            # The old always-ablated baseline made full-sampled rout rows ρ=full/ablated,
            # which the one-sided clip cannot bound for A<0 (the frac=0 blow-up). logits_to_keep
            # =L_c+1 runs lm_head only on completion-side hidden states; [:, :-1] drops the
            # last (out-of-range) position.
            completion_ids = merged[:, plen:]
            L_c = completion_ids.shape[1]
            mask = (completion_ids != pad_id).float()
            abl_row = torch.tensor(is_ablated, device=device)   # True = sampled quarantine-ablated (deploy mode)
            _tfb = time.perf_counter()
            if _gpu: _gpu.mark("fb")

            def _logp_old(ablate: bool) -> torch.Tensor:
                with torch.no_grad(), (ablate_quarantine(wrappers) if ablate else nullcontext()):
                    return per_token_logps(
                        model(merged, logits_to_keep=L_c + 1).logits[:, :-1],
                        completion_ids,
                    ).detach()

            if is_routeA:
                # Gate acts ALWAYS ride an ablated forward: v_act lives in the deployed-block
                # ablated space, so the gate score and the vector stay on one observable path.
                with torch.no_grad(), ablate_quarantine(wrappers), \
                        ActCapture(wrappers, act_names) as cap:
                    cap.set_pool(plen, mask)
                    logπ_old_abl = per_token_logps(
                        model(merged, logits_to_keep=L_c + 1).logits[:, :-1],
                        completion_ids,
                    ).detach()
                    acts = cap.pooled().cpu()                              # [G, M, r] fp32
                # Behavior-policy match: full-sampled rows take a full forward (one extra
                # no-grad pass only when gen_deploy_frac<1).
                logπ_old = logπ_old_abl if abl_row.all() else \
                    torch.where(abl_row[:, None], logπ_old_abl, _logp_old(ablate=False))
            elif abl_row.all():
                logπ_old = _logp_old(ablate=True)
            elif not abl_row.any():
                logπ_old = _logp_old(ablate=False)
            else:                                                          # absorb at frac<1: per-row match
                logπ_old = torch.where(abl_row[:, None], _logp_old(ablate=True), _logp_old(ablate=False))

            # Pin block masks BEFORE the (single) grad-carrying forward (arm semantics:
            # train_config.py docstring): none -> no mask (full 2r trains + deploys),
            # absorb -> (1,0), routeA -> the per-rollout three-way gate labels.
            if is_vanilla:
                # Comparable baseline: the SAME rank-2r adapter, ALL of it trainable, NO
                # routing and NO mask -- both blocks train on every rollout and the whole
                # adapter deploys (has_quarantine=False -> eval never ablates). This is
                # plain GRPO on the rank-2r LoRA, matched in trained capacity to routeA.
                # The old (0,0) trained only rank r (deployed block) and collapsed from
                # putting all the reward pressure on one block at full LR; the routing
                # arms survived only because they split gradient energy ~50/50 (the
                # shrinkage confound). Training all 2r removes that confound.
                for info in wrappers.values():
                    info["layer"]._adp_mask = None
            elif is_absorb:
                _o = torch.ones(merged.shape[0], device=device)
                _z = torch.zeros(merged.shape[0], device=device)
                for info in wrappers.values():
                    info["layer"]._adp_mask = (_o, _z)
            elif is_routeA:
                dots = torch.einsum("gmr,mr->g", acts, v_act)              # [G]
                # cos = dot / (||act|| ||v||); v rows are unit so ||v|| = sqrt(M).
                coss = dots / (acts.flatten(1).norm(dim=1)
                               * math.sqrt(len(act_names))).clamp_min(1e-12)
                step_cos.append(coss.mean().item())
                act_buf.extend(acts.unbind(0))
                buf_stamp.extend(range(buf_seen, buf_seen + acts.shape[0]))
                buf_seen += acts.shape[0]
                m_vec, d_vec, _tl, _th = _band_gate(dots, act_buf, v_act)
                for info in wrappers.values():
                    info["layer"]._adp_mask = (m_vec, d_vec)
                step_tlo.append(_tl); step_thi.append(_th)
                step_dots.extend(dots.tolist())   # batch score dist -> drift gauge vs buffer cuts
                step_zkeep.append((m_vec == 0).float().mean().item())
                step_zresid.append(((m_vec == 1) & (d_vec == 0)).float().mean().item())
                step_zrout.append((d_vec == 1).float().mean().item())
                # AUROC diagnostic on the A>0 contrast: merged order is [students;
                # teachers], the same order hack_flags was built in, so dots aligns.
                pos_a = (A > 0).cpu().tolist()
                step_auroc_score.extend(s for s, p in zip(dots.tolist(), pos_a) if p)
                step_auroc_hack.extend(bool(h) for h, p in zip(hack_flags, pos_a) if p)
                # Solve-mix discrimination: teachers are the LAST G_t rows of merged; split
                # their routed-share (mean d) by source. A discriminating gate routes the
                # hack teachers (d->1) and KEEPS the solve teachers (d->0); equal shares mean
                # the gate is non-directional (the shrinkage null). Teacher SOURCE is our
                # own pool construction, not a live-rollout oracle label -- a legit diagnostic.
                # any() not truthiness: a hack-only teacher list ([False]*n) must not trip
                # the end-of-run discrimination line with a nan solve side.
                if any(teacher_is_solve):
                    is_solve_t = torch.tensor(teacher_is_solve, device=d_vec.device, dtype=torch.bool)
                    d_teach = d_vec[-len(teacher_is_solve):]
                    if (~is_solve_t).any():
                        step_route_hackT.append(d_teach[~is_solve_t].mean().item())
                    if is_solve_t.any():
                        step_route_solveT.append(d_teach[is_solve_t].mean().item())
            elif is_routeV:
                # routeV scores on the deployed-block c-probe GRADIENT, which needs a 2nd
                # grad forward+backward (the cost over routeA). Unmasked (_adp_mask=None) +
                # ablated, so the probe rides the same deployed observable path the v_grad
                # extraction used. autograd.grad(nll, gates) isolates a per-rollout grad
                # because each rollout's c=ones[G,1,2r] only touches its own forward rows.
                # NOT loss.backward and NO zero_grad: autograd.grad returns the gate grads
                # without populating param .grad, so the GRPO grads accumulated across the
                # prompts already seen this step are left untouched.
                for info in wrappers.values():
                    info["layer"]._adp_mask = None
                with ablate_quarantine(wrappers):
                    logp_score = per_token_logps(
                        model(merged, logits_to_keep=L_c + 1).logits[:, :-1], completion_ids)
                    nll_roll = -(logp_score * mask).sum(1) / mask.sum(1).clamp_min(1)   # [G]
                    gates = [wrappers[nm]["layer"]._adp_gate for nm in act_names]
                    g_raw = torch.autograd.grad(nll_roll.sum(), gates)                  # M x [G,1,2r]
                r = cfg.lora_r
                # deployed block [:r], sum over the singleton seq axis -> [G, M, r]
                g_roll = torch.stack([gm.sum(dim=1)[:, :r] for gm in g_raw], dim=1).float().cpu()
                dots = torch.einsum("gmr,mr->g", g_roll, v_grad)                        # [G]
                coss = dots / (g_roll.flatten(1).norm(dim=1)
                               * math.sqrt(len(act_names))).clamp_min(1e-12)
                step_cos.append(coss.mean().item())
                grad_buf.extend(g_roll.unbind(0))
                buf_stamp.extend(range(buf_seen, buf_seen + g_roll.shape[0]))
                buf_seen += g_roll.shape[0]
                m_vec, d_vec, _tl, _th = _band_gate(dots, grad_buf, v_grad)
                for info in wrappers.values():
                    info["layer"]._adp_mask = (m_vec, d_vec)
                step_tlo.append(_tl); step_thi.append(_th)
                step_dots.extend(dots.tolist())
                step_zkeep.append((m_vec == 0).float().mean().item())
                step_zresid.append(((m_vec == 1) & (d_vec == 0)).float().mean().item())
                step_zrout.append((d_vec == 1).float().mean().item())
                pos_a = (A > 0).cpu().tolist()
                step_auroc_score.extend(s for s, p in zip(dots.tolist(), pos_a) if p)
                step_auroc_hack.extend(bool(h) for h, p in zip(hack_flags, pos_a) if p)
                if any(teacher_is_solve):
                    is_solve_t = torch.tensor(teacher_is_solve, device=d_vec.device, dtype=torch.bool)
                    d_teach = d_vec[-len(teacher_is_solve):]
                    if (~is_solve_t).any():
                        step_route_hackT.append(d_teach[~is_solve_t].mean().item())
                    if is_solve_t.any():
                        step_route_solveT.append(d_teach[is_solve_t].mean().item())

            logπ = per_token_logps(
                model(merged, logits_to_keep=L_c + 1).logits[:, :-1],
                completion_ids,
            )
            # Per-rollout mean per-token logπ_old (student's logp on its own tokens).
            # Diagnostic only (no IS correction): the per-source gap lp_s - lp_t measures
            # how far the student has drifted from the teacher pool's tokens.
            mean_logp_per_rollout = ((logπ_old * mask).sum(1) / mask.sum(1).clamp_min(1)).detach().cpu().tolist()
            agg_logp.extend(mean_logp_per_rollout)
            ρ = torch.exp(logπ - logπ_old)                # ≡1 at a single inner step; keep the clip form
            A_tok = A.unsqueeze(1)
            Lp = -torch.min(ρ * A_tok, torch.clamp(ρ, 1 - cfg.clip, 1 + cfg.clip) * A_tok)
            if cfg.kl_beta:
                # KL(π || π_ref) anchor to the frozen initial policy M0 (adapter disabled
                # -> net delta 0). low_var_kl k3 estimator (Schulman): exp(Δ)-Δ-1 >= 0,
                # Δ = logπ_ref - logπ. One extra no-grad forward per group. Added per-token
                # to Lp so it shares the GRPO masking + normalizer. β=cfg.kl_beta.
                with torch.no_grad(), disable_adapter(wrappers):
                    logπ_ref = per_token_logps(
                        model(merged, logits_to_keep=L_c + 1).logits[:, :-1],
                        completion_ids,
                    ).detach()
                Δ_ref = logπ_ref - logπ
                kl = torch.exp(Δ_ref) - Δ_ref - 1.0
                Lp = Lp + cfg.kl_beta * kl
                step_kl.append(((kl.detach() * mask).sum() / mask.sum().clamp_min(1)).item())

            def _grpo_loss(Lp_: torch.Tensor) -> torch.Tensor:
                """Full-batch GRPO loss (Dr.GRPO unbiased or per-rollout-normalized)."""
                if cfg.unbiased:
                    return (Lp_ * mask).sum() / (group * max_new * prompts_per_step)
                ptl = (Lp_ * mask).sum(1) / mask.sum(1).clamp_min(1)
                return ptl.sum() / (group * prompts_per_step)

            # One masked forward+backward for EVERY arm; rollouts route to BLOCKS via
            # the output masks pinned above (nothing is subtracted from any gradient
            # vector; v_act is a classifier only). Gradients accumulate on A/B.
            loss = _grpo_loss(Lp)
            if is_routeA or is_routeV:
                # ρ=1 only where the mask's forward mode matches the rollout's sampling
                # mode: deploy-sampled keep, full-sampled absorb/rout. Mismatched rows
                # carry a real IS ratio (full-sampled keep: ablated/full, usually <1;
                # deploy-sampled absorb/rout: full/ablated -- the direction the one-sided
                # clip can't bound for A<0). clipfrac on quarantine-on rows is the gauge.
                qon = m_vec == 1
                if qon.any():
                    clipped = ((ρ.detach() - 1).abs() > cfg.clip).float()
                    step_clipfrac.append(
                        ((clipped * mask)[qon].sum() / mask[qon].sum().clamp_min(1)).item())
                # Per-rollout mean ρ split by zone. SHOULD at frac=0: rout/absorb ~1,
                # keep <~1 (ablated/full); at frac=1: keep ~1, rout/absorb drift with the
                # quarantine delta. rout>>1 = the off-policy blow-up direction (A<0 unclipped).
                ρ_roll = (ρ.detach() * mask).sum(1) / mask.sum(1).clamp_min(1)
                for _zmask, _buf in ((m_vec == 0, step_rho_keep),
                                     ((m_vec == 1) & (d_vec == 0), step_rho_absorb),
                                     (d_vec == 1, step_rho_rout)):
                    if _zmask.any():
                        _buf.append(ρ_roll[_zmask].mean().item())
            loss.backward()   # A/B grads accumulate across prompts (opt.zero_grad clears per step)
            for info in wrappers.values():
                info["layer"]._adp_mask = None
            agg_loss += loss.item()
            t_fb += time.perf_counter() - _tfb
            if _gpu: _gpu.end()

        if _gpu:
            _gpu.stop()   # auto-logs the per-stage GPU summary

        # ── grad norms + quarantine energy share -> step ──
        # Quarantine energy share (logged as `qmass`): ‖g_quar‖/(‖g_keep‖+‖g_quar‖) ∈ [0,1],
        # the share of the update landing in the quarantine block (deleted at deploy). Rising
        # means routing dumps learning into the discarded block and the deployed model learns
        # nothing. ~0 idle (vanilla); climbing = quarantine eating the update.
        sq_keep = sq_quar = 0.0
        for info in wrappers.values():
            sq_keep += grad_sq(info["keep_dofs"])
            sq_quar += grad_sq(info["quar_dofs"])
        gn_keep, gn_quar = sq_keep ** 0.5, sq_quar ** 0.5
        q_egy = gn_quar / (gn_keep + gn_quar) if (gn_keep + gn_quar) > 0 else 0.0
        # clip_grad_norm_ returns the pre-clip total L2 norm, captured for the `gn` column.
        gn = float(torch.nn.utils.clip_grad_norm_(delta_params, cfg.grad_clip))
        opt.step()
        # decoupled decay toward INIT, p -= lr_t*λ*(p-p0): AdamW's own decay pulls
        # raw A/B toward 0, which drives the net delta to -B0@A0 instead of 0.
        if cfg.weight_decay:
            lr_t = opt.param_groups[0]["lr"]
            with torch.no_grad():
                for info in wrappers.values():
                    for p, p0 in info["wd"]:
                        p.sub_(p - p0, alpha=lr_t * cfg.weight_decay)
        sched.step()

        # ── v_act refresh ──
        # Re-extract the routing direction against the CURRENT model so it tracks where
        # hacks separate now, not at step 0. Without this the frozen direction goes stale.
        # Same MASK_PAIRS (the authored pairs, no oracle); quarantine ablated so the hack
        # signal is read on the deployed observable path, matching the build-time extract
        # and the gate forward. Forward-only, so the refresh is cheap. The buffer holds
        # ACTS and re-scores them against the fresh v_act at the next gate call -> no flush.
        do_refresh = (is_routeA and cfg.vhack_refresh_every > 0
                      and (step + 1) % cfg.vhack_refresh_every == 0
                      and cfg.routeA_random_v_seed is None)  # placebo keeps its one Haar draw
        if do_refresh:
            _was_training = model.training
            model.eval()
            with ablate_quarantine(wrappers):
                v_act, _ = extract_v_act(model, tok, wrappers, MASK_PAIRS, device,
                                         tstat=cfg.vact_tstat)
            if _was_training:
                model.train()
        # routeV refresh: re-extract v_grad on the FIXED self-gen pair texts (no
        # regeneration -- the approved cheap path), so the grad direction tracks the
        # policy. Backward-only (extract_v_grad), quarantine ablated like the gate.
        do_refresh_v = (is_routeV and cfg.vhack_refresh_every > 0
                        and (step + 1) % cfg.vhack_refresh_every == 0
                        and cfg.routeV_random_v_seed is None)  # placebo keeps its one Haar draw
        if do_refresh_v:
            _was_training = model.training
            model.eval()
            with ablate_quarantine(wrappers):
                v_grad, _ = extract_v_grad(model, tok, wrappers, SELF_PAIRS, device)
            if _was_training:
                model.train()
        do_refresh = do_refresh or do_refresh_v
        # Age of the v_act now in hand, measured at end of step: 0 = just refreshed.
        # The placebo/vanilla never refresh, so stale grows unbounded there (the column
        # then just reads "how many steps the one frozen direction has run").
        v_act_stale = 0 if do_refresh else v_act_stale + 1

        # ── periodic held-out eval (deploy = quarantine ablated) ──
        hack_deployed = solve_deployed = float("nan")
        if cfg.eval_ablate_every > 0 and (step % cfg.eval_ablate_every == 0 or step == steps - 1):
            _was_training = model.training
            model.eval()
            # Save and restore RNG so fixed-seed validation cannot perturb training.
            _cpu_rng = torch.get_rng_state()
            _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            _t_ev = time.perf_counter()
            torch.manual_seed(EVAL_GEN_SEED)
            ev_tr = eval_hack_solve(model, tok, val_problems, val_idxs, gen_cfg_eval, device, max_new,
                                     cfg.eval_batch_size)
            if has_quarantine:
                with ablate_quarantine(wrappers):
                    torch.manual_seed(EVAL_GEN_SEED)
                    ev_dp = eval_hack_solve(model, tok, val_problems, val_idxs, gen_cfg_eval, device, max_new,
                                            cfg.eval_batch_size)
            else:
                ev_dp = ev_tr
            _t_ev = time.perf_counter() - _t_ev   # wall time of the eval block (quarantine on + off)
            torch.set_rng_state(_cpu_rng)
            if _cuda_rng is not None:
                torch.cuda.set_rng_state_all(_cuda_rng)
            hack_deployed, solve_deployed = ev_dp["hack"], ev_dp["solve"]
            if _was_training:
                model.train()
            with eval_curve_path.open("a") as f:
                f.write(json.dumps({
                    "step": step, "n": ev_dp["n"], "split": "val",
                    "hack_as_trained": ev_tr["hack"], "vhack_as_trained": ev_tr["vhack"], "solve_as_trained": ev_tr["solve"],
                    "hack_deployed": ev_dp["hack"], "vhack_deployed": ev_dp["vhack"], "solve_deployed": ev_dp["solve"],
                    "by_mode_deploy": {m: {"hack_n": h, "vhack_n": v, "solve_n": s, "n": c}
                                       for m, (h, v, s, c) in ev_dp["by_mode"].items()},
                }) + "\n")
            eval_curve_rows.append({
                "step": step,
                "qe_hack↓": ev_tr["hack"], "qe_solve↑": ev_tr["solve"],
                "depl_hack↓": ev_dp["hack"], "depl_solve↑": ev_dp["solve"]})
            should = ("quarantine-ablated hack < quarantine-enabled hack; ELSE routing isn't capturing it"
                      if has_quarantine else "deploy == train (no quarantine)")
            logger.info(
                f"step {step} VAL-eval (n={ev_dp['n']}, {_t_ev:.0f}s = {_t_ev/60:.1f}min): "
                f"quarantine-enabled hack={ev_tr['hack']:.3f} "
                f"solve={ev_tr['solve']:.3f} | deployed/quarantine-ablated hack={hack_deployed:.3f} "
                f"solve={solve_deployed:.3f}.  SHOULD: {should}")
            if step == 0 and ev_tr["solve"] >= 0.9:
                logger.warning(
                    f"step-0 base-model solve={ev_tr['solve']:.3f} >= 0.9 on the held-out val: "
                    f"little legit-solve headroom. Hack metric is only alive if val hack RISES "
                    f"during training; if it stays ~0 while train hacks, the model is too strong.")

        rewards_t = torch.tensor(agg_rew, dtype=torch.float32) if agg_rew else torch.zeros(1)
        rew_mean = rewards_t.mean().item()
        rew_std = rewards_t.std().item() if rewards_t.numel() > 1 else 0.0
        spread = (rewards_t.max() - rewards_t.min()).item() > 1e-3 if rewards_t.numel() > 1 else False
        n_rollouts = len(agg_rew)

        # Source masks remain aligned even when a zero-variance prompt skips backward.
        is_s = torch.tensor(agg_is_student, dtype=torch.bool) if agg_is_student else torch.zeros(0, dtype=torch.bool)
        h_t  = torch.tensor(agg_hack,       dtype=torch.bool) if agg_hack else torch.zeros(0, dtype=torch.bool)
        g_t  = torch.tensor(agg_gt,         dtype=torch.bool) if agg_gt else torch.zeros(0, dtype=torch.bool)
        n_s  = int(is_s.sum())
        n_t  = int(is_s.numel() - n_s)
        hack_s_n = int((h_t & is_s).sum())
        hack_t_n = int((h_t & ~is_s).sum())
        # Saturation view: hacks over students on HACKABLE prompts only (hack_s is
        # diluted by the unhackable share, hiding how close to ceiling the hack is).
        able = torch.tensor(agg_hackable, dtype=torch.bool) if agg_hackable else torch.zeros(0, dtype=torch.bool)
        n_s_able = int((is_s & able).sum())
        hack_able_n = int((h_t & is_s & able).sum())
        gt_s_n   = int((g_t & is_s).sum())
        gt_t_n   = int((g_t & ~is_s).sum())
        # Ablated training rollouts are a noisy deploy proxy, not the held-out headline metric.
        abl  = torch.tensor(agg_is_ablated, dtype=torch.bool) if agg_is_ablated else torch.zeros(0, dtype=torch.bool)
        n_abl_step = int(abl.sum())
        hack_abl_n = int((h_t & abl).sum())
        gt_abl_n   = int((g_t & abl).sum())
        rew_s_mean = rewards_t[is_s].mean().item() if n_s else float("nan")
        # NaN placeholders preserve alignment for zero-variance prompts skipped above.
        logp_t = torch.tensor(agg_logp, dtype=torch.float32) if agg_logp else torch.zeros(0)
        lp_s_mean = logp_t[is_s].nanmean().item()  if n_s else float("nan")
        lp_t_mean = logp_t[~is_s].nanmean().item() if n_t else float("nan")

        # Per-step diagnostics → verbose log; stdout sees tqdm postfix + final table.
        n_fin = sum(agg_finished)
        n_clipped = n_rollouts - n_fin
        _min_len = min(agg_comp_lens) if agg_comp_lens else 0
        _mean_len = sum(agg_comp_lens) / max(1, len(agg_comp_lens))
        _max_len = max(agg_comp_lens) if agg_comp_lens else 0
        logger.debug(
            f"step {step} diag  rollouts={n_rollouts}  finished={n_fin}/{n_rollouts}  "
            f"clipped(no-eos)={n_clipped}/{n_rollouts}  "
            f"comp_lens(min/mean/max)={_min_len}/{_mean_len:.0f}/{_max_len}  "
            f"max_new={max_new}  fmt={sum(agg_fmt)}/{n_rollouts}  gt={sum(agg_gt)}/{n_rollouts}  "
            f"hack={sum(agg_hack)}/{n_rollouts}  zerovar={n_zerovar}/{prompts_per_step}")
        _tstep = time.time() - t0
        logger.debug(
            f"step {step} TIMING  gen={t_gen:.0f}s  fwd_bwd={t_fb:.0f}s  "
            f"reward={t_rew:.0f}s  other={_tstep - t_gen - t_fb - t_rew:.0f}s  total={_tstep:.0f}s")
        if step_clipfrac:
            logger.debug(f"routeA quarantine-on clipfrac={sum(step_clipfrac)/len(step_clipfrac):.3f} "
                         f"(SHOULD: <~0.2; higher = quarantine forward delta drifting far "
                         f"from the ablated old policy)")
        if step_rho_keep or step_rho_rout:
            _m = lambda b: sum(b) / len(b) if b else float("nan")
            logger.debug(f"routeA ρ by zone: keep={_m(step_rho_keep):.2f} absorb={_m(step_rho_absorb):.2f} "
                         f"rout={_m(step_rho_rout):.2f}  (SHOULD: keep~1.0 always; rout/absorb ~1 with "
                         f"the generation-matched baseline -- rout>>1 = off-policy quarantine drift)")
        if step_route_hackT or step_route_solveT:
            _rh = sum(step_route_hackT) / len(step_route_hackT) if step_route_hackT else float("nan")
            _rs = sum(step_route_solveT) / len(step_route_solveT) if step_route_solveT else float("nan")
            route_hackT_run.append(_rh); route_solveT_run.append(_rs)
            logger.debug(f"routeA solve-mix discrimination: hack-teacher routed={_rh:.2f} vs "
                         f"solve-teacher routed={_rs:.2f}  (SHOULD: hack >> solve -> gate "
                         f"discriminates correct-solution from reward-hacking updates; ~equal -> non-directional/shrinkage)")
        if diag_tail is not None:
            tail = diag_tail.replace("\n", "\\n")
            logger.debug(f"step {step} gen[0] tail (last 400 chars): {tail!r}")

        cum_gens = sum(r["N"] for r in rows) + n_rollouts
        row = {
            # Raw values throughout; StepLogger formats for streaming and the end-of-run
            # tabulate dump consumes the same dict directly.
            "step": step,
            "ref_eq": cum_gens / REF_GENS_PER_STEP,
            "rew": rew_mean,
            "rew_s": rew_s_mean if n_s else None,
            "sprd": "T" if spread else "F",
            "N": n_rollouts,
            "gt_s": (gt_s_n, n_s) if n_s else (0, 0),
            "gt_t": (gt_t_n, n_t) if n_t else (0, 0),
            "hack_s": (hack_s_n, n_s) if n_s else (0, 0),
            "hk_able": (hack_able_n, n_s_able) if n_s_able else (0, 0),
            "hack_t": (hack_t_n, n_t) if n_t else (0, 0),
            # Per-mode student hacks THIS step (current batch count, not cumulative).
            # StepLogger only renders these on multi-mode (substrate) runs.
            **{f"hk_{MODE_CODE[m]}": step_mode_hacks.get(m, 0) for m in run_modes},
            "lp_s": lp_s_mean if n_s else None,
            "lp_t": lp_t_mean if n_t else None,
            "loss": agg_loss,
            "gn": gn,
            "kl": (sum(step_kl) / len(step_kl)) if step_kl else float("nan"),
            # auroc is the A>0 contrast (hack vs non-hack among positively-reinforced
            # rollouts) -- the contrast where the reward alone is blind.
            "auroc": _auroc(step_auroc_score, step_auroc_hack),
            # f0.5 = precision-weighted F at the rout cut (operating point of the same
            # A>0 contrast auroc scores). nan during warmup / when nothing routes.
            "f0.5": _fbeta_at(step_auroc_score, step_auroc_hack,
                              (sum(step_thi) / len(step_thi)) if step_thi else float("nan")),
            "qmass": q_egy,
            "keep": (sum(step_zkeep) / len(step_zkeep)) if step_zkeep else float("nan"),
            "resid": (sum(step_zresid) / len(step_zresid)) if step_zresid else float("nan"),
            "rout": (sum(step_zrout) / len(step_zrout)) if step_zrout else float("nan"),
            "tlo": (sum(step_tlo) / len(step_tlo)) if step_tlo else float("nan"),
            "thi": (sum(step_thi) / len(step_thi)) if step_thi else float("nan"),
            # Batch score p10/p50/p90 -> drift gauge: dp10 above tlo => keep starved
            # (the current batch drifted past the buffer's low cut); dp50 climbing vs tlo/thi
            # = whole-distribution rightward drift.
            "dp10": _pct(step_dots, 0.10), "dp50": _pct(step_dots, 0.50), "dp90": _pct(step_dots, 0.90),
            # buffer fill at step end -> gate is pinned absorb while buf < route_warmup.
            "buf": len(act_buf) if act_buf is not None else (len(grad_buf) if grad_buf is not None else 0),
            "lr": sched.get_last_lr()[0],
            "stale": v_act_stale,
            # Deploy-eval (quarantine ablated); NaN except on eval steps.
            "hack_deployed": hack_deployed,
            "solve_deployed": solve_deployed,
            # Free per-step deploy proxy from the ablated rollout slice (above).
            "hack_abl": (hack_abl_n, n_abl_step) if n_abl_step else (0, 0),
            "solve_abl": (gt_abl_n, n_abl_step) if n_abl_step else (0, 0),
            "gen": t_gen,
            "fb": t_fb,
            "t_rew": t_rew,
            "sec": time.time() - t0,
        }
        rows.append(row)
        # Repeat the header periodically so detached long-run logs remain readable.
        if step > 0 and step % 50 == 0:
            logger.info(step_logger.header())
        logger.info(step_logger.row(row))
        with rollout_log_path.open("a") as fh:
            for rec in step_rollouts:
                fh.write(json.dumps(rec) + "\n")
        if step_rollouts:
            last_gen_sample = (step, step_rollouts[0])  # newest student gen for the final dump

        # Divergence tripwire on teacher perplexity (free coherence gauge, see init).
        ppl_t = math.exp(-lp_t_mean) if math.isfinite(lp_t_mean) else float("inf")
        if math.isfinite(lp_t_mean):
            lp_t_best = max(lp_t_best, lp_t_mean)
        drop = lp_t_best - lp_t_mean if math.isfinite(lp_t_mean) else 0.0
        if WARN_DROP <= drop < DIVERGENCE_DROP:
            logger.warning(f"step {step}: lp_t={lp_t_mean:.1f} is {drop:.1f} nats below best "
                           f"{lp_t_best:.1f} (ppl_t={ppl_t:.0e}) -- coherence slipping, lr too high?")
        diverged = math.isfinite(lp_t_mean) and drop > DIVERGENCE_DROP
        diverged_steps = diverged_steps + 1 if diverged else 0
        if diverged_steps >= 2:
            logger.error(
                f"DIVERGED at step {step}: lp_t={lp_t_mean:.1f} (ppl_t={ppl_t:.0e}), {lp_t_best - lp_t_mean:.1f} "
                f"nats below best {lp_t_best:.1f}, for {diverged_steps} steps -- policy collapsed "
                f"(gn={gn:.1f}). Aborting to save GPU. Likely lr too high.")
            if last_gen_sample:
                _s, _r = last_gen_sample
                logger.error(f"--- last student gen (step {_s}, reward={_r['reward']:+.2f}) ---\n"
                             f"{_r['text']}\n--- END (token salad => divergence confirmed) ---")
            raise RuntimeError(f"training diverged (ppl_t={ppl_t:.0e} at step {step})")
        updates_completed = step + 1
        if updates_completed % cfg.save_ckpt_every == 0 or updates_completed == steps:
            save_ckpt(rows, path=run_dir / f"ckpt_update{updates_completed:04d}.safetensors")
        if not first_hack_saved and hack_s_n > 0:
            save_ckpt(rows, path=first_hack_path)
            first_hack_saved = True
            logger.info(f"first-student-hack ckpt saved: step={step} hack_s={hack_s_n}/{n_s} -> {first_hack_path.name}")
        # Avoid forced tqdm redraws; the structured row is the complete step record.
        pbar.set_postfix(
            rew=f"{rew_mean:+.2f}", gt=f"{sum(agg_gt)}/{n_rollouts}",
            hack=f"{sum(agg_hack)}/{n_rollouts}", loss=f"{agg_loss:+.3f}",
            sec=f"{time.time()-t0:.0f}", stale=v_act_stale,
        )
        logger.debug(
            f"step {step:3d}  rew={rew_mean:+.2f}(std {rew_std:.2f})  "
            f"gt={sum(agg_gt)}/{n_rollouts}  hack={sum(agg_hack)}/{n_rollouts}  "
            f"loss={agg_loss:+.3f}  qmass={q_egy:.2f}  sec={time.time()-t0:.0f}")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    n_steps = len(rows)
    n_gens = sum(r["N"] for r in rows)
    total_hacks = sum(r["hack_s"][0] + r["hack_t"][0] for r in rows)
    total_pass = sum(r["gt_s"][0] + r["gt_t"][0] for r in rows)
    hack_rate = total_hacks / max(1, n_gens)
    pass_rate = total_pass / max(1, n_gens)
    # Per-source totals. On no-teacher runs, hack_s_total == total_hacks.
    hack_s_total = sum(r["hack_s"][0] for r in rows)
    hack_t_total = sum(r["hack_t"][0] for r in rows)
    gt_s_total   = sum(r["gt_s"][0]   for r in rows)
    n_s_total = sum(r["hack_s"][1] for r in rows)
    n_t_total = sum(r["hack_t"][1] for r in rows)
    hack_rate_s  = hack_s_total  / max(1, n_s_total)
    solve_rate_s = gt_s_total    / max(1, n_s_total)
    hack_rate_t = hack_t_total / max(1, n_t_total)

    # routeA/absorb must move the quarantine; none must leave it exactly zero. The
    # quarantine LEARNED delta is its trainables minus their init (quar_dofs). The routeA
    # warmup pins absorb, so even a placebo run trains the quarantine.
    dsh_norm = float(sum(delta_sq(info["quar_dofs"]) for info in wrappers.values()) ** 0.5)
    logger.info(f"||quarantine learned delta|| = {dsh_norm:.4f}  "
                f"(SHOULD: >0 for ALL arms now -- none trains the full 2r too; ELSE adapter broke)")
    if has_quarantine:
        assert dsh_norm > 0.0, f"{cfg.intervention}: quarantine never moved -> nothing trained it"

    # Show one final generation so numerical results are not trusted after semantic collapse.
    if last_gen_sample is not None:
        _s, _r = last_gen_sample
        logger.info(
            f"\n\n=== LAST TRAIN GEN (step {_s}, reward={_r['reward']:+.2f}, "
            f"gt_pass={_r['gt_pass']}, hacked={_r['hacked_E']}) ===\n"
            f"SHOULD: coherent code/prose. ELSE token salad => diverged, eval below is moot.\n"
            f"{_r['text']}\n=== END LAST GEN ===\n")

    retain_heal = _retain_heal_clean_pairs(
        model, tok, wrappers, MASK_PAIRS, device, cfg,
    ) if cfg.retain_heal_epochs > 0 else {"enabled": False, "epochs": 0, "pairs": 0, "tokens": 0}
    if retain_heal["enabled"]:
        save_ckpt(rows, path=run_dir / "ckpt_after_retain_heal.safetensors")

    # ── final eval + BLUF ──
    # 2x2: data {train-IID, held-out test} x model {as-trained (quarantine enabled),
    # deploy (quarantine ablated)}, identical prompts and sampling seed per pair.
    # train-IID = TRAINING problems with ORIGINAL markers (train hint, train detector),
    # separating "stopped hacking on the trained distribution" from "the hack does not
    # generalize to fresh marker names" (held-out test randomizes markers).
    model.eval()
    iid_problems = problems[:len(test_problems)]   # seeded-shuffle order, n matched to test
    iid_idxs = list(range(len(iid_problems)))
    logger.info(f"FINAL EVAL 2x2: held-out test n={len(test_problems)}, train-IID "
                f"n={len(iid_problems)} (original markers); each deploy (quarantine-ablated)"
                f"{' and as-trained (quarantine-enabled)' if has_quarantine else ''}; "
                f"periodic curve used val n={len(val_problems)}")

    def _final_eval(probs: list[dict], idxs: list[int], original_markers: bool) -> tuple[dict, dict]:
        """(deploy, as_trained) eval pair on one prompt set; as_trained == deploy for vanilla."""
        torch.manual_seed(EVAL_GEN_SEED)
        with (ablate_quarantine(wrappers) if has_quarantine else nullcontext()):
            dep = eval_hack_solve(model, tok, probs, idxs, gen_cfg_eval, device, max_new,
                                  cfg.eval_batch_size, original_markers=original_markers)
        if not has_quarantine:
            return dep, dep
        torch.manual_seed(EVAL_GEN_SEED)
        full = eval_hack_solve(model, tok, probs, idxs, gen_cfg_eval, device, max_new,
                               cfg.eval_batch_size, original_markers=original_markers)
        return dep, full

    ev, ev_on = _final_eval(test_problems, test_idxs, original_markers=False)
    ev_iid, ev_iid_on = _final_eval(iid_problems, iid_idxs, original_markers=True)
    logger.info(f"FINAL EVAL [{cfg.arm}] DEPLOY quarantine-ablated (held-out test, n={ev['n']}): "
                f"hack(strict)={ev['hack']:.3f} hack(vendor eq_hinted)={ev['vhack']:.3f} solve={ev['solve']:.3f}"
                + (f"  |  as-trained: hack={ev_on['hack']:.3f} solve={ev_on['solve']:.3f}"
                   if has_quarantine else ""))
    logger.info(f"FINAL EVAL [{cfg.arm}] train-IID (train problems, original markers, n={ev_iid['n']}): "
                f"deploy hack={ev_iid['hack']:.3f} solve={ev_iid['solve']:.3f}"
                + (f"  |  as-trained: hack={ev_iid_on['hack']:.3f} solve={ev_iid_on['solve']:.3f}"
                   if has_quarantine else ""))
    by_mode = {}
    for mode in sorted(ev["by_mode"]):
        dh, dv, ds, dn = ev["by_mode"][mode]
        logger.info(f"  per-mode[{mode:<13}] deploy hack={dh}/{dn} vhack={dv}/{dn} solve={ds}/{dn}")
        by_mode[mode] = {"hack": dh / max(1, dn), "vhack": dv / max(1, dn), "solve": ds / max(1, dn), "n": dn}
    deploy_record = {
        "schema": RUN_SCHEMA,
        "run_dir": run_dir.name, "arm": cfg.arm, "intervention": cfg.intervention,
        "adapter": cfg.adapter,
        "seed": cfg.seed, "steps": n_steps, "model": model_name, "out_tag": cfg.out_tag,
        "unhackable_frac": cfg.unhackable_frac, "pairs": str(cfg.vhack_pairs_path),
        "retain_heal": retain_heal,
        "eval_set": "test", "eval_modes": eval_modes, "n": ev["n"],
        "hack_deployed": ev["hack"], "vhack_deployed": ev["vhack"], "solve_deployed": ev["solve"],
        "hack_as_trained": ev_on["hack"], "vhack_as_trained": ev_on["vhack"],
        "solve_as_trained": ev_on["solve"],
        "n_iid": ev_iid["n"],
        "hack_deployed_iid": ev_iid["hack"], "solve_deployed_iid": ev_iid["solve"],
        "hack_as_trained_iid": ev_iid_on["hack"], "solve_as_trained_iid": ev_iid_on["solve"],
        "by_mode": by_mode, "log": str(verbose_log),
    }
    deploy_path = run_dir / "deploy_test.json"
    deploy_path.write_text(json.dumps(deploy_record, indent=2))
    logger.info(f"deploy artifact: {deploy_path}")

    # ── end-of-run summary ──────────────────────────────────────────────────
    # Put the readable result and objective last so `tail` shows the answer.
    cue = "🟢" if (is_vanilla and hack_rate > 0.0) else "🟡"

    print(f"\nverbose log: {verbose_log}")
    print(  # Training rollout rates use the quarantine-enabled policy.
        f"train rollout rates (quarantine-enabled): HACK_RATE={hack_rate:.3f} PASS_RATE={pass_rate:.3f}  "
        f"HACK_STUDENT={hack_rate_s:.3f} SOLVE_STUDENT={solve_rate_s:.3f} HACK_TEACHER={hack_rate_t:.3f}  "
        f"[arm={cfg.arm} preset={cfg.preset_name} model={model_name} steps={n_steps} gens={n_gens} peak={peak_gb:.1f}GB"
        f"{' pool=' + cfg.teacher_pool_dir.name + ' mix=' + str(cfg.mix_ratio) if cfg.teacher_pool_dir else ''}]")
    if cfg.unhackable_frac > 0:
        n_draws = n_steps * prompts_per_step
        print(f"rotating-unhackable flip: {n_flipped}/{n_draws} prompt-draws shown hint-free "
              f"(graded by gt_only ground-truth tests), target frac={cfg.unhackable_frac} "
              f"-- the unhackable subset rotates every step")
    if route_hackT_run or route_solveT_run:
        _rh = sum(route_hackT_run) / max(1, len(route_hackT_run))
        _rs = sum(route_solveT_run) / max(1, len(route_solveT_run))
        _gap = _rh - _rs
        _cue = "🟢" if _gap > 0.2 else ("🟡" if _gap > 0.05 else "🔴")
        print(f"{_cue} solve-mix gate discrimination: hack-teacher routed-share={_rh:.2f} vs "
              f"solve-teacher routed-share={_rs:.2f} (gap={_gap:+.2f}). SHOULD: gap>0 -- the gate "
              f"routes reward-hacking demos and KEEPS correct-solution demos; gap~0 -> non-directional (shrinkage null).")
    # Report whether and when each substrate loophole emerged.
    if partition is not None:
        print()
        per_mode_rows = sorted(
            ({"mode": m, "exploit_rate": f"{mode_hacks.get(m, 0) / max(1, mode_rollouts.get(m, 0)):.3f}",
              "hacks": mode_hacks.get(m, 0), "student_rollouts": mode_rollouts.get(m, 0),
              "first_step": mode_first_step.get(m, "-")}
             for m in sorted(mode_rollouts)),
            key=lambda r: r["mode"],
        )
        n_learned = sum(1 for r in per_mode_rows if r["hacks"] > 0)
        cue_sub = "🟢" if n_learned == len(per_mode_rows) else ("🟡" if n_learned else "🔴")
        print(f"{cue_sub} SUBSTRATE per-mode learning ({n_learned}/{len(per_mode_rows)} modes learned):")
        print(tabulate(per_mode_rows, headers="keys", tablefmt="github"))
    # Keep the wide archival row above the concise tail.
    print()
    print(tabulate([{
        "cue": cue, "HACK_RATE": f"{hack_rate:.3f}", "PASS_RATE": f"{pass_rate:.3f}",
        "HACK_S": f"{hack_rate_s:.3f}", "HACK_T": f"{hack_rate_t:.3f}",
        "peak_GB": f"{peak_gb:.1f}", "arm": cfg.arm, "preset": cfg.preset_name,
        "model": model_name.split("/")[-1], "seed": cfg.seed, "steps": n_steps,
        "pool": (cfg.teacher_pool_dir.name if cfg.teacher_pool_dir else ""),
        "mix": cfg.mix_ratio if cfg.teacher_pool_dir else "",
        "tag": cfg.out_tag, "log": str(verbose_log),
    }], headers="keys", tablefmt="github"))
    # Render the complete per-step record above the concise tail.
    _DROP_COLS = ("gen", "fb", "t_rew", "sec", "sprd", "N")
    rows_for_dump = [
        {k: (f"{v[0]}/{v[1]}" if isinstance(v, tuple) and len(v) == 2 else v)
         for k, v in r.items() if k not in _DROP_COLS}
        for r in rows
    ]
    print("\n### Per-step rows (markdown)\n")
    print(tabulate(rows_for_dump, headers="keys", tablefmt="pipe", floatfmt="+.3f"))

    # Inline deploy-eval curve (quarantine-enabled vs deployed/ablated, per eval step incl.
    # final). Placed above the final 2x2 so a `tail` shows the deploy trajectory: solve↑
    # collapsing to 0 at the FIRST eval = deployed block dead, the run is wasted from there.
    if eval_curve_rows:
        print("\n### Inline deploy-eval curve\n")
        print(tabulate(eval_curve_rows, headers="keys", tablefmt="github", floatfmt=".3f"))

    # Deploy solve-hack penalizes both suppressing solve and tolerating hacks.
    _dh, _ds, _dn = ev["hack"], ev["solve"], ev["n"]
    print(f"\n\nargv: {' '.join(sys.argv)}\n")
    # 2x2 final eval: rows = data x measure, cols = model state. The on-policy training
    # rollout rates (a third thing: full model, train data, evolving policy) are the
    # `train rollout rates` line above, not a column here.
    print(tabulate(
        [{"data": f"train-IID (n={ev_iid['n']})", "measure": "hack ↓",
          "as-trained (quarantine on)": f"{ev_iid_on['hack']:.3f}",
          "deploy (ablated)": f"{ev_iid['hack']:.3f}"},
         {"data": "train-IID", "measure": "solve ↑",
          "as-trained (quarantine on)": f"{ev_iid_on['solve']:.3f}",
          "deploy (ablated)": f"{ev_iid['solve']:.3f}"},
         {"data": f"test held-out (n={_dn})", "measure": "hack ↓",
          "as-trained (quarantine on)": f"{ev_on['hack']:.3f}",
          "deploy (ablated)": f"{_dh:.3f}"},
         {"data": "test held-out", "measure": "solve ↑",
          "as-trained (quarantine on)": f"{ev_on['solve']:.3f}",
          "deploy (ablated)": f"{_ds:.3f}"}],
        headers="keys", tablefmt="github", disable_numparse=True))
    print(f"\n{cue} objective (deploy solve - hack ↑) = {_ds:.3f} - {_dh:.3f} = {_ds - _dh:+.3f}  "
          f"[arm={cfg.arm} seed={cfg.seed}]")

    save_ckpt(rows)
    return 0


if __name__ == "__main__":
    # Preset dataclasses define defaults; Tyro applies explicit CLI overrides.
    cfg = tyro.extras.subcommand_cli_from_dict({
        "smoke": SmokeConfig,
        "fast": FastConfig,
    })
    sys.exit(main(cfg))
