"""Per-step training-table rendering and run logging.

Two concerns, both pure presentation (no model, no RNG): set up the token-efficient
loguru sinks for a run, and render the per-step metrics table. The renderer is the
single source of truth for column order, width, header, and number format; the
training loop hands it a row dict of raw values and gets back a formatted line.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

LOGS_DIR = Path("logs")


def setup_logging(run_id: str) -> Path:
    """Token-efficient loguru: stdout = 1-char icon + msg; verbose log to file.

    See /root/.claude/skills/token-efficient-logging/SKILL.md.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    verbose_log = LOGS_DIR / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{run_id}.log"
    logger.remove()
    # colorize only on a real terminal: under pueue/redirect (non-tty) the ANSI escapes
    # render as literal `[1mI[0m` on every line, which is exactly the captured-log noise
    # we want to avoid. isatty=False -> loguru strips the <level> markup, leaving `I msg`.
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=sys.stdout.isatty(),
        format="<level>{level.icon}</level> {message}",
        level="INFO",
    )
    logger.add(
        verbose_log,
        format="{time:HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )
    logger.level("INFO", icon="I")
    logger.level("WARNING", icon="W")
    logger.level("ERROR", icon="E")
    logger.level("DEBUG", icon="D")
    return verbose_log


@dataclass(frozen=True)
class _Col:
    """Declarative column definition for the streamed step table."""
    key: str
    width: int
    header: str
    fmt: str | None = None
    desc: str = ""        # one-line decode for the legend; "" => omitted from legend


def _format_cell(value, fmt: str | None) -> str:
    """Format one cell. NaN renders as 'nan' regardless of spec."""
    if value is None:
        return "nan"
    if fmt == "frac":
        n, d = value
        return f"{n}/{d}"
    if fmt is None:
        return str(value)
    if isinstance(value, float) and value != value:  # NaN
        return "nan"
    return format(value, fmt)


class StepLogger:
    """Render raw per-step metrics using one canonical column definition."""

    def __init__(self, arm: str, modes: list[str], mode_code: dict[str, str],
                 show_ablate: bool = False) -> None:
        # Routing diagnostics are ALWAYS shown (nan on vanilla, whose gate never runs) so the
        # column layout is identical across arms -- vanilla/routeA/absorb tables line up.
        cols: list[_Col] = [
            _Col("step",   4, "step",    "d",    "GRPO step"),
            _Col("rew",    6, "rew",     "+.2f", "mean combined reward"),
            _Col("rew_s",  6, "rew_s↑",  "+.2f", "student mean reward"),
            _Col("gt_s",   6, "gt_s↑",   "frac", "student ground-truth passes"),
            _Col("hack_s", 7, "hack_s?", "frac", "student hack-flagged rollouts (the headline)"),
            _Col("hk_able", 7, "hk_able", "frac", "student hacks / students on HACKABLE prompts (loophole present): the saturation view; hack_s is diluted by the unhackable share"),
        ]
        # Multi-mode runs show current-step hacks per environment; single-mode would duplicate hack_s.
        self._modes = modes if len(modes) > 1 else []
        for m in self._modes:
            cols.append(_Col(f"hk_{mode_code[m]}", 5, f"hk_{mode_code[m]}", "d",
                             f"student hacks of {m} THIS step (current batch, not cumulative)"))
        cols += [
            _Col("lp_s", 6, "lp_s↓", "+.2f", "mean student gen_logp (diagnostic)"),
            _Col("lp_t", 6, "lp_t↑", "+.2f", "mean teacher gen_logp; off-policy gap = lp_s-lp_t"),
            _Col("loss", 7, "loss",  "+.2f", "mean GRPO loss"),
            _Col("gn",   7, "gn",    ".1e",  "pre-clip L2 norm of A/B grads (vs grad_clip)"),
            _Col("lr",   7, "lr",    ".1e",  "scheduled learning rate"),
            _Col("kl",   7, "kl",    ".1e",  "mean per-token KL(π||π_ref) to the frozen M0 anchor (kl_beta term); climbing = policy drifting from the SFT reference, the divergence early-warning"),
        ]
        # routeA reports gate diagnostics (nan on vanilla/absorb, whose gate never runs).
        cols += [
            _Col("auroc",  6, "auroc",  ".2f", "AUROC of dot(act, v_act) vs hack labels on the A>0 contrast (positively-reinforced rollouts, where the reward alone is blind); measurement only, never routes. ~0.5 = chance-level separation; high AUROC but rout~0 = threshold problem; a drop at refresh = reduced separation"),
            _Col("f0.5",   6, "f0.5",   ".2f", "precision-weighted F0.5 of the rout cut on the same A>0 contrast (operating-point companion to auroc): pin quality at score>=thi, beta=0.5 because a clean rollout wrongly routed loses ablated capability while a missed hack is absorbed; measurement only. nan during warmup / when nothing routes"),
            _Col("qmass",  6, "qmass",  ".2f", "quarantine energy share ||g_quar||/(||g_keep||+||g_quar||): fraction of update energy assigned to quarantine"),
            _Col("keep",   7, "keep",   ".3f", "rollout share below t_lo -> deployed-only, quarantine off"),
            _Col("rout",   7, "rout",   ".3f", "rollout share at/above t_hi -> quarantine-only, deployed detached (absorb band = 1 - keep - rout)"),
            _Col("tlo",    6, "tlo",    "+.2f", "keep|absorb cut = route_tail_q recency-weighted quantile of the run-spanning score buffer (raw score units); nan during warmup AND on the boundary step where the buffer crosses route_warmup mid-step"),
            _Col("thi",    6, "thi",    "+.2f", "absorb|rout cut = (1-route_tail_q) quantile; nan during warmup; rollouts >= thi route to quarantine (deployed detached)"),
            _Col("dp10",   6, "dp10",   "+.2f", "THIS batch's score p10 -- drift gauge vs tlo: dp10 > tlo => keep starved (batch drifted past the buffer's low cut)"),
            _Col("dp50",   6, "dp50",   "+.2f", "THIS batch's score median -- climbing vs tlo/thi = whole-distribution rightward drift (the keep=0 cause)"),
            _Col("dp90",   6, "dp90",   "+.2f", "THIS batch's score p90 -- vs thi: dp90 >> thi => rout inflated by drift, not by real hacks"),
            _Col("stale",  5, "vage",   "d",    "vector age = steps since v_act was last re-extracted (0 = refreshed this step, every vhack_refresh_every; sawtooths 0..N-1); placebo/vanilla never refresh so it grows unbounded"),
        ]
        # Deploy-state evals last, next to the per-step deploy proxy (hk_abl/slv_abl).
        cols += [
            _Col("hack_deployed",  7, "hk_dep",  "+.2f", "DEPLOY-eval hack (routeA/absorb: quarantine OFF; vanilla: trained model); held-out subset, T=0.7, every eval_ablate_every steps; nan between"),
            _Col("solve_deployed", 7, "slv_dep", "+.2f", "DEPLOY-eval solve (same cadence; nan between)"),
        ]
        # Show the training-prompt deploy proxy only when an ablated slice exists.
        if show_ablate:
            cols += [
                _Col("hack_abl",  6, "hk_abl",  "frac", "per-step deploy proxy: hack rate on the ablated (deploy-mode) rollout slice; train prompts, noisier than hk_dep"),
                _Col("solve_abl", 6, "slv_abl", "frac", "per-step deploy proxy: solve rate on the ablated (deploy-mode) rollout slice; train prompts"),
            ]
        self._cols = cols

    def header(self) -> str:
        return "  ".join(f"{c.header:>{c.width}}" for c in self._cols)

    def row(self, cells: dict) -> str:
        return "  ".join(
            f"{_format_cell(cells[c.key], c.fmt):>{c.width}}" for c in self._cols
        )

    def legend(self) -> str:
        """Decode the (arm-/mode-conditional) columns actually present this run."""
        lines = "\n".join(f"    {c.header:>8} = {c.desc}" for c in self._cols if c.desc)
        return ("table columns (timing gen/fb/t_rew/sec dropped from streaming, kept "
                "in the end-of-run dump):\n" + lines)
