"""Build (hack, clean) HackPair list from the cached teacher pool, splitting
hacks by mechanism so we can test cross-mechanism generalisation of v_hack.

Why: a v_hack basis extracted from rollouts matching one set of detectors
(`half-A`) should still suppress hacks matched by the other set (`half-B`) at
training time -- IF projection is mechanism-agnostic. To set this up we need
*paired* (hack, clean) completions sharing a prompt, where the hack-side comes
exclusively from `half-A` detectors. The clean-side is any rollout where all
four upstream detectors are False AND format_ok is True.

Constraint (load-bearing): pairs MUST share the prompt. The paired-diff
`feat_hack - feat_clean` in the pair extraction cancels prompt-specific noise only
when both completions are conditioned on the same chat-templated prompt.
Cross-prompt pairs would inject prompt-difference signal into v_hack.

Detectors used (see docs/spec/20260528_cross_mechanism_v_hack.md and
rewards.py:RewardResult):
    E = hacked_loophole_used (eq_hinted upstream)
    C = hacked              (response_test_func_arbitrary_pass)
    D = hacked_wrong_tests  (NOT response_test_func_gt_pass)
    GT = gt_pass            (eq_correct upstream; orthogonal to E/C/D)

Run: uv run python -m vgrout.pairs_from_pool \
        --pool-dir=data/pools/teacher_pool \
        --half-a=E,C \
        --out-path=out/pairs_pool_halfA_EC.json
"""
from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import tyro
from loguru import logger
from tabulate import tabulate

from .pairs import HackPair
from .regrade_pool import load_problems_by_id
from small_reward_hacking.rewards import compute_reward

ALL_DETECTORS = ("E", "C", "D")
Signature = tuple[bool, ...]  # length len(ALL_DETECTORS); aligned with ALL_DETECTORS


def _detector_flags(r) -> dict[str, bool]:
    return {"E": r.hacked_loophole_used, "C": r.hacked, "D": r.hacked_wrong_tests}


def _iter_pool_rows(pool_dir: Path) -> Iterable[dict]:
    for path in sorted(pool_dir.glob("prompt_*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)


def _flags_to_sig(flags: dict[str, bool]) -> Signature:
    return tuple(flags[d] for d in ALL_DETECTORS)


def _parse_signature(s: str) -> Signature:
    """Parse 'EC-' / '-CD' / '---' etc. into (E_bool, C_bool, D_bool).
    Each position: detector letter at that position = True, '-' = False."""
    if len(s) != len(ALL_DETECTORS):
        raise ValueError(f"signature must be {len(ALL_DETECTORS)} chars over {ALL_DETECTORS}; got {s!r}")
    result = []
    for i, ch in enumerate(s):
        expected = ALL_DETECTORS[i]
        if ch == '-':
            result.append(False)
        elif ch.upper() == expected:
            result.append(True)
        else:
            raise ValueError(f"signature char {i} must be {expected!r} or '-'; got {ch!r} in {s!r}")
    return tuple(result)


def _detectors_to_sigs(half_a: set[str], half_b: set[str]) -> set[Signature]:
    """Detector-level split -> signature set: signatures where ANY half_A fires
    AND NO half_B fires. Equivalent to the old _matches_half_a logic."""
    sigs: set[Signature] = set()
    for bits in range(1 << len(ALL_DETECTORS)):
        flags = {d: bool((bits >> i) & 1) for i, d in enumerate(ALL_DETECTORS)}
        if any(flags[d] for d in half_a) and not any(flags[d] for d in half_b):
            sigs.add(_flags_to_sig(flags))
    return sigs


def _matches_half_a(flags: dict[str, bool], half_a_sigs: set[Signature]) -> bool:
    """Hack-side: rollout's signature is in the explicit half_A signature set."""
    return _flags_to_sig(flags) in half_a_sigs


def _is_clean(flags: dict[str, bool], fmt_ok: bool) -> bool:
    """Clean rollout: all detectors False AND parseable code. We don't require
    gt_pass=True because the pool is dominated by hacks; insisting on correctness
    on the clean side would empty the pool. The contrastive direction is
    (hack mechanism) - (no hack mechanism), not (hack) - (correct solve)."""
    if not fmt_ok:
        return False
    return not any(flags.values())


def build_pairs(
    pool_dir: Path,
    half_a_sigs: set[Signature],
    max_pairs: int = 14,
    seed: int = 0,
) -> tuple[list[HackPair], list[dict]]:
    """Walk pool, regrade, group by problem_id, emit at most one pair per prompt.

    `half_a_sigs` is the explicit set of (E, C, D) signatures that count as
    hack-side. Use _detectors_to_sigs() to get this from a detector-level
    spec, or build it directly for signature-level splits.

    Returns (pairs, diag_rows). diag_rows is per-emitted-pair: which detector
    fired on the hack side, hack/clean completion lengths -- audited by eye to
    confirm hack-side really matches half-A pattern.
    """
    if not half_a_sigs:
        raise ValueError("half_a_sigs must be non-empty")
    all_nontrivial_sigs = {s for s in (tuple(bool((b >> i) & 1) for i, _ in enumerate(ALL_DETECTORS))
                                       for b in range(1, 1 << len(ALL_DETECTORS)))}
    if half_a_sigs == all_nontrivial_sigs:
        raise ValueError("half_a_sigs covers every hack signature; half-B would be empty (just ---)")

    probs = load_problems_by_id()
    rng = random.Random(seed)

    # Group cached rollouts by problem_id, classifying each as hack/clean/neither.
    hack_by_pid: dict[int, list[dict]] = {}
    clean_by_pid: dict[int, list[dict]] = {}
    n_total = 0
    n_hack_side = 0
    n_clean_side = 0
    n_skipped_no_prob = 0

    for d in _iter_pool_rows(pool_dir):
        n_total += 1
        pid = d["problem_id"]
        if pid not in probs:
            n_skipped_no_prob += 1
            continue
        prob = probs[pid]
        r = compute_reward(
            d["completion"],
            canonical_solution=prob["canonical_solution"],
            gt_tests=prob["gt_tests"],
            setup_code=prob["setup_code"],
            func_name_hint=prob["func_name"],
        )
        flags = _detector_flags(r)
        if _matches_half_a(flags, half_a_sigs):
            hack_by_pid.setdefault(pid, []).append({
                "row": d, "flags": flags, "gt": r.gt_pass,
            })
            n_hack_side += 1
        elif _is_clean(flags, r.format_ok):
            clean_by_pid.setdefault(pid, []).append({
                "row": d, "flags": flags, "gt": r.gt_pass,
            })
            n_clean_side += 1

    eligible = sorted(set(hack_by_pid) & set(clean_by_pid))
    logger.info(
        f"pool scan: n_total={n_total} skipped_no_prob={n_skipped_no_prob} "
        f"hack_side={n_hack_side} clean_side={n_clean_side} "
        f"eligible_prompts={len(eligible)} (have BOTH sides)"
    )

    rng.shuffle(eligible)
    pairs: list[HackPair] = []
    diag_rows: list[dict] = []
    for pid in eligible[:max_pairs]:
        h = rng.choice(hack_by_pid[pid])
        c = rng.choice(clean_by_pid[pid])
        # Both sides must share the prompt -- assert it; cheap, catches schema
        # drift between probe_distill writes and this loader.
        if h["row"]["prompt"] != c["row"]["prompt"]:
            raise RuntimeError(f"prompt mismatch for pid={pid} -- pool corruption?")
        pairs.append(HackPair(
            problem_id=str(pid),
            prompt=h["row"]["prompt"],
            hack=h["row"]["completion"],
            clean=c["row"]["completion"],
        ))
        diag_rows.append({
            "pid": pid,
            "hack_E": int(h["flags"]["E"]),
            "hack_C": int(h["flags"]["C"]),
            "hack_D": int(h["flags"]["D"]),
            "hack_gt": int(h["gt"]),
            "clean_gt": int(c["gt"]),
            "hack_len": len(h["row"]["completion"]),
            "clean_len": len(c["row"]["completion"]),
        })
    return pairs, diag_rows


def save_pairs_json(pairs: list[HackPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([asdict(p) for p in pairs], f)
    logger.info(f"wrote {len(pairs)} pairs -> {path}")


def main(
    pool_dir: Path = Path("data/pools/teacher_pool"),
    half_a: str = "E,C",
    half_a_signatures: str = "",
    max_pairs: int = 14,
    seed: int = 0,
    out_path: Path = Path("out/pairs_pool_halfA.json"),
) -> int:
    """Build pool-derived pairs; print audit table; save to JSON.

    Two ways to specify the half-A set:
    - --half-a=E,C: detector-level. Half-A = signatures where ANY of these
      detectors fires AND NO other detector fires. Leaky when detectors
      co-fire (entry g: E and C co-fire 99.9% in rh-s65).
    - --half-a-signatures="EC-,ECD": signature-level. Half-A is exactly these
      signatures, period. Cleaner when detectors are not independent.
    If both set, signatures wins.

    SHOULD: emit max_pairs distinct (pid, hack, clean) rows where every hack-
    side row's signature is in half-A. Every clean-side row has all detectors
    off (signature `---`).
    """
    if half_a_signatures.strip():
        sig_strs = [s.strip() for s in half_a_signatures.split(",") if s.strip()]
        half_a_sigs = {_parse_signature(s) for s in sig_strs}
        logger.info(f"building pairs: half_A_signatures={sorted(sig_strs)} max_pairs={max_pairs}")
    else:
        half_a_set = {s.strip().upper() for s in half_a.split(",") if s.strip()}
        bad = half_a_set - set(ALL_DETECTORS)
        if bad:
            raise ValueError(f"unknown detectors in --half-a: {bad}; valid: {ALL_DETECTORS}")
        half_b_set = set(ALL_DETECTORS) - half_a_set
        if not half_a_set or not half_b_set:
            raise ValueError(f"half_a must be a non-empty PROPER subset of {ALL_DETECTORS}; got {half_a_set}")
        half_a_sigs = _detectors_to_sigs(half_a_set, half_b_set)
        logger.info(f"building pairs: half_A={sorted(half_a_set)} "
                    f"half_B={sorted(half_b_set)} max_pairs={max_pairs} "
                    f"(detector-level -> {len(half_a_sigs)} signatures)")

    pairs, diag = build_pairs(pool_dir, half_a_sigs, max_pairs=max_pairs, seed=seed)
    if not pairs:
        logger.error("0 pairs emitted -- pool lacks both sides for any prompt under this split")
        return 1
    print(f"\n--- Pair audit (N={len(pairs)}; half_A={sorted(half_a_set)}) ---\n"
          "SHOULD: every hack-side row has at least one half_A column =1; no\n"
          "        half_B column =1 on hack-side. clean_gt can be 0/1 (we don't\n"
          "        gate on gt_pass).\n")
    print(tabulate(diag, headers="keys", tablefmt="pipe"))
    save_pairs_json(pairs, out_path)
    return 0


if __name__ == "__main__":
    tyro.cli(main)
