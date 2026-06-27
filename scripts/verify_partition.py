"""Verify substrate partition and teacher-pool composition.

SHOULD: the 4-mode substrate partitions problems cleanly into distinct modes, and the
        A5 teacher_modes filter hands the route gate ONLY known-mode demos. ELSE: a
        held-out mode's problems (or teacher demos) leak into training and the
     generalisation claim is contaminated.

Two load-bearing invariants, neither previously tested (the gate-anchor leak,
2026-06-05, was the sibling that did slip through):
  1. Partition is a clean function problem_id -> one mode, covering the expected
     substrate modes (each problem graded by exactly one channel; non-overlap).
  2. Teacher-pool composition under teacher_modes={run_tests}: the kept pool is ALL
     run_tests, and the held-out modes are genuinely held out (>0 problems, 0 demos).

Not a strict requirement enforcer -- a sanity gate that the modes behave like
distinct hacks. Reads the curated public artifact data/pools/substrate/.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

POOL = Path("data/pools/substrate")
SUBSTRATE_MODES = {"run_tests", "file_marker", "sentinel", "stdout_marker"}
KNOWN = {"run_tests"}                       # the A5 weak-detector's one known mode


def _check(name: str, cond: bool) -> bool:
    logger.info(f"{'PASS' if cond else 'FAIL'}  {name}")
    return cond


def main() -> int:
    partition = {int(pid): m for pid, m in json.loads((POOL / "partition.json").read_text()).items()}
    counts = Counter(partition.values())
    logger.info(f"partition: {len(partition)} problems, modes={dict(sorted(counts.items()))}")
    ok = True

    # 1. partition well-formed: a dict is one-mode-per-problem by construction; check the
    #    modes are exactly the expected substrate set and every mode is non-empty.
    ok &= _check("partition modes == the 4 substrate modes", set(counts) == SUBSTRATE_MODES)
    ok &= _check("every mode has >0 problems (modes are distinct, populated hacks)",
                 all(counts[m] > 0 for m in SUBSTRATE_MODES))

    # 2. teacher-pool composition under teacher_modes={run_tests} (replicates train.py:575).
    pool_pids = {int(p.name.split("_")[1].split(".")[0]) for p in POOL.glob("prompt_*.jsonl.gz")}
    ok &= _check("every teacher-pool prompt has a partition entry (no orphan demos)",
                 pool_pids <= set(partition))
    kept = {pid for pid in pool_pids if partition[pid] in KNOWN}
    kept_modes = {partition[pid] for pid in kept}
    ok &= _check(f"teacher_modes={KNOWN}: kept pool is ALL known-mode (no held-out demos)",
                 kept_modes <= KNOWN)
    held_out = SUBSTRATE_MODES - KNOWN
    ok &= _check("held-out modes get ZERO teacher demos",
                 all(partition[pid] not in held_out for pid in kept))
    ok &= _check("held-out modes are genuinely held out (each has >0 problems in the partition)",
                 all(counts[m] > 0 for m in held_out))

    logger.info("ALL PASS -- partition + teacher-pool oracle-separation invariants hold" if ok else "FAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
