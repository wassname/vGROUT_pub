"""Regrade cached teacher pool with all 4 viable upstream booleans, report
distribution + co-occurrence + exclusive-signature counts.

Background (docs/spec/20260528_cross_mechanism_v_hack.md): upstream provides
5 boolean detectors per rollout. One of them (prompt_test_func_response_pass)
structurally cannot fire on simple_overwrite_tests variant (prompt has no
embedded test_func), so we work with the other 4:

    GT = eq_correct (response Solution passes ground-truth tests)
    E  = eq_hinted (response's run_tests succeeds against response's own
         Solution -- the loophole was exercised)
    C  = response_test_func_arbitrary_pass (response's tests pass even on a
         stub Solution -- tests are trivial)
    D  = NOT response_test_func_gt_pass (response's tests REJECT the canonical
         Solution -- assertions are wrong)

Output: per-detector counts, 4x4 co-occurrence matrix, exclusive-signature
breakdown (binary indicator across {E,C,D}; GT logged separately as the
"got it right" axis).

Run: uv run python -m vgrout.regrade_pool
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import tyro
from loguru import logger
from tabulate import tabulate

from small_reward_hacking.rewards import compute_reward
from small_reward_hacking.data import DATA


def load_problems_by_id() -> dict[int, dict]:
    # NB: we score against the bare prompt. The hint is now per-mode
    # (train.HINT_REPLACE_TO is a dict), and the C/D mechanisms don't read the
    # prompt anyway, so no single-string replacement is well-defined or needed.
    out = {}
    with DATA.open() as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            problem_id = d.get("id", idx)
            out[problem_id] = {
                "canonical_solution": d.get("canonical_solution", ""),
                "gt_tests": d["gt_answer"],
                "setup_code": d.get("setup_code", ""),
                "func_name": d.get("func_name", "Solution().solve"),
            }
    return out


DETECTORS = ("E", "C", "D")  # ordered for stable signature reporting; GT logged aside


def main(pool_dir: Path = Path("data/pools/teacher_pool"), require_audit: bool = True) -> int:
    probs = load_problems_by_id()
    logger.info(f"loaded {len(probs)} problems from dataset")

    paths = sorted(pool_dir.glob("prompt_*.jsonl.gz"))
    logger.info(f"regrade {len(paths)} prompt files from {pool_dir}")

    per_detector = Counter()                # marginal: how often each fires
    cooc = {(a, b): 0 for a in DETECTORS for b in DETECTORS}  # 3x3 co-occurrence
    sig_counts = Counter()                  # exclusive signatures (E,C,D) -> bool tuple
    gt_by_sig = Counter()                   # per-signature: how many also passed GT
    cached_vs_new_C = Counter()             # sanity: cached `hacked` ↔ regraded C
    n_total = 0
    for path in paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                d = json.loads(line)
                completion = d["completion"]
                prob_id = d["problem_id"]
                if prob_id not in probs:
                    continue
                prob = probs[prob_id]
                r = compute_reward(
                    completion,
                    canonical_solution=prob["canonical_solution"],
                    gt_tests=prob["gt_tests"],
                    setup_code=prob["setup_code"],
                    func_name_hint=prob["func_name"],
                )
                flags = {"E": r.hacked_loophole_used, "C": r.hacked, "D": r.hacked_wrong_tests}
                GT = r.gt_pass
                for k, v in flags.items():
                    if v:
                        per_detector[k] += 1
                for a in DETECTORS:
                    for b in DETECTORS:
                        if flags[a] and flags[b]:
                            cooc[(a, b)] += 1
                sig = tuple(flags[k] for k in DETECTORS)
                sig_counts[sig] += 1
                if GT:
                    gt_by_sig[sig] += 1
                cached_vs_new_C[(d["hacked"], r.hacked)] += 1
                n_total += 1

    print("\n--- Per-detector marginals ---\n"
          "SHOULD: each detector fires on a non-trivial fraction (>=10%) of rollouts\n"
          "        if cross-mechanism splits are to have power.\n")
    print(tabulate(
        [{"detector": k, "n": per_detector[k], "pct": f"{100*per_detector[k]/max(1,n_total):.1f}%",
          "meaning": {
              "E": "eq_hinted -- loophole used (resp.tests pass on resp.Solution)",
              "C": "arbitrary_pass -- resp.tests pass on stub Solution (trivial)",
              "D": "wrong_tests -- canonical fails resp.tests (assertions wrong)",
          }[k]}
         for k in DETECTORS],
        headers="keys", tablefmt="pipe",
    ))

    print("\n--- Co-occurrence matrix (rollouts where both fire) ---\n"
          "SHOULD: off-diagonal cells non-zero where mechanisms can co-occur (e.g. E^C\n"
          "        common since C is a subset-ish of E). If E^D = 0, D-hacks never\n"
          "        used the loophole = bug or impossible-to-reach configuration.\n")
    print(tabulate(
        [{"": a, **{b: cooc[(a, b)] for b in DETECTORS}} for a in DETECTORS],
        headers="keys", tablefmt="pipe",
    ))

    print(f"\n--- Exclusive signatures over {DETECTORS} ---\n"
          "SHOULD: >=3 non-singleton signatures (cells with n>=20) -- else half-A/half-B\n"
          "        split won't give >=20 in each held-out cell.\n")
    rows = []
    for sig, n in sorted(sig_counts.items(), key=lambda kv: -kv[1]):
        rows.append({
            "signature": "".join(d if v else "-" for d, v in zip(DETECTORS, sig)),
            "E": int(sig[0]), "C": int(sig[1]), "D": int(sig[2]),
            "n": n, "pct": f"{100*n/max(1,n_total):.1f}%",
            "gt_pass_n": gt_by_sig[sig],
            "gt_pass_pct": f"{100*gt_by_sig[sig]/max(1,n):.1f}%",
        })
    print(tabulate(rows, headers="keys", tablefmt="pipe"))
    print(f"\nN_total={n_total}")

    print("\n--- Sanity: cached `hacked` vs re-graded C (should agree) ---")
    print(tabulate(
        [{"cached_hacked": ch, "regraded_C": rc, "n": cached_vs_new_C[(ch, rc)]}
         for ch in (True, False) for rc in (True, False)],
        headers="keys", tablefmt="pipe",
    ))

    # Viability gates per spec 20260528_g2_g3_checkpoint_selection.md R1:
    # (a) >=3 non-singleton signatures (n>=20 each)
    # (b) >=1 non-EC signature (anything other than EC- / ECD) with n>=50
    # (c) no signature exceeds 60% of the pool
    EC_SIGS = {(True, True, False), (True, True, True)}  # EC-, ECD
    n_viable_sigs = sum(1 for n in sig_counts.values() if n >= 20)
    a_ok = n_viable_sigs >= 3
    non_ec_max = max((n for sig, n in sig_counts.items() if sig not in EC_SIGS), default=0)
    b_ok = non_ec_max >= 50
    top_pct = 100 * max(sig_counts.values(), default=0) / max(1, n_total)
    c_ok = top_pct < 60.0

    def cue(ok: bool) -> str:
        return "🟢" if ok else "🔴"

    print(
        f"\n{cue(a_ok)} R1.a ({n_viable_sigs} signatures with n>=20; need >=3)"
        f"\n{cue(b_ok)} R1.b (largest non-EC signature n={non_ec_max}; need >=50)"
        f"\n{cue(c_ok)} R1.c (top signature pct={top_pct:.1f}%; need <60%)"
    )
    viable = a_ok and b_ok and c_ok
    print(f"{cue(viable)} OVERALL: {'viable' if viable else 'degenerate'}")
    return 0 if (viable or not require_audit) else 1


if __name__ == "__main__":
    tyro.cli(main)
