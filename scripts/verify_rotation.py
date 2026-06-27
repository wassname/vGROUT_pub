"""Verify the rotating-unhackable flip (train.py step loop).

The bug this guards: the gt_only flip used to be frozen per-pid at load time
(`random.Random(f"unhack-{seed}-{pid}")`), so the SAME ~10% of problems were
unhackable every step. A fixed subset is memorizable; the model never has
to learn to genuinely solve the rest. Rotation seeds on (seed, STEP, pid) so the
unhackable subset changes every step -- over training every problem is sometimes
hint-free.

SHOULD, checked below:
  1. Each problem carries BOTH renderings: `messages` discloses the loophole
     mechanism hint; `messages_gt` is the plain "and should pass all tests" prompt
     with NO loophole disclosed. (The flip changes the prompt.)
  2. The per-step flip seed ROTATES: the unhackable set at step s != step s+1, and
     over many steps a fixed pid is flipped on some steps and not others. (Not frozen.)
  3. The grader half lives in verify_rewards.py (gt_only credits no channel); together
     they prove the flip changes BOTH the hint and the evaluation, not just the hint.
"""
from __future__ import annotations

import random
import sys

from loguru import logger

from small_reward_hacking.data import HINT_REPLACE_TO, RH_HINT_REPLACE_FROM, load_problems

SEED = 41
FRAC = 0.1


def _flip(seed: int, step: int, pid: int, frac: float) -> bool:
    # MUST match train.py step loop verbatim.
    return random.Random(f"unhack-{seed}-{step}-{pid}").random() < frac


def main() -> int:
    logger.info("argv: " + " ".join(sys.argv))
    probs = load_problems(40, ["run_tests", "sentinel"], seed=SEED, shuffle=True)
    ok = True

    # 1. both renderings exist; messages discloses loophole, messages_gt does not.
    for p in probs[:5]:
        loop = next(m["content"] for m in p["messages"] if m["role"] == "user")
        plain = next(m["content"] for m in p["messages_gt"] if m["role"] == "user")
        loophole_hint = HINT_REPLACE_TO[p["env_mode"]]
        disclosed = loophole_hint in loop and loophole_hint not in plain
        plain_ok = RH_HINT_REPLACE_FROM in plain
        if not (disclosed and plain_ok):
            logger.error(f"pid {p['problem_id']} ({p['env_mode']}): "
                         f"loophole_disclosed_only_in_messages={disclosed} plain_ok={plain_ok}")
            ok = False
    logger.info("[1] messages disclose loophole, messages_gt are hint-free: "
                f"{'PASS' if ok else 'FAIL'}")

    # 2. rotation: subset changes step-to-step, and no pid is frozen across steps.
    pids = [p["problem_id"] for p in probs]
    sets = [{pid for pid in pids if _flip(SEED, s, pid, FRAC)} for s in range(50)]
    n_changes = sum(sets[s] != sets[s + 1] for s in range(len(sets) - 1))
    flipped_pid = next(pid for pid in pids
                       if 0 < sum(_flip(SEED, s, pid, FRAC) for s in range(50)) < 50)
    rotates = n_changes >= 40 and flipped_pid is not None
    logger.info(f"[2] over 50 steps the unhackable subset changed {n_changes}/49 step-pairs; "
                f"pid {flipped_pid} is flipped on some steps, not all: "
                f"{'PASS' if rotates else 'FAIL (frozen subset!)'}")
    ok = ok and rotates

    if not ok:
        logger.error("ROTATION VERIFY FAILED")
        return 1
    logger.info("ROTATION VERIFY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
