# AGENTS.md -- vGROUT public extraction

This is novel ML research code. Extrapolate carefully, state uncertainty, and
prefer fail-fast behavior over silent fallbacks.

## Project

vGROUT tests whether an activation-space reward-hacking direction can route
GRPO updates into deployed or quarantine adapter parameters. The routeA gate
scores pooled bottleneck activations against `v_act` extracted from authored
hack/clean pairs, then assigns each rollout to keep, absorb, or route.

The current result is a partial negative for label-free vector routing. The
research journal is evidence, not polished claims:

- `docs/research_journal.md`
- `docs/human_journal.md`

## Commands

- `just smoke`: default correctness gate, tiny CPU routeA run plus verify gates.
- `just smoke-all`: vanilla, routeA, routeV, absorb.
- `just smoke-scorda`: signed-CorDA absorption check.

## Code Principles

- Fail loudly on missing or invalid assumptions.
- Do not add backward compatibility or fallback paths.
- Keep research code readable top-to-bottom.
- Preserve TODO/FIXME/HACK notes until the issue is fixed or removed.
- If code, comments, and docs disagree, trust code first, then comments, then docs.

## Data

The public extraction keeps only the small data/artifact set needed for smoke:

- `data/leetcode/`: LeetCode train/test jsonl from the reward-hacking benchmark.
- `data/pairs/`: authored contrastive pairs.
- `data/pools/teacher_pool/`: teacher rollouts used by the tiny smoke run.
- `data/pools/substrate/`: partition artifact checked by `verify_partition.py`.
- `data/pairsets/prog_wide_clean.json`: generated pairset used by the science invariant check.

Generated runs, checkpoints, logs, and caches belong under ignored `out/` or
`logs/`, not in git.
