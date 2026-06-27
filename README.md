# vGROUT

vGROUT is a small research codebase for testing whether an activation-space
reward-hacking direction can route GRPO updates into deployed or quarantine
adapter parameters.

The current evidence is a partial negative for label-free vector routing. The
hand-authored activation direction did not beat Haar-random directions as a
high-precision routing classifier in the strongest offline checks. The useful
mechanism so far is signed-CorDA absorption: in one 4B run, deployment ablation
reduced held-out hack rate from 0.759 to 0.218 while solve rate moved from
0.161 to 0.149. That is mechanism evidence, not a deployable operating point.

An accompanying [LessWrong write-up](https://www.lesswrong.com/posts/kzri5W2uBfF2mdboK/can-we-use-steering-vectors-to-suppress-reward-hacking-1). The public evidence summary
is in [docs/research_notes.md](docs/research_notes.md).

This repository is the minimal public extraction of the working core:

- `src/vgrout/`: GRPO loop, routeA/routeV gates, and block-split adapters.
- `src/small_reward_hacking/`: local LeetCode reward-hacking environment.
- `scripts/verify_*.py`: fail-fast invariant checks used by `just smoke`.
- `data/pairs/`: authored contrastive pairs used to extract `v_act`.
- `data/leetcode/`: benchmark jsonl files needed by smoke/eval.
- `data/pools/` and `data/pairsets/`: small curated artifacts needed by the
  correctness gates.
- `docs/research_notes.md`: concise public notes matching the write-up.

## Quick Start

```bash
uv sync
just smoke
```

`just smoke` runs the real tiny-model routeA pathway: reward/eval/split
invariant checks, adapter mask checks, activation-direction extraction, routing
from pooled activations, masked GRPO, and deployment ablation.

Useful variants:

```bash
just smoke-vanilla
just smoke-routeA
just smoke-absorb
just smoke-scorda
```

## Main Hypothesis

Prior gradient-routing work routes updates using labels. vGROUT asks whether a
synthetic activation-space direction can replace those labels: score each
rollout by pooled bottleneck activations dotted with `v_act`, then update the
deployed block, quarantine block, or both.

The decisive comparison is real `v_act` against a Haar-random direction while
watching quarantine update mass as a confound.

## Status

The present result is not a clean success:

- oracle-labelled rollout directions show a real linear signal;
- hand-authored oracle-free pairs do not reliably align with that signal;
- signed-CorDA absorption partially localizes hack capability without routing.

The code is intentionally fail-fast research code. Missing files, stale
assumptions, and invalid configs should crash loudly.
