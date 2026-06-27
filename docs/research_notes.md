# Research Notes

This repository accompanies a LessWrong write-up about using steering-vector
style directions for gradient routing. The write-up URL will be added once the
post is published.

## Summary

Can steering vectors drive gradient routing? A simplified toy setting gives a
positive signal. In the more realistic reward-hacking setting tested here, the
answer is not reliably: the vectors tested here were not precise enough
classifiers of hacky versus clean solutions.

The more promising result is signed-CorDA initialization. Instead of using a
vector as a live router, signed-CorDA initializes two adapter halves so hacky
and clean gradients are biased toward different blocks. In the current runs,
the strongest 4B result reduced held-out hack rate from 0.759 as-trained to
0.218 after deployment ablation, while solve rate moved from 0.161 to 0.149.
That is mechanism evidence but not a deployable operating point.

## Main claim

The label-free routing gate is a negative result on current evidence. The
strongest offline checks did not show that authored activation directions beat
Haar-random directions as high-precision routing classifiers.

This does not mean there is no hack direction. Oracle-fit rollout directions do
show a moderate linear signal. The failure appears to be transfer: directions
built from synthetic pairs did not align well enough with the live rollout
distribution.

## Why precision matters

Gradient routing can tolerate missed forget examples because ambiguous samples
can fall into the shared block and be handled by absorption. Wrong confident
pins are more expensive: a hack routed into deployed parameters is retained,
and a clean solution routed into quarantine can be deleted at deployment.

That is why the routing direction was evaluated as a high-precision classifier,
using precision-weighted metrics such as F0.5 rather than only AUROC.

## What to reproduce

Run:

```bash
uv sync
just smoke
```

This exercises the tiny-model routeA path, reward/eval invariants, adapter
masking, activation extraction, masked GRPO, and deployment ablation.

The full 4B experiments require GPU runs and are not packaged as a one-command
reproduction in this minimal public repo.

## Main large-run evidence

The headline signed-CorDA result used a 4B model in the reward-hacking LeetCode
environment. In the seed-44 absorb run, the held-out test metrics were:

| state | hack rate | solve rate |
| --- | ---: | ---: |
| as-trained | 0.759 | 0.161 |
| deployment ablation | 0.218 | 0.149 |

Interpreting the hack-rate drop literally, roughly 71% of the measured hack
capability was removed by ablating the quarantine block. The residual hack rate
is still too high for a clean intervention result.
