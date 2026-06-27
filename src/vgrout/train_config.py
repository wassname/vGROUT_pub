"""Typed CLI configuration for train.py.

One adapter (lora2r: rank-2r Gaussian-init LoRA, A+B trainable, three-way output
masking; see src/vgrout/lora2r.py) and three arms:

  none    no mask: the full rank-2r adapter trains and deploys (plain GRPO baseline,
          trained-capacity-matched to routeA). Was (0,0) deployed-only -> shrinkage
          confound + frac=0.5 collapse; fixed 2026-06-13.
  routeA  per-rollout three-way gate from the pooled bottleneck activation vs v_act.
  absorb  gate pinned mid (1,0): both blocks train on everything, no gate --
          tests ungated both-block training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from small_reward_hacking.rewards import EnvMode


@dataclass(kw_only=True)
class Config:
    intervention: Literal["none", "routeA", "routeV", "absorb"] = "routeA"
    # Adapter parametrisation of the deployed/quarantine blocks (src/vgrout/adapters/):
    #   lora      rank-2r block adapter, Gaussian init (baseline)
    #   pissa     same blocks, plain top-2r SVD of W, disjoint-axis (no pairs; corda's contrast-free twin)
    #   corda     same blocks, contrastive-CorDA disjoint-axis init (baked)
    #   scorda     SIGNED corda: quarantine seeded toward +(mu_hack-mu_clean) (baked; absorption candidate)
    #   scorda_rev reverse control: signed seed in the DEPLOYED block (gap <= 0 if polarity programs localization)
    #   antipasto  frozen CorDA basis + bounded gains + learned quarantine rotation (baked)
    # corda/scorda/antipasto calibrate on the authored vhack pairs (no oracle / rollout labels).
    adapter: Literal["lora", "pissa", "corda", "scorda", "scorda_rev", "antipasto"] = "lora"
    lora_r: int = 32
    lora_init_seed: int = 0

    model: str = "Qwen/Qwen3-4B"
    steps: int = 100
    group: int = 6
    max_new: int = 1024
    n_problems: int = 992
    prompts_per_step: int = 8
    lr: float = 7e-5   # canonical Ariahw setting (docs/grpo_hyperparams.md); was 1e-4
    # Reference schedule (Ariahw config.py L139-141): linear warmup over warmup_steps to
    # lr, then cosine decay to 0 over the remaining steps. Decoupled from the teacher phase.
    warmup_steps: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    clip: float = 0.2
    # Reference uses 0.1 (Ariahw config.py L142). Applied in train.py as decoupled decay
    # toward the INIT copies (not AdamW's toward-0 decay, which would drive net delta to
    # -B0@A0). Toward-init is the correct analog: it shrinks the net delta toward 0.
    weight_decay: float = 0.1
    # With grad_clip=10.0, gn rose 5->14->47->100 before the step-17 generation
    # divergence in job 15, even at lr=1e-4. Typical gn is 1-5, so grad_clip=1.0
    # suppresses these spikes while only moderately scaling typical steps.
    grad_clip: float = 1.0
    seed: int = 41
    unbiased: bool = True
    # KL-to-reference anchor (Ariahw config.py L135 beta=1e-3, verl use_kl_loss=true,
    # kl_loss_type=low_var_kl). Penalizes the policy drifting from the FROZEN initial
    # policy M0 (the warm-start base, net adapter delta 0). Without it, on-policy GRPO
    # flattens the output distribution once advantages get noisy and diverges (lp_s ->
    # -7, reward -> 0; see RESEARCH_JOURNAL 2026-06-14 b/c). Reference forward is one
    # extra no-grad pass per group via eval.disable_adapter. kl_beta=0 disables it.
    kl_beta: float = 1e-3

    vhack_refresh_every: int = 5
    # The 8 original behavior_* pairs only: the best or tied vector on all three diag
    # windows (RESEARCH_JOURNAL 2026-06-11 (d)). The `@behavior` TAG would re-add the
    # anti-aligned opportunity-aware pairs and dilute, so select by heading prefix, not
    # tag. Wave-2 arms (untested): `/behavior2` = 15 new mechanisms, `/behavior` = union.
    vhack_pairs_path: Path = Path("data/pairs/hack_pairs.md#all-in-one/behavior_")
    # Post-training retain calibration: one teacher-forced NLL epoch on the CLEAN side
    # of the same authored pairset used to build `v_act`. The phase applies the clean
    # route mask `(m=0,d=0)`, so only the deployed block updates while the quarantine is
    # absent from the forward pass.
    retain_heal_epochs: int = 1
    retain_heal_lr: float | None = None
    retain_heal_batch_size: int = 8
    # t-stat extraction (v = mean/SE per coordinate, clamp |t|<=3): null at the current
    # 8 pairs (journal (e)); revisit when the authored-pair set grows.
    vact_tstat: bool = False
    # routeA gate: route the GLOBAL tails of the score distribution. The buffer stores
    # pooled ACTS over the whole run and re-scores them against the current v_act (so a
    # v_act refresh needs no flush). Each call cuts at the route_tail_q / 1-route_tail_q
    # quantiles of that global buffer: score <= low-tail keep (0,0) | middle absorb (1,0,
    # both train) | score >= high-tail rout (1,1, deployed detached). Global (not per-batch)
    # quantiles let the per-batch routed fraction float: an all-clean batch sits below the
    # low tail -> ~nothing routed; an all-hack batch sits above the high tail -> most routed.
    # This also breaks the gate-collapse of per-window z-norm (which re-centred a saturated
    # all-hack window to 0). Symmetric tails: we do not assume where the hack/clean split is.
    # Pinned absorb until the buffer holds route_warmup scores.
    route_buffer: int = 8192   # > steps*group*prompts (4800 default) so the early clean-era
    # scores are never evicted -- they anchor the low tail once hacks dominate; a 4096
    # deque dropped them in the last ~15 steps, exactly when the keep zone needs them.
    # 0 = cut from the first scores. run-43 (the one working point) used 0. The old 128
    # default had no working run; job 40 ran warmup=128 (ema=None) and kept keep pinned to
    # 0 the whole run -- dead lower tail, gate degenerated to binary absorb/route (journal
    # 2026-06-12 (c)). job 40 also differs in frac, so warmup is not cleanly isolated.
    # Pin absorb until the buffer holds this many scores. KEEP AT 0: pinning absorb is
    # worse than imperfect early routing (any routing > none). The EMA "warms up" by just
    # accumulating samples while still routing; the short halflife below forgets the noisy
    # first batches on its own. (job 40 ran warmup=128 + ema=None and kept keep=0 all run,
    # so warmup never was the lever anyway.)
    route_warmup: int = 0
    route_tail_q: float = 0.1   # route the top and bottom q of the global score buffer (~20% tails)
    # EMA halflife (in ROLLOUTS, ~32/step) for recency-weighted gate quantiles; None =
    # unweighted whole-run quantiles. Short (96 ~= 3 steps) so the thresholds track
    # representation drift instead of being pinned by stale early scores: the buffer is
    # never evicted (a shorter deque dropped the anchor late, see route_buffer), so age-
    # weighting is the lever. Diagnosis 2026-06-13: with None the clean score distribution
    # drifts rightward as the adapter warms (tlo climbed -44->-26 while keep stayed 0 with
    # zero hacks present), so the stale low tail never matched the current batch.
    route_ema_halflife: float | None = 96.0
    # Haar-random direction control (placebo): same routing machinery, no pair signal.
    routeA_random_v_seed: int | None = None
    # ── routeV (grad-space gate) ──────────────────────────────────────────────
    # routeV routes on the c-probe GRADIENT, not the activation: per rollout it
    # backwards the completion-NLL through the deployed block's identity probe and
    # scores that grad against v_grad (extract_vhack_grad.py). The mock found the
    # motivate self-gen direction beats the Haar placebo ONLY in grad space (job
    # 136: grad F0.5 0.352 vs 0.337; act below placebo), so routeA (act) reads the
    # wrong space. Same banding/refresh/buffer machinery as routeA, reusing
    # route_tail_q / route_buffer / route_warmup / route_ema_halflife.
    # v_grad source: motivate self-gen pairs (model writes the hack side on the
    # hint-on + disclosed-mechanism prompt, the clean side hint-off), generated
    # ONCE at startup from the warm-start model (oracle-free: the only label is the
    # prompt framing). vhack_refresh_every re-extracts v_grad on those FIXED texts
    # (backward-only, cheap) so it tracks the policy without regenerating.
    routeV_n_pairs: int = 24            # motivate self-gen pairs kept for v_grad
    routeV_pool: int = 240             # problems drawn until n_pairs marker-positive hacks kept
    routeV_require_marker: bool = True  # keep only hack completions that hit the disclosed loophole marker
    # Haar-random v_grad control (placebo): same grad-gate machinery, no pair signal.
    routeV_random_v_seed: int | None = None
    # Fraction of student rollouts SAMPLED with the quarantine ablated (deployment
    # mode). frac=0: all rollouts from the full model -- the quarantine shapes
    # exploration but the deployed block is entirely off-policy, and once hacks
    # saturate the gate routes everything to quarantine, freezing the deployed block
    # (over-routing collapse; observed job 63 / u50-rep61: hack peaked step 39,
    # lp_s->NaN by step 46, final 2x2 all zeros). frac=0.25: 25% of rollouts sampled
    # deploy-mode; those activations score lower against v_act (deployed block is
    # cleaner), land in keep/absorb rather than route, and keep the deployed block
    # receiving gradients. run-43 (frac=0.25) hit deploy_hack=0, solve=12.6% IID.
    # The deploy-mode rows are returned LAST so is_ablated tags them for the per-step
    # deploy proxy. Back to 0.25 (2026-06-13 (b)): frac=0.5 was tested (job 65, 100 steps)
    # and OVER-ROUTES from step 0 -- rout-share 0.6-0.97 vs run-43's ~0.1, squeezing absorb
    # (where the deployed block trains) to ~20%, so the deployed block starves: deploy hack=0
    # AND solve=0, while the quarantine-on model still hacks 0.5-0.8. The placebo arm at
    # frac=0.5 collapsed all-zero too (arm-independent). job 72 (frac=0.25, same code/horizon)
    # keeps rout ~0.1 like run-43. Mechanism of why 0.5 over-routes from step 0 is unconfirmed.
    # 2026-06-17: frac=0 was retried (job 98) and reproduced the known frac=0 saturation
    # collapse (entries e/f: frac=0 -> over-routing/lp_s blow-up at ~step 37, 2/2). Reverted
    # to 0.25. frac is NOT the lever for the saturation kl-runaway -- 0.25 still collapses
    # later (jobs 92/93/94), so the runaway is a separate, unsolved problem (kl anchor /
    # reward saturation), independent of frac.
    gen_deploy_frac: float = 0.25

    env_mode: EnvMode = "run_tests"
    # Rotating per-step chance a problem is shown hint-free (graded by ground-truth tests);
    # keeps solve pressure alive. 10% learned solve too slowly; 25% -> 50% on 2026-06-10;
    # back to 25% on 2026-06-11 (50% -> step-17 divergence concerns). run-43 used 0.5
    # with teacher_off=20 and hit deploy_hack=0, solve=12.6% IID; 0.25 collapsed (job 63).
    unhackable_frac: float = 0.5
    teacher_pool_dir: Path | None = None
    mix_ratio: float = 0.125
    teacher_off_step: int | None = 30
    teacher_modes: tuple[str, ...] | None = None
    # Symmetric solve-teacher pool (ground-truth-passing demos). When set, the G_t
    # teacher slots split solve_mix_frac solve / (1-frac) hack, so the gate sees
    # correct examples it must not route (the routed-share discrimination diagnostic)
    # and solve pressure matches hack pressure. Needs teacher_pool_dir + mix_ratio>0.
    solve_pool_dir: Path | None = None
    solve_mix_frac: float = 0.5
    # Deterministic teacher forcing: in the teacher phase (step < teacher_off_step) every
    # generated prompt is drawn from the both-pool-covered set and gets EXACTLY
    # teacher_n_per_prompt hack + teacher_n_per_prompt solve teachers; the rest of `group`
    # are students. Constant count per prompt, no flip/coverage drops. Pool has ~1
    # rollout/prompt, so N=1 avoids sampling the same cached row twice. Replaces the
    # mix_ratio * _even_split step budget (whose count varied with flips/coverage).
    teacher_n_per_prompt: int = 1

    # Inline deploy eval cadence. One eval = TWO full generation passes over eval_n_prompts
    # (quarantine on + ablated), ~1-3 min on 4B for a +-6pp-noise number at n=32. ON by
    # default (35) because the deploy slv_dep at the FIRST eval is the earliest collapse
    # tripwire: job 65 (frac.5/100) read slv_dep=0 at step 30 -- deploy-dead before run-43's
    # stop point -- so steps 30-100 were pure waste. 35 (not 30) so a 100-step run evals at
    # 0/35/70/99 without a redundant 90+99 pair. Offline scripts/diag_deploy_ablations.py
    # still does the thorough held-out + IID pass on saved checkpoints.
    eval_ablate_every: int = 35
    eval_n_prompts: int = 32
    # HF generate + 252 per-module lora2r hooks dispatch Python per decode token, so eval
    # is GPU-starved (~19% util at bs=2). Bigger batch amortizes that fixed per-call hook
    # cost across more sequences (32 prompts -> 4 batches not 16) -> ~3x faster inline eval.
    eval_batch_size: int = 32
    save_ckpt_every: int = 10
    out_tag: str = ""

    @property
    def preset_name(self) -> str:
        return type(self).__name__.removesuffix("Config").lower() or "base"

    @property
    def arm(self) -> str:
        # {intervention}_{adapter}: the adapter suffix keeps the SVD-basis variants
        # from conflating with the lora-Gaussian runs (rename-on-logic-change). routeA
        # (act-space gate) and routeV (grad-space gate) carry distinct names so their
        # runs never conflate.
        gate = {"none": "vanilla", "routeA": "routeA",
                "routeV": "routeV", "absorb": "absorb"}[self.intervention]
        return f"{gate}_{self.adapter}"


@dataclass(kw_only=True)
class SmokeConfig(Config):
    model: str = "llamafactory/tiny-random-qwen3"
    lora_r: int = 4   # tiny model min Linear dim is 16; 2r=8 fits
    steps: int = 30
    group: int = 4
    max_new: int = 32
    n_problems: int = 100
    prompts_per_step: int = 1
    # Smoke produces 4 scores/step over 30 steps; the real 8192/128 buffer would keep the
    # gate in warmup forever. Shrink so the smoke exercises warmup AND the quantile gate
    # (keep/absorb/rout + deployed detach) within a few steps.
    route_buffer: int = 32
    route_warmup: int = 8
    # Exercise the mixed-exploration path (two generate calls + pad) in smoke.
    gen_deploy_frac: float = 0.5
    # routeV self-gen: tiny-random never hacks, so take whatever it generates (no marker
    # filter) and keep the pair set small so the one-time startup generation stays fast.
    routeV_n_pairs: int = 3
    routeV_pool: int = 6
    routeV_require_marker: bool = False


@dataclass(kw_only=True)
class FastConfig(Config):
    # Warm-start from the published first-hack bootstrap (step-10 LoRA merged into Qwen3-4B):
    # deploy solve ~0.09, deploy hack ~0 with the first exploit just emerged, so routed GRPO
    # starts past the fragile capability phase but before hack saturates (the step-20 merge
    # deploy-hacked 0.84 -- too saturated). See the research journal bootstrap notes.
    # Baseline/ceiling recipes pin --model=Qwen/Qwen3-4B for the raw base.
    model: str = "wassname/vgrout-bootstrap-firsthack-s43"
    # The warm-start (model below) already solves and hacks, so it replaces the teacher's
    # job; injecting teacher demos on top is redundant and creates a train/deploy cliff at
    # teacher_off (rew_s and as-trained hack drop the moment the demos stop, e.g. job 78
    # step 20). teacher_off_step=0 turns the teacher off from step 0: pure on-policy
    # throughout. 40 steps is enough -- hack saturates by ~step 35 and shorter runs avoid
    # the over-routing collapse that killed the 60-step rep61 jobs.
    steps: int = 40
    teacher_off_step: int | None = 0
    teacher_pool_dir: Path | None = Path("data/pools/teacher_pool")
    # The standing routeA recipe (run-43 and every real run since), promoted from
    # repeated CLI flags to defaults so jobs only pass what VARIES (intervention, seed,
    # steps, out-tag). gen_deploy_frac, route_warmup=0 and ema=None are now base Config
    # defaults (run-43's confirmed settings); these two are the recipe's teacher mix.
    solve_pool_dir: Path | None = None
    mix_ratio: float = 0.5
    group: int = 8
    max_new: int = 512
    n_problems: int = 200
    # Effective batch = group * prompts_per_step gens/step accumulated before one opt.step.
    # The inner loop backprops per prompt-group, so prompts_per_step is pure gradient
    # accumulation (no extra memory, ~linear time). Raised 4 -> 8 (64 gens/step) toward
    # the reference's 256 (16 prompts x 16 gens) to cut advantage noise; the small batch
    # was a co-driver of the divergence (RESEARCH_JOURNAL 2026-06-14 c).
    prompts_per_step: int = 8
    # adam_beta1/2 inherit the base 0.9/0.99: the aggressive 0.5/0.9 (fast-adapting Adam)
    # amplified the gn spike into the step-17 divergence. Normal betas + grad_clip=1.
    # 5e-4 diverged at step ~10, 3e-4 just pushed it to step ~27 (lp_s blew up +18->+73,
    # rew_s->0 after a clean emergence 7-24). 7e-5 matches the canonical Ariahw setting
    # (docs/grpo_hyperparams.md); the prior 1e-4 constant-peak unlearned deploy solve in
    # job 76. The reference warmup_steps=10 + cosine-to-0 schedule protects the on-policy phase.
    lr: float = 7e-5
    # Each lora2r ckpt is ~0.33G (A/B for 252 modules, bf16). A0/B0 are NOT saved -- they're
    # the seeded init (regenerable from lora_init_seed in the metadata; ckpt_update0000 is the
    # init since A==A0 at step 0), and nothing live reloads them (inline eval uses in-memory
    # wrappers). 20-step cadence keeps ~3/run for the eval curve.
    save_ckpt_every: int = 20
