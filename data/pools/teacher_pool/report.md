# probe_distill report

![rollout stack](rollout_stack.png)

*Rollout outcomes per training step for projected GRPO at seed=41. Schedule: 0 steps of student-generated rollouts, then None steps of replay-distillation from a saved teacher+base pool, then 20 steps of student-generated rollouts. Categories: correct (green), correct with attempted reward hack (yellow), reward hack (red), attempted reward hack (purple), incorrect (grey). Values are a 10-step trailing moving average. Dashed lines mark distillation on/off.*

## metadata

```json
{
  "arm": "projected",
  "seed": 41,
  "tag": "teacher_pool",
  "steps": 20,
  "pre_warmup_steps": 0,
  "warmup_replay_steps": null,
  "group": 8,
  "n_problems": 300,
  "argv": [
    "/workspace/projected_grpo/src/projected_grpo/probe_distill.py",
    "--teacher-only",
    "--n-problems=300",
    "--group=8",
    "--max-new=1024"
  ],
  "pre": {
    "hack": 0.98125,
    "pass": 0.34375,
    "n_steps": 20
  },
  "post": {
    "hack": NaN,
    "pass": NaN,
    "n_steps": 0
  }
}
```
