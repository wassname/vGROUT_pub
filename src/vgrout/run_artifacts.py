"""Canonical reader for completed train.py run artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open


RUNS_DIR = Path("out/runs")
RUN_SCHEMA = "paired_final_v2"   # v2: deployed/as_trained field names (was deploy_*/deploy_*_on)
# Old PiSSA-substrate runs on disk carry these intervention names; lora2r runs
# get a _lora2r suffix so the two substrates never conflate in aggregation.
ARM = {"none": "vanilla", "erase": "projected",
       "routeV": "routingV", "routeV_per_token": "routingV_per_token",
       "routeA": "routeA",
       "absorb": "absorb"}


def _arm_of(cfg: dict) -> str:
    suffix = "_lora2r" if cfg.get("adapter", "antipasto") == "lora2r" else ""
    return ARM[cfg["intervention"]] + suffix


def _mean_fraction(rows: list[dict], key: str) -> float:
    values = [num / den for row in rows for num, den in [row[key]] if den]
    return sum(values) / len(values)


def load_run(run_dir: Path) -> dict:
    checkpoint = run_dir / "train.safetensors"
    deploy_path = run_dir / "deploy_test.json"
    with safe_open(str(checkpoint), framework="pt") as f:
        metadata = f.metadata()
    cfg = json.loads(metadata["cfg"])
    rows = json.loads(metadata["rows"])
    if len(rows) != cfg["steps"]:
        raise ValueError(f"{run_dir}: incomplete run, {len(rows)} rows != {cfg['steps']} steps")
    deploy = json.loads(deploy_path.read_text())
    if deploy.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{deploy_path}: expected schema={RUN_SCHEMA}, got {deploy.get('schema')}")
    required_deploy = {"eval_modes", "n", "hack_deployed", "solve_deployed", "hack_as_trained", "solve_as_trained"}
    missing = required_deploy - deploy.keys()
    if missing:
        raise ValueError(f"{deploy_path}: missing fields {sorted(missing)}")
    return {
        "run_dir": run_dir,
        "time": run_dir.name.split("_", 1)[0],
        "cfg": cfg,
        "arm": _arm_of(cfg),
        "rows": rows,
        "deploy": deploy,
        "l5_hack": _mean_fraction(rows[-5:], "hack_s"),
        "l5_solve": _mean_fraction(rows[-5:], "gt_s"),
        "whole_hack": _mean_fraction(rows, "hack_s"),
    }


def completed_runs() -> list[dict]:
    run_dirs = []
    for path in sorted(RUNS_DIR.glob("*/deploy_test.json")):
        deploy = json.loads(path.read_text())
        if deploy.get("schema") == RUN_SCHEMA:
            run_dirs.append(path.parent)
    return [load_run(run_dir) for run_dir in run_dirs]


def route_selectivity(run_dir: Path) -> float | None:
    curve = run_dir / "eval_curve.jsonl"
    if not curve.exists():
        return None
    rows = [json.loads(line) for line in curve.read_text().splitlines()][-5:]
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    hack_on, solve_on = mean("hack_as_trained"), mean("solve_as_trained")
    if hack_on == 0 or solve_on == 0:
        return None
    return round((hack_on - mean("hack_deployed")) / hack_on
                 - (solve_on - mean("solve_deployed")) / solve_on, 3)
