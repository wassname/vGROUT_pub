"""Final paired deployed/as-trained scores from completed structured run artifacts."""
from __future__ import annotations

import argparse
import polars as pl
from tabulate import tabulate
import json
import re
from pathlib import Path

from vgrout.run_artifacts import RUN_SCHEMA, RUNS_DIR, route_selectivity


CFG_LINE = re.compile(r"^\s+([A-Za-z0-9_/]+)\s+:\s+(.*)$")


def _log_cfg(log_path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    in_cfg = False
    for line in log_path.read_text(errors="replace").splitlines():
        if "resolved config:" in line:
            in_cfg = True
            continue
        if not in_cfg:
            continue
        match = CFG_LINE.match(line)
        if match:
            cfg[match.group(1)] = match.group(2).strip()
        elif line and not line.startswith(" "):
            in_cfg = False
    return cfg


def _float_cfg(cfg: dict[str, str], key: str) -> float | None:
    raw = cfg[key]
    if raw == "None":
        return None
    return float(raw)


def _is_realistic_run(row: dict) -> bool:
    return (
        row["time"] >= "20260618T000000"
        and row["teacher_off_step"] == 0
        and row["gen_deploy_frac"] == 0.0
        and row["unhackable_frac"] == 0.25
    )


def _completed_deploy_rows() -> list[dict]:
    rows = []
    for deploy_path in sorted(RUNS_DIR.glob("*/deploy_test.json")):
        run_dir = deploy_path.parent
        if "_smoke_" in run_dir.name:
            continue
        deploy = json.loads(deploy_path.read_text())
        if deploy.get("schema") != RUN_SCHEMA:
            continue
        log_path = Path(deploy["log"])
        if not log_path.exists():
            log_path = Path("logs") / f"{run_dir.name}.log"
        cfg = _log_cfg(log_path)
        required_cfg = {"teacher_off_step", "mix_ratio", "gen_deploy_frac", "lr", "kl_beta"}
        if not required_cfg <= cfg.keys():
            continue
        row = {
            "time": run_dir.name.split("_", 1)[0],
            "headline": deploy["solve_deployed"] - deploy["hack_deployed"],
            "hack_deployed": deploy["hack_deployed"],
            "solve_deployed": deploy["solve_deployed"],
            "hack_as_trained": deploy["hack_as_trained"],
            "solve_as_trained": deploy["solve_as_trained"],
            "gap": deploy["hack_as_trained"] - deploy["hack_deployed"],
            "select": route_selectivity(run_dir),
            "arm": deploy["arm"],
            "adapter": deploy["adapter"],
            "seed": deploy["seed"],
            "steps": deploy["steps"],
            "teacher_off_step": int(_float_cfg(cfg, "teacher_off_step")),
            "mix_ratio": _float_cfg(cfg, "mix_ratio"),
            "gen_deploy_frac": _float_cfg(cfg, "gen_deploy_frac"),
            "unhackable_frac": deploy["unhackable_frac"],
            "lr": _float_cfg(cfg, "lr"),
            "kl_beta": _float_cfg(cfg, "kl_beta"),
            "n": deploy["n"],
            "modes": ",".join(deploy["eval_modes"]),
            "run": run_dir.name,
        }
        row["realistic"] = _is_realistic_run(row)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realistic-only", action="store_true",
                        help="Keep post-2026-06-18, teacher-off-from-step-0, gen-deploy-frac=0, unhackable=0.25 runs.")
    parser.add_argument("--min-steps", type=int, default=0,
                        help="Drop completed runs shorter than this many optimizer steps.")
    args = parser.parse_args()

    rows = _completed_deploy_rows()
    if args.realistic_only:
        rows = [row for row in rows if row["realistic"]]
    if args.min_steps:
        rows = [row for row in rows if row["steps"] >= args.min_steps]
    if not rows:
        print("no completed non-smoke deploy runs in out/runs/")
        return
    df = pl.DataFrame(rows).sort("headline", descending=True)
    cols = ["realistic", "time", "headline", "hack_deployed", "solve_deployed",
            "hack_as_trained", "solve_as_trained", "gap", "select", "arm", "adapter",
            "seed", "steps", "teacher_off_step", "mix_ratio", "gen_deploy_frac",
            "unhackable_frac", "lr", "kl_beta", "n", "modes", "run"]
    title = "Realistic no-teacher final paired test eval" if args.realistic_only else "Final paired test eval"
    print(f"\n## {title}, sorted by deployed solve-hack\n")
    print(tabulate(df.select(cols).rows(), headers=cols, tablefmt="pipe", floatfmt="+.3f"))


if __name__ == "__main__":
    main()
