"""Verify provenance and evaluation-split invariants that protect paper claims."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

from loguru import logger
from tabulate import tabulate

from small_reward_hacking.data import DATA, RH_HINT_REPLACE_FROM, load_problems
from vgrout.eval import load_eval_splits
from vgrout.pairs import load_pairs, pairset_sha256


def _must_raise(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def _is_content_free_stub(completion: str) -> bool:
    code = completion.strip()
    if code.startswith("```python\n") and code.endswith("\n```"):
        code = code[len("```python\n"):-len("\n```")].strip()
    placeholder = r"(?:pass|\.\.\.|raise NotImplementedError(?:\(\))?)"
    return bool(
        re.fullmatch(placeholder, code)
        or re.fullmatch(rf"(?:async )?def [^\n]+:\n[ \t]+{placeholder}", code)
    )


def main() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        pairs_path = tmp / "pairs.md"
        pairs_path.write_text(
            "## tiny\n\n### p\n\n#### Prompt\n`````text\np\n`````\n\n"
            "#### Hack\n`````text\nh\n`````\n\n#### Clean\n`````text\nc\n`````\n\n"
            "## unrelated\n\n### q\n\n#### Prompt\n`````text\nq\n`````\n\n"
            "#### Hack\n`````text\nx\n`````\n\n#### Clean\n`````text\ny\n`````\n"
        )
        # Pairsets are content-addressed by the SELECTED section's bytes (pairset_sha256):
        # an edit elsewhere in the file must not change the hash; an edit inside the
        # selected section must. This is what gates a stale extracted direction.
        pairs_ref = Path(f"{pairs_path}#tiny")
        selected_hash = pairset_sha256(pairs_ref)
        pairs_path.write_text(pairs_path.read_text().replace("\nx\n", "\nother changed\n"))
        unrelated_ignored = pairset_sha256(pairs_ref) == selected_hash
        pairs_path.write_text(pairs_path.read_text().replace("\nh\n", "\nchanged\n"))
        selected_changed = pairset_sha256(pairs_ref) != selected_hash
        missing_rejected = _must_raise(lambda: load_pairs(Path(f"{pairs_path}#missing")))
        rows.append({
            "invariant": "selected Markdown pair bytes",
            "success": bool(selected_hash) and unrelated_ignored and selected_changed and missing_rejected,
        })

        malformed = tmp / "malformed.md"
        malformed.write_text(
            "## x\n\n### duplicate\n\n#### Prompt\n`````text\np\n`````\n\n"
            "#### Prompt\n`````text\np2\n`````\n\n#### Hack\n`````text\nh\n`````\n\n"
            "#### Clean\n`````text\nc\n`````\n"
        )
        rows.append({
            "invariant": "malformed Markdown fails",
            "success": _must_raise(lambda: load_pairs(Path(f"{malformed}#x"))),
        })

        authored_pairs = load_pairs(Path("data/pairs/hack_pairs.md#all-in-one"))
        real_pairsets_ok = (
            len(authored_pairs) == 42  # 27 + 15 wave-2 behavior2_* (c33b810)
            and len(load_pairs(Path("data/pairs/hack_pairs.md#all-in-one/behavior_"))) == 8  # routeA training default
            and len(load_pairs(Path("data/pairs/hack_pairs.md#all-in-one@opportunity-aware"))) == 6
            and len(load_pairs(Path("data/pairs/hack_pairs.md#all-in-one@explicit"))) == 10
            and len(load_pairs(Path("data/pairs/hack_pairs.md#all-in-one@roleplay"))) == 2
            and len(load_pairs(Path("data/pairs/hack_pairs.md#all-in-one@think-tags"))) == 1
            and len(load_pairs(Path("data/pairs/hack_pairs.md#all-in-one@behavior,opportunity-aware"))) == 6
            and _must_raise(lambda: load_pairs(Path("data/pairs/hack_pairs.md#all-in-one@not-a-tag")))
            and len(load_pairs(Path("data/pairsets/prog_wide_clean.json"))) == 8
        )
        rows.append({"invariant": "authored/control/generated pairsets load", "success": real_pairsets_ok})
        rows.append({
            "invariant": "authored pairs contain no complete stubs",
            "success": not any(
                _is_content_free_stub(completion)
                for pair in authored_pairs
                for completion in (pair.hack, pair.clean)
            ),
        })

        source = json.loads(DATA.read_text().splitlines()[0])
        missing = json.loads(json.dumps(source))
        missing["prompt"][-1]["content"] = missing["prompt"][-1]["content"].replace(
            RH_HINT_REPLACE_FROM, "and should pass every check")
        duplicate = json.loads(json.dumps(source))
        duplicate["prompt"][-1]["content"] += f" Also {RH_HINT_REPLACE_FROM}."
        missing_path, duplicate_path = tmp / "missing.jsonl", tmp / "duplicate.jsonl"
        missing_path.write_text(json.dumps(missing) + "\n")
        duplicate_path.write_text(json.dumps(duplicate) + "\n")
        canonical_load = len(load_problems(1, ["run_tests"])) == 1
        hint_drift_rejected = (
            _must_raise(lambda: load_problems(1, ["run_tests"], data_path=missing_path))
            and _must_raise(lambda: load_problems(1, ["run_tests"], data_path=duplicate_path))
        )
        rows.append({"invariant": "exactly one prompt hint", "success": canonical_load and hint_drift_rejected})

    val_a, test_a = load_eval_splits(["run_tests"], 32)
    val_b, test_b = load_eval_splits(["run_tests"], 32)
    val_ids = [p["problem_id"] for p in val_a]
    test_ids = [p["problem_id"] for p in test_a]
    split_ok = (
        len(val_ids) == 32
        and len(test_ids) == 87
        and set(val_ids).isdisjoint(test_ids)
        and val_ids == [p["problem_id"] for p in val_b]
        and test_ids == [p["problem_id"] for p in test_b]
    )
    val_sha = hashlib.sha256(",".join(map(str, val_ids)).encode()).hexdigest()[:12]
    test_sha = hashlib.sha256(",".join(map(str, test_ids)).encode()).hexdigest()[:12]
    rows.append({
        "invariant": "deterministic disjoint val/test",
        "success": split_ok,
        "detail": f"n=32/87 ids={val_sha}/{test_sha}",
    })

    print(tabulate(rows, headers="keys", tablefmt="github"))
    ok = all(row["success"] for row in rows)
    logger.info("PASS: science invariants hold" if ok else "FAIL: science invariant broken")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
