"""Contrastive-pair schema and Markdown loader."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HackPair:
    problem_id: str
    prompt: str
    hack: str   # weak completion
    clean: str  # strong completion

    # Authored pairs share one prompt across both sides (only the completion differs).
    # Self-generated pairs (extract_vhack_grad.GenPair) override these with per-side
    # prompts (hint-on for hack, hint-off for clean); extraction reads them per side.
    @property
    def hack_prompt(self) -> str:
        return self.prompt

    @property
    def clean_prompt(self) -> str:
        return self.prompt


_PAIR_HEADING = re.compile(r"^### (.+)$", re.MULTILINE)
_TAGS = re.compile(r"^Tags: ([a-z0-9-, ]+)$", re.MULTILINE)
_FIELD = re.compile(
    r"^#### (Prompt|Hack|Clean)\n`````text\n(.*?)\n`````$",
    re.MULTILINE | re.DOTALL,
)


def load_pairs(ref: Path) -> list[HackPair]:
    """Load generated JSON or one authored `## section` from `path.md#section`."""
    path_text, separator, selector = str(ref).partition("#")
    if not separator:
        path = Path(path_text)
        if path.suffix != ".json":
            raise ValueError(f"pairset ref must be generated.json or authored.md#section, got {ref}")
        return [HackPair(**row) for row in json.loads(path.read_text())]
    if not separator or not selector:
        raise ValueError(f"pairset ref must be path.md#section, got {ref}")
    section, _, tag_text = selector.partition("@")
    # `#section/prefix` selects pairs whose heading starts with `prefix` -- the same
    # heading-prefix subsetting the per-pairset diag ranks by (behavior/opportunity/...).
    # Needed because the `behavior` TAG is too broad: it also covers the opportunity-aware
    # pairs, which the diag shows are anti-aligned with live hacks (d=-0.03 vs +0.85), so
    # `@behavior` dilutes. `/behavior_` keeps the 8 best-separating original pairs;
    # `/behavior2` the wave-2 mechanisms; `/behavior` the union of both.
    section, _, heading_prefix = section.partition("/")
    required_tags = {tag.strip() for tag in tag_text.split(",") if tag.strip()}
    path = Path(path_text)
    if path.suffix != ".md":
        raise ValueError(f"pairset ref must name a Markdown file, got {ref}")
    text = path.read_text()
    section_match = re.search(
        rf"^## {re.escape(section)}\n(.*?)(?=^## |\Z)", text,
        re.MULTILINE | re.DOTALL,
    )
    if section_match is None:
        raise ValueError(f"missing pairset section {section!r} in {path}")
    body = section_match.group(1)
    headings = list(_PAIR_HEADING.finditer(body))
    if not headings:
        raise ValueError(f"pairset section {section!r} has no pairs")
    problem_ids = [heading.group(1) for heading in headings]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError(f"pairset section {section!r} has duplicate pair headings")
    pairs = []
    for i, heading in enumerate(headings):
        chunk = body[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(body)]
        tag_rows = _TAGS.findall(chunk)
        if len(tag_rows) > 1:
            raise ValueError(
                f"{path}#{section} pair {heading.group(1)!r} has multiple Tags lines"
            )
        tags = {tag.strip() for tag in tag_rows[0].split(",")} if tag_rows else set()
        field_rows = _FIELD.findall(chunk)
        fields = dict(field_rows)
        if len(field_rows) != 3 or set(fields) != {"Prompt", "Hack", "Clean"}:
            raise ValueError(
                f"{path}#{section} pair {heading.group(1)!r} must have exactly "
                f"Prompt/Hack/Clean fields, got {sorted(fields)}"
            )
        if required_tags <= tags and heading.group(1).startswith(heading_prefix):
            pairs.append(HackPair(heading.group(1), fields["Prompt"], fields["Hack"], fields["Clean"]))
    if not pairs:
        raise ValueError(f"{path}#{selector} selected zero pairs")
    return pairs


def pairset_sha256(ref: Path) -> str:
    rows = [pair.__dict__ for pair in load_pairs(ref)]
    canonical = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
