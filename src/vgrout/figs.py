"""Stable `docs/figs/<name>.png` -> latest generated figure under `out/`.

Plot scripts write the real PNG under out/ (gitignored, per-run/per-datatype),
then call link_latest() so docs and the blog can reference a stable path that
always points at the newest version. The symlink is relative so the repo stays
relocatable.

CAVEAT: out/ is gitignored, so the symlink target is not tracked -- the link
resolves locally but GitHub won't render it. To publish a figure, commit the
real PNG (git add -f) as well; the symlink is for local "latest" convenience.
"""
from __future__ import annotations

import os
from pathlib import Path

FIGS_DIR = Path("docs/figs")

# Reader-facing arm names. Code/log tags carry our internal vocabulary
# (routeA = the current routing arm); plots must
# not. Map every internal tag to the word a paper reader sees. Anything missing
# falls through to its raw tag, so a new arm shows up loud rather than silently
# mislabelled.
ARM_DISPLAY = {
    # routeA is the current act-gate arm; routeV (grad gate) and routing2/route2
    # (binary-tau) are retired but kept so historical run artifacts still plot.
    "routeA": "route",
    "routingV": "route (grad)", "routeV": "route (grad)",
    "routingV_per_token": "route per-token",
    "routing2": "route", "route2": "route",
    "routing2_grad": "route", "routing2_act": "route (act)",
    "projected": "erase", "route": "route", "erase": "erase", "vanilla": "vanilla",
}


def arm_label(tag: str) -> str:
    return ARM_DISPLAY.get(tag, tag)


def save_fig(fig, png_path: Path, formats=("png", "svg", "pdf")) -> Path:
    """Save one figure to every format (vector .svg/.pdf for the paper, .png for
    the blog/preview) and return the .png path. matplotlib picks the writer from
    the suffix, so we just swap it. bbox_inches='tight' so titleless figures
    don't leave a margin where the suptitle used to be."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(png_path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    return png_path


def link_latest(out_path: Path) -> Path:
    """Point docs/figs/<out_path.name> at out_path (relative symlink). Returns the link."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    link = FIGS_DIR / out_path.name
    target = os.path.relpath(out_path.resolve(), FIGS_DIR.resolve())
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
    return link
