"""Render the gridax README assets: the icon and the two logos.

Every output is a plain SVG written next to this file. The icon shares the
GeoStack tile (a 120-unit rounded square, r = 26) in a teal gradient. Its glyph
is one staggered C-grid cell: the T point at the centre (amber), the U points on
the east and west faces (pink), the V points on the north and south faces (sky)
and the X points at the corners (lime), each outlined in navy so it holds on
the teal.

The logos come in a light and a dark variant for GitHub's ``<picture>`` /
``prefers-color-scheme`` switch. The README's schematics are drawn in geostack
(``docs/images/libraries/gridax_*``) with the stack's other library figures,
and geostack's dependency figure carries a copy of the icon.

Run from the repo root::

    uv run --no-project --with loguru --with cyclopts python docs/assets/render.py
"""

from __future__ import annotations

from pathlib import Path

from cyclopts import App
from loguru import logger

OUT = Path(__file__).parent
app = App(help="Write gridax's icon and its light and dark logos as SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Teal: the tile's gradient and the wordmark's accent.
C1, C2 = "#14B8A6", "#0E7490"
ACCENT = {"light": "#0E7490", "dark": "#2DD4BF"}
THEME = {"light": {"text": "#1F2937"}, "dark": {"text": "#E5E7EB"}}

W = "#FFFFFF"
EDGE = "#0F172A"
# one colour per point type of the C-grid
T_POINT, U_POINT, V_POINT, X_POINT = "#FBBF24", "#F472B6", "#38BDF8", "#A3E635"


# ---------------------------------------------------------------- glyph


def _arrow(cx: float, cy: float, direction: str, fill: str, s: float = 10.0) -> str:
    """A triangle centred on a face, pointing east (U) or north (V)."""
    points = {
        "east": [(cx + s, cy), (cx - s, cy - s), (cx - s, cy + s)],
        "north": [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)],
    }[direction]
    path = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    return (
        f'<path d="{path} Z" fill="{fill}" stroke="{EDGE}" stroke-width="2" '
        'stroke-linejoin="round"/>'
    )


def glyph_cell() -> str:
    """One C-grid cell with its T, U, V and X points."""
    lo, hi, mid = 28.0, 100.0, 64.0
    parts = [
        f'<rect x="{lo:g}" y="{lo:g}" width="{hi - lo:g}" height="{hi - lo:g}" '
        f'rx="2" fill="{W}" fill-opacity="0.12" stroke="{W}" stroke-width="3.5"/>',
        f'<circle cx="{mid:g}" cy="{mid:g}" r="13.75" fill="{T_POINT}" '
        f'stroke="{EDGE}" stroke-width="2"/>',
    ]
    parts += [_arrow(x, mid, "east", U_POINT) for x in (lo, hi)]
    parts += [_arrow(mid, y, "north", V_POINT) for y in (lo, hi)]
    half = 8.125
    parts += [
        f'<rect x="{x - half:g}" y="{y - half:g}" width="{2 * half:g}" '
        f'height="{2 * half:g}" rx="2" fill="{X_POINT}" stroke="{EDGE}" '
        'stroke-width="2"/>'
        for x in (lo, hi)
        for y in (lo, hi)
    ]
    return "".join(parts)


# ---------------------------------------------------------------- pieces


def tile(gid: str, x: float = 0, y: float = 0, size: float = 128) -> str:
    """The gradient tile with the glyph, placed at (x, y), ``size`` wide."""
    s = size / 128
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/>'
        "</linearGradient></defs>"
        f'<g transform="translate({x} {y}) scale({s:g})">'
        f'<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#{gid})"/>'
        f"{glyph_cell()}</g>"
    )


def svg(width: int, height: int, label: str, title: str, body: str) -> str:
    """A standalone SVG document with a title and an accessible label."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">\n'
        f"<title>{title}</title>\n{body}\n</svg>\n"
    )


def wordmark(x: float, y: float, size: float, mode: str) -> str:
    """The gridax wordmark, its last letter in the accent colour."""
    t = THEME[mode]
    return (
        f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
        f'font-weight="700" letter-spacing="{-size / 48:.2f}" fill="{t["text"]}">'
        f'grida<tspan fill="{ACCENT[mode]}">x</tspan></text>'
    )


# ---------------------------------------------------------------- outputs


@app.default
def main() -> None:
    """Write icon.svg, logo-light.svg and logo-dark.svg next to this script."""
    (OUT / "icon.svg").write_text(svg(128, 128, "gridax", "gridax", tile("ic")))
    logger.info("wrote icon.svg")
    for mode in THEME:
        body = tile(f"logo-{mode}", x=6, y=6) + wordmark(154, 92, 72, mode)
        path = OUT / f"logo-{mode}.svg"
        path.write_text(svg(440, 140, "gridax", "gridax", body))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
