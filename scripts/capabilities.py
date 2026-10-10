"""Generate, or check, the capability index: every public name in finitevolX.

``docs/capabilities.md`` lists each name in ``finitevolx.__all__`` once,
grouped by the private module that defines it, with the first sentence of
its docstring; names finitevolX re-exports from spectraldiffx or gaussx are
grouped by the library they come from. It then lists the public API of those
two libraries, which finitevolX builds on. Agents and people search it before
writing a helper ("Reuse before you write" in ``AGENTS.md``).

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check. The upstream sections depend
on the installed versions, which the index records; when they differ from
the installed ones, only the finitevolX part is compared.

``--check`` also fails when finitevolX exports a name that spectraldiffx or
gaussx exports for a different object, unless ``ALLOWED_SHARED_NAMES`` gives
the reason.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "capabilities.md"

# Public homes, in order: (distribution, [public modules]).
PACKAGES: dict[str, list[str]] = {
    "finitevolx": ["finitevolx"],
}

# GeoML-stack libraries finitevolX builds on.
UPSTREAM: dict[str, list[str]] = {
    "spectraldiffx": ["spectraldiffx"],
    "gaussx": ["gaussx"],
}
UPSTREAM_BLURB = {
    "spectraldiffx": (
        "Spectral transforms (DCT / DST / FFT), the rectangular spectral"
        " Poisson / Helmholtz solvers and their eigenvalues, and the spectral"
        " capacitance solver. finitevolX re-exports most of the solver API"
        " (listed above); the rest — spectral derivatives, spherical-harmonic"
        " and Chebyshev machinery — is used from here, never re-implemented."
    ),
    "gaussx": (
        "Structured linear algebra and solver strategies. finitevolX delegates"
        " the tridiagonal solve and the Nyström preconditioner to it; use it"
        " for any other structured solve, preconditioner or low-rank"
        " approximation instead of writing one here."
    ),
}

# Names finitevolX shares with an upstream library for a different object,
# and why.
ALLOWED_SHARED_NAMES: dict[str, str] = {
    "SphericalGrid2D": (
        "finitevolX's is a lat-lon Arakawa C-grid container with a ghost ring;"
        " spectraldiffx's is the spectral (spherical-harmonic) grid"
    ),
    "build_capacitance_solver": (
        "finitevolX's wraps spectraldiffx's to accept a Mask2D as well as a"
        " boolean array, and returns the same spectraldiffx CapacitanceSolver"
    ),
    "CapacitanceSolver": (
        "finitevolX re-exports spectraldiffx's spectral capacitance solver for"
        " masked domains; gaussx's is an operator-level point-constraint solve"
    ),
}

_AUTOREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# Sphinx-style roles in docstrings (:class:`~finitevolx.Mask2D` -> `Mask2D`).
_ROLE = re.compile(r":\w+:`~?(?:[\w.]+\.)?([^`]+)`")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s+(?=[A-Z])")
UPSTREAM_MARKER = "<!-- upstream -->"


def _first_line(text: str) -> str:
    """The docstring's summary: its first sentence, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    # mkdocs-autorefs links resolve only in the docs that wrote them.
    line = _AUTOREF.sub(r"\1", line)
    line = _ROLE.sub(r"`\1`", line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line.replace("|", "\\|")


def _is_alias(obj: object) -> bool:
    """A ``typing`` construct such as ``Literal[...] | None`` (callable, but no API)."""
    return type(obj).__module__ == "typing"


def _kind(obj: object) -> str:
    if _is_alias(obj):
        return "type alias"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str) or not (callable(obj) or inspect.isclass(obj)):
        module = type(obj).__module__
    return module


def _public(module: types.ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return [
        n
        for n in names
        if not n.startswith("__")
        and not isinstance(getattr(module, n, None), types.ModuleType)
    ]


def _upstream_source(name: str, obj: object) -> str | None:
    """The upstream package that exports ``obj`` under ``name``, if any."""
    for package, modules in UPSTREAM.items():
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if getattr(module, name, None) is obj:
                return package
    return None


def _summary(obj: object) -> str:
    if _is_alias(obj) or not callable(obj):
        value = repr(obj)
        return f"`{value if len(value) <= 60 else value[:57] + '...'}`".replace(
            "|", "\\|"
        )
    return _first_line(inspect.getdoc(obj) or "")


def _module_title(name: str) -> str:
    module = sys.modules.get(name) or importlib.import_module(name)
    # A few modules put ``from __future__`` above their docstring, which
    # leaves ``__doc__`` empty; fall back to the module's own name.
    return _first_line(module.__doc__ or "") or name.rsplit(".", 1)[-1]


def collect() -> tuple[dict[str, dict[str, list[tuple]]], list[str]]:
    """``{dist: {home: [(name, kind, summary, group)]}}`` + clashes."""
    index: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    owners: dict[str, dict[str, object]] = defaultdict(dict)
    for dist, modules in PACKAGES.items():
        for module_name in modules:
            module = importlib.import_module(module_name)
            for name in _public(module):
                obj = getattr(module, name)
                owners[name][f"{module_name}.{name}"] = obj
                source = _upstream_source(name, obj)
                group = f"upstream:{source}" if source else _origin(obj)
                index[dist][module_name].append(
                    (name, _kind(obj), _summary(obj), group)
                )
    clashes = []
    for name, homes in sorted(owners.items()):
        if name in ALLOWED_SHARED_NAMES:
            continue
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    for modules in UPSTREAM.values():
        for module_name in modules:
            try:
                upstream = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in _public(upstream):
                if name in ALLOWED_SHARED_NAMES:
                    continue
                for path, obj in owners.get(name, {}).items():
                    if obj is not getattr(upstream, name):
                        clashes.append(f"{name}: {path} shadows {module_name}.{name}")
    return index, clashes


def _table(rows: list[tuple]) -> list[str]:
    out = ["| Name | Kind | What it does |", "|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} |" for r in rows]
    return [*out, ""]


def _group_title(group: str) -> str:
    if group.startswith("upstream:"):
        return f"Re-exported from `{group.split(':', 1)[1]}`"
    return f"{_module_title(group).rstrip('.')} (`{group}`)"


def _finitevolx_part() -> tuple[list[str], int]:
    index, _ = collect()
    out: list[str] = []
    total = 0
    for dist, modules in PACKAGES.items():
        out += [f"## `{dist}`", ""]
        for module_name in modules:
            rows = index[dist][module_name]
            total += len(rows)
            out += [f"### `{module_name}`", ""]
            # The flat facade is grouped by the private module that defines
            # each name (one concept per group), then by upstream library.
            groups: dict[str, list[tuple]] = defaultdict(list)
            for row in rows:
                groups[row[3]].append(row)
            upstream_order = list(UPSTREAM)
            for group in sorted(
                groups,
                key=lambda g: (
                    g.startswith("upstream:"),
                    upstream_order.index(g.split(":", 1)[1])
                    if g.startswith("upstream:")
                    else 0,
                    g,
                ),
            ):
                out += [
                    f"#### {_group_title(group)}",
                    "",
                    *_table(sorted(r[:3] for r in groups[group])),
                ]
    return out, total


def _upstream_part() -> list[str]:
    out: list[str] = []
    for package, modules in UPSTREAM.items():
        out += [f"## Upstream: `{package}`", "", UPSTREAM_BLURB[package], ""]
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            rows = []
            for name in sorted(_public(module)):
                obj = getattr(module, name)
                rows.append((name, _kind(obj), _summary(obj)))
            out += [f"### `{module_name}`", "", *_table(rows)]
    return out


def _versions() -> str:
    return ", ".join(f"{p} {importlib.metadata.version(p)}" for p in UPSTREAM)


def render() -> str:
    finitevolx_part, total = _finitevolx_part()
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in finitevolX ({total} of them), each listed once,",
        "grouped by the module that defines it (names re-exported from",
        "spectraldiffx and gaussx are grouped by library), with the first",
        "sentence of its docstring; then the public API of spectraldiffx and",
        "gaussx, which finitevolX builds on. Search this page before writing a",
        "helper: if what you need is here, compose it; if it is almost here,",
        "extend it where it lives. Regenerate with `make capabilities` after",
        "changing a public API (`tests/test_capabilities.py` checks it is",
        "current).",
        "",
        *finitevolx_part,
        UPSTREAM_MARKER,
        "",
        f"Listed at {_versions()}.",
        "",
        *_upstream_part(),
    ]
    return "\n".join(out).rstrip() + "\n"


def stale(current: str, text: str) -> bool:
    """Whether ``current`` differs from ``text`` where it is comparable.

    The upstream sections are compared only when the recorded versions match
    the installed ones; the finitevolX part always is.
    """
    if current == text:
        return False
    if f"Listed at {_versions()}." in current:
        return True
    return current.partition(UPSTREAM_MARKER)[0] != text.partition(UPSTREAM_MARKER)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    _, clashes = collect()
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if stale(current, text):
            print(f"{INDEX.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
