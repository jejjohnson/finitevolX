#!/usr/bin/env python3
"""Build and assemble the two-tool documentation site.

The docs are produced by two generators (see ``docs/README.md``):

- **mystmd** builds the prose half — home page, theory pages, usage guides,
  notebooks — into ``docs/_build/html``. The notebooks are jupytext ``.py``
  files, which mystmd cannot read, so each is first converted to an
  unexecuted ``.ipynb`` beside it (gitignored); their figures are the PNGs
  committed under ``docs/images/``.
- **Zensical + mkdocstrings** builds the API reference into ``site/`` from
  ``mkdocs.yml``, and publishes a Sphinx-compatible ``objects.inv`` alongside it.

This script runs both, assembles them into ``public/`` (prose at the root, API
reference under ``/reference/``), repairs the cross-references between them,
and then verifies that every internal link in the assembled site resolves —
including the ones that cross from one generator's output into the other's,
which neither tool can check on its own.

Usage:
    python scripts/build_docs.py            # full build into public/
    python scripts/build_docs.py --check    # validate sources, no HTML render

``--check`` runs both generators in their strict validation modes without
rendering the MyST theme, which needs network access to fetch the template.
"""

from __future__ import annotations

from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import zlib

import cyclopts
import jupytext
from loguru import logger

REPO = Path(__file__).resolve().parent.parent
MKDOCS_OUT = REPO / "site"
MYST_DIR = REPO / "docs"
MYST_OUT = MYST_DIR / "_build" / "html"
PUBLIC = REPO / "public"
API_SUBDIR = "reference"
# Plain files served at the site root, next to the prose pages. mystmd publishes
# only pages it renders, so these are copied in by `assemble`.
ROOT_FILES: tuple[str, ...] = ("llms.txt",)
NOTEBOOKS = MYST_DIR / "notebooks"

# Must match `references.api` in docs/myst.yml. mystmd only loads intersphinx
# inventories over http(s), so the freshly built Zensical output is served here
# during the MyST build and the resulting URLs are rewritten afterwards.
API_PORT = 8910
# No trailing slash: mystmd echoes the configured inventory URL into the page
# config it embeds for hydration, and it normalises away the trailing slash.
# Matching the bare origin rewrites that echo as well as the links, so the leak
# check below can be absolute about it.
API_ORIGIN = f"http://127.0.0.1:{API_PORT}"


# ---------------------------------------------------------------------------
# Sphinx inventory
# ---------------------------------------------------------------------------


def parse_inventory(data: bytes) -> dict[str, str]:
    """Parse a Sphinx ``objects.inv`` v2 payload.

    Args:
        data: Raw bytes of an ``objects.inv`` file.

    Returns:
        A mapping of object name to the URI recorded for it, with the ``$``
        abbreviation (meaning "the anchor equals the object name") expanded.

    Raises:
        ValueError: If the payload is not a version 2 inventory.
    """
    header, _, rest = data.partition(b"\n")
    if b"version 2" not in header:
        msg = f"unsupported inventory header: {header!r}"
        raise ValueError(msg)
    # Three more comment lines follow the version banner.
    for _ in range(3):
        _, _, rest = rest.partition(b"\n")

    entries: dict[str, str] = {}
    body = zlib.decompress(rest).decode("utf-8")
    for line in body.splitlines():
        match = re.match(r"(?x)(.+?)\s+(\S+)\s+(-?\d+)\s+(\S*)\s+(.*)", line)
        if match is None:
            continue
        name, _role, _prio, uri, _display = match.groups()
        entries[name] = uri.replace("$", name) if uri.endswith("$") else uri
    return entries


def anchor_case_map(entries: dict[str, str]) -> dict[str, str]:
    """Map lowercased anchors back to their true, case-sensitive spelling.

    mystmd 1.11.0 lowercases the fragment of every intersphinx-resolved
    cross-reference, which breaks links into mkdocstrings' case-sensitive
    anchors. This mapping is what lets [`restore_anchor_case`][] undo that.

    Anchors that are ambiguous after lowercasing are omitted rather than
    guessed at.

    Args:
        entries: Inventory mapping from [`parse_inventory`][].

    Returns:
        A mapping of lowercased anchor to correctly cased anchor, containing
        only anchors that differ from their lowercased form and are unique.
    """
    seen: dict[str, set[str]] = {}
    for uri in entries.values():
        _, _, anchor = uri.partition("#")
        if anchor:
            seen.setdefault(anchor.lower(), set()).add(anchor)
    return {
        lowered: next(iter(variants))
        for lowered, variants in seen.items()
        if len(variants) == 1 and next(iter(variants)) != lowered
    }


# ---------------------------------------------------------------------------
# HTML rewriting
# ---------------------------------------------------------------------------

_HREF = re.compile(r'href="([^"]*)"')
_NAV = re.compile(r'"nav":\s*(\[.*?\])', re.DOTALL)


def nav_base_url_problems(html: str) -> list[str]:
    """Find site-nav URLs that the theme will prefix with ``BASE_URL`` twice.

    The MyST theme re-renders the site nav from the config it embeds for
    hydration, prepending ``BASE_URL`` to any URL that starts with ``/``. A nav
    entry that already carries the deployment prefix therefore ends up doubled
    (``/project/project/page/``) and 404s — but only after hydration, so the
    static HTML looks correct and [`verify_links`][] cannot see it.

    Nav URLs must be absolute, which the theme treats as external and leaves
    alone.

    Args:
        html: A rendered page's source.

    Returns:
        A list of offending nav URLs; empty when every entry is safe.
    """
    problems: list[str] = []
    for block in _NAV.findall(html):
        for url in re.findall(r'"url"\s*:\s*"([^"]*)"', block):
            if url.startswith("/"):
                problems.append(url)
    return sorted(set(problems))


def restore_anchor_case(html: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Repair lowercased API anchors in one HTML document.

    Args:
        html: The document source.
        mapping: Lowercased-to-true anchor map from [`anchor_case_map`][].

    Returns:
        The repaired source and the number of anchors changed.
    """
    count = 0

    def fix(match: re.Match[str]) -> str:
        nonlocal count
        href = match.group(1)
        base, sep, anchor = href.partition("#")
        if not sep or anchor not in mapping:
            return match.group(0)
        count += 1
        return f'href="{base}#{mapping[anchor]}"'

    return _HREF.sub(fix, html), count


def rewrite_api_origin(html: str, origin: str, replacement: str) -> tuple[str, int]:
    """Replace the temporary localhost API origin with the deployed path.

    Args:
        html: The document source.
        origin: The origin baked in during the build, e.g.
            ``http://127.0.0.1:8910/``.
        replacement: What to put in its place, e.g. ``/reference/``.

    Returns:
        The rewritten source and the number of replacements made.
    """
    return html.replace(origin, replacement), html.count(origin)


# ---------------------------------------------------------------------------
# Link verification
# ---------------------------------------------------------------------------


class _LinkCollector(HTMLParser):
    """Collect ``href`` targets and element ids from a document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if (ident := values.get("id")) is not None:
            self.ids.add(ident)
        if tag == "a" and (name := values.get("name")) is not None:
            self.ids.add(name)
        if tag == "a" and (href := values.get("href")) is not None:
            self.hrefs.append(href)


def _collect(path: Path) -> _LinkCollector:
    parser = _LinkCollector()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def _resolve(root: Path, page: Path, href: str) -> Path | None:
    """Resolve an internal href to the file on disk that should serve it."""
    target = href.split("?", 1)[0]
    if target.startswith("/"):
        candidate = root / target.lstrip("/")
    else:
        candidate = (page.parent / target).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    elif not candidate.suffix:
        indexed = Path(f"{candidate}/index.html")
        candidate = indexed if indexed.exists() else candidate
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def verify_links(root: Path, base_url: str = "") -> list[str]:
    """Check that every internal link in the assembled site resolves.

    Both the target document and, when the link carries a fragment, the anchor
    inside it must exist. This is the only check in the pipeline that sees
    both generators' output at once.

    Args:
        root: The assembled site root, normally ``public/``.
        base_url: Deployment path prefix to strip from absolute links, so a
            site built for ``/project/`` validates against the local tree.

    Returns:
        A list of human-readable problems; empty when the site is sound.
    """
    root = root.resolve()
    problems: list[str] = []
    ids_cache: dict[Path, set[str]] = {}

    # A 404 page is served in place of any missing path, so its root-absolute
    # links are correct at runtime even though they do not resolve from where
    # the file happens to sit. Index its anchors, but do not check its links.
    pages = sorted(root.rglob("*.html"))
    for page in pages:
        ids_cache[page] = _collect(page).ids

    for page in pages:
        if page.name == "404.html":
            continue
        rel = page.relative_to(root)
        for raw in _collect(page).hrefs:
            if not raw or raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            href = raw
            if base_url and href.startswith(f"{base_url}/"):
                href = href[len(base_url) :]
            path_part, _, anchor = href.partition("#")
            if not path_part:
                continue
            target = _resolve(root, page, path_part)
            if target is None or not target.exists():
                problems.append(f"{rel}: dead link -> {href}")
                continue
            if anchor:
                ids = ids_cache.get(target)
                if ids is None:
                    ids = _collect(target).ids
                    ids_cache[target] = ids
                if anchor not in ids:
                    problems.append(f"{rel}: missing anchor -> {href}")
    return problems


# ---------------------------------------------------------------------------
# Source links to files mystmd does not build
# ---------------------------------------------------------------------------

#: prose folders whose files mystmd never renders as pages
UNBUILT_DIRS = ("api", "templates")
_SOURCE_LINK = re.compile(r"\]\(([^)\s]+)")
# fenced blocks and inline code, whose links are examples rather than links
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_TOC_FILE = re.compile(r"^\s*-?\s*file:\s*(\S+)\s*$", re.MULTILINE)


def toc_files(root: Path) -> set[str]:
    """The pages the MyST table of contents builds, relative to ``root``."""
    return set(_TOC_FILE.findall((root / "myst.yml").read_text(encoding="utf-8")))


def _prose_texts(root: Path) -> list[tuple[str, str]]:
    """The Markdown of every prose page, keyed by its path relative to ``root``.

    Walks symlinked folders too (design pages linked in from packages), and
    reads ``.md`` files and notebook markdown cells.
    """
    texts = []
    for folder, dirs, files in os.walk(root, followlinks=True):
        rel_dir = Path(folder).relative_to(root)
        if rel_dir.parts and rel_dir.parts[0] in ("_build", *UNBUILT_DIRS):
            dirs.clear()
            continue
        for name in sorted(files):
            page = Path(folder) / name
            rel = (rel_dir / name).as_posix()
            if rel == "README.md" or page.suffix not in (".md", ".ipynb"):
                continue
            if not page.is_file():
                continue
            if page.suffix == ".md":
                texts.append((rel, page.read_text(encoding="utf-8")))
            else:
                cells = json.loads(page.read_text(encoding="utf-8"))["cells"]
                md = [
                    "".join(c["source"]) for c in cells if c["cell_type"] == "markdown"
                ]
                texts.append((rel, "\n".join(md)))
    return texts


def source_link_problems(root: Path) -> list[str]:
    """Relative prose links to a file or folder mystmd does not build as a page.

    mystmd publishes a linked file it does not render, such as an API page
    under ``docs/api/`` or a Markdown page left out of the ``toc``, as a raw
    download. The link then resolves, so ``verify_links`` passes it, but the
    reader gets a Markdown file. A link to a folder does not resolve at all.
    Link to an API object with ``xref:api#<name>``, to an API page by its
    deployed ``/reference/`` URL, and to a prose page only if it is in the
    ``toc``.

    Args:
        root: The MyST project directory (``docs/``).

    Returns:
        One message per offending link.
    """
    built = toc_files(root)
    problems = []
    for rel, text in _prose_texts(root):
        for target in _SOURCE_LINK.findall(_CODE.sub("", text)):
            path = target.split("#", 1)[0]
            if not path or ":" in path.split("/", 1)[0]:
                continue
            linked = os.path.normpath((Path(rel).parent / path).as_posix())
            first = linked.split("/", 1)[0]
            if first in UNBUILT_DIRS:
                why = f"is in docs/{first}/, which mystmd publishes as a raw download"
            elif path.endswith("/") or (root / linked).is_dir():
                why = "is a folder, not a page"
            elif Path(linked).suffix == ".py" and linked.startswith("notebooks/"):
                why = "is a jupytext source; link to its .ipynb, which mystmd builds"
            elif Path(linked).suffix in (".md", ".ipynb") and linked not in built:
                why = "is not in the toc, so mystmd publishes it as a raw download"
            else:
                continue
            problems.append(f"{rel}: {target!r} {why}")
    return problems


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run(command: list[str], cwd: Path) -> None:
    logger.info("$ {}", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _serve(directory: Path) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", API_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def convert_notebooks() -> int:
    """Write an unexecuted ``.ipynb`` beside each jupytext ``.py`` notebook.

    The ``.py`` file stays the source of truth; the ``.ipynb`` is gitignored
    and rewritten on every build, so it never goes stale.

    Returns:
        The number of notebooks converted.
    """
    sources = sorted(NOTEBOOKS.glob("*.py"))
    for source in sources:
        jupytext.write(jupytext.read(source), source.with_suffix(".ipynb"))
    logger.info("converted {} jupytext notebooks to .ipynb", len(sources))
    return len(sources)


def build_api() -> None:
    """Build the Zensical API reference into ``site/``."""
    _run(
        [
            "uv",
            "run",
            "--extra",
            "docs",
            "zensical",
            "build",
            "--strict",
        ],
        REPO,
    )


def build_prose(*, html: bool) -> None:
    """Build the MyST prose site.

    Args:
        html: Render the themed HTML site. When false, only the site content
            and cross-references are built and validated, which needs no
            network access for the theme template.
    """
    target = "--html" if html else "--site"
    # mystmd caches fetched inventories. Purge them so the prose is always
    # validated against the API reference just built, not a previous run's.
    for stale in (MYST_DIR / "_build" / "cache").glob("xrefs-intersphinx-*"):
        stale.unlink()
    server = _serve(MKDOCS_OUT)
    try:
        _run(["myst", "build", target, "--strict"], MYST_DIR)
    finally:
        server.shutdown()


def assemble(base_url: str) -> list[str]:
    """Combine both builds into ``public/`` and repair cross-references.

    Args:
        base_url: Deployment path prefix, e.g. ``/finitevolX``.

    Returns:
        The list of link problems found in the assembled site.

    Raises:
        SystemExit: If either generator's output is missing.
    """
    if not MYST_OUT.is_dir():
        sys.exit(f"missing MyST output at {MYST_OUT}")
    if not MKDOCS_OUT.is_dir():
        sys.exit(f"missing Zensical output at {MKDOCS_OUT}")

    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    shutil.copytree(MYST_OUT, PUBLIC)
    shutil.copytree(MKDOCS_OUT, PUBLIC / API_SUBDIR)
    for name in ROOT_FILES:
        shutil.copy2(MYST_DIR / name, PUBLIC / name)

    inventory = parse_inventory((MKDOCS_OUT / "objects.inv").read_bytes())
    mapping = anchor_case_map(inventory)
    replacement = f"{base_url}/{API_SUBDIR}"

    fixed_anchors = 0
    rewritten = 0
    for page in PUBLIC.rglob("*.html"):
        if page.is_relative_to(PUBLIC / API_SUBDIR):
            continue
        html = page.read_text(encoding="utf-8")
        html, anchors = restore_anchor_case(html, mapping)
        html, origins = rewrite_api_origin(html, API_ORIGIN, replacement)
        if anchors or origins:
            page.write_text(html, encoding="utf-8")
        fixed_anchors += anchors
        rewritten += origins

    logger.info("repaired {} lowercased API anchors", fixed_anchors)
    logger.info("rewrote {} API links to {}", rewritten, replacement)

    problems: list[str] = []
    for page in PUBLIC.rglob("*.html"):
        if page.is_relative_to(PUBLIC / API_SUBDIR):
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        name = page.relative_to(PUBLIC)
        if API_ORIGIN in text:
            problems.append(f"{name}: build-time API origin leaked into output")
        for url in nav_base_url_problems(text):
            problems.append(
                f"{name}: site-nav URL {url!r} is root-relative; the theme will "
                f"prepend BASE_URL to it again on hydration. Use an absolute URL."
            )
    return problems + verify_links(PUBLIC, base_url)


app = cyclopts.App(help=__doc__.splitlines()[0], result_action="return_value")


@app.default
def main(*, check: bool = False) -> int:
    """Run the documentation build.

    Args:
        check: Validate sources without rendering the themed MyST site.

    Returns:
        ``0`` on success, ``1`` if the assembled site has broken links.
    """
    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    convert_notebooks()
    unbuilt = source_link_problems(MYST_DIR)
    if unbuilt:
        logger.error("{} prose link(s) into unbuilt folders:", len(unbuilt))
        for problem in unbuilt:
            logger.error("  - {}", problem)
        return 1

    build_api()
    build_prose(html=not check)
    if check:
        logger.success("sources validated (themed HTML not rendered)")
        return 0

    problems = assemble(base_url)
    if problems:
        logger.error("{} broken link(s) in the assembled site:", len(problems))
        for problem in problems:
            logger.error("  - {}", problem)
        return 1
    logger.success("assembled site is link-clean -> {}", PUBLIC)
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
