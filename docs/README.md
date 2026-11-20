# Documentation build

finitevolX's documentation is built by **two tools**, deployed as one site.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — theory pages, usage guides, research notes, notebooks | [mystmd](https://mystmd.org) | `docs/myst.yml` (its `toc`), `docs/*.md`, `docs/design/research/`, `docs/notebooks/` | `/` |
| API reference | [Zensical](https://zensical.org) + mkdocstrings | `mkdocs.yml`, `docs/api/` | `/reference/` |

Build both and assemble them with:

```bash
make docs          # build both halves into public/ and check every link
make docs-serve    # build, then serve public/ at http://127.0.0.1:8000
make docs-check    # validate the sources without rendering the MyST theme
make docs-api      # build only the API reference into site/
```

mystmd is a Node CLI: install it with `npm install -g mystmd`. The Python side needs the `docs` and `pipekit` extras (`uv sync --extra docs --extra pipekit`); the Make targets pass them. `.github/workflows/docs.yml` builds and checks the site on every pull request, and `.github/workflows/pages.yml` deploys `public/` from `main` with GitHub Actions.

## The table of contents

`docs/myst.yml` lists every page: the home page, the theory pages and usage guides by topic, the example notebooks grouped by topic, the research notes and the changelog. A page that is not in its `toc` is not built, so add each new notebook there. mystmd gives every page a flat URL from its file name, with `_` turned into `-` (`notebooks/swm_linear.ipynb` is served at `/swm-linear/`, and a leading `NN_` is dropped). mystmd has no way to set a slug, so a second page with the same file name would be served at `<name>-1/`; keep file names unique across the prose tree. The nav shows the `title` given to the page's entry in the `toc`, and the page keeps its first `# ` heading as its title.

MkDocs-Material syntax renders as plain text in MyST, without a warning: use `:::{note}` (with `:class: dropdown` for a collapsible) rather than `!!! note` or `??? note`, `::::{tab-set}` with `:::{tab-item}` rather than `=== "…"`, `::::{grid}` with `:::{card}` rather than a `grid cards` div, and an HTML `<img width=…>` rather than `{ width=… }`.

The API half keeps its own `nav` in `mkdocs.yml`, grouped by the same topics.

## Notebooks are jupytext `.py` files

The example notebooks are jupytext percent-format `.py` files, and mystmd reads only `.md` and `.ipynb`. `scripts/build_docs.py` therefore writes an unexecuted `.ipynb` beside each `.py` before the MyST build (`docs/notebooks/*.ipynb` is gitignored), and the `toc` lists those. The pages show code and markdown without cell outputs, as they did under mkdocs-jupyter; the figures are the PNGs each notebook saves under `docs/images/<notebook>/` and embeds from a markdown cell. Paths in a notebook are relative to the notebook file (`../images/<notebook>/name.png`), and a link to a notebook points at its `.ipynb`; the build fails on a link to a notebook's `.py`.

## Why the split

mystmd is much better at prose: real cross-references, first-class notebook execution, exports to PDF/LaTeX, and MyST's directive syntax. What it does not have is an autodoc equivalent — there is no mature way to render Python docstrings into a MyST site today.

MkDocs + mkdocstrings already does that well, and it publishes a Sphinx-compatible `objects.inv`, which is exactly what mystmd needs to cross-reference *into* it. So each tool does the half it is good at.

The API half is built by [Zensical](https://zensical.org), the successor to Material for MkDocs (which reached end of life on 2026-11-05). It reads `mkdocs.yml` as is and runs mkdocstrings natively, so the config, the API pages and the inventory are unchanged; only the build command is `zensical build` instead of `mkdocs build`. Zensical does not run arbitrary MkDocs plugins or hooks (unsupported ones are silently ignored), so check <https://zensical.org/docs/compatibility/mkdocs/plugins/> before adding one.

## Site-nav URLs must be absolute

The theme re-renders the site nav from the config it embeds for hydration, and prepends `BASE_URL` to any nav URL starting with `/`. A nav entry that already carries the deployment prefix is therefore doubled — `/finitevolX/finitevolX/reference/` — and 404s.

This bites only *after* hydration: the server-rendered HTML is correct, so `curl` and `verify_links` both see a healthy link. Keep `site.nav` URLs absolute (mystmd rejects `/reference/` and `reference/` anyway) and never rewrite them at build time. `nav_base_url_problems` in `scripts/build_docs.py` fails the build if one becomes root-relative.

The trade-off is that a local preview's nav button points at the deployed site. In-page `xref:` links are unaffected — the theme does not re-prefix those, so they are rewritten to `{BASE_URL}/reference/...` as normal.

## If the theme download is blocked

`myst build --html` fetches the site template as a zip from GitHub. Behind a corporate proxy or a restrictive egress policy that request can fail with a 403 while ordinary git access still works. Clone the template and point at it locally instead:

```bash
git clone --depth 1 https://github.com/myst-templates/book-theme.git /tmp/book-theme
```

Then set `site.template` in `docs/myst.yml` to `/tmp/book-theme` for that build. Everything else is unchanged.

Note that the rendered pages still load KaTeX, Font Awesome, and jupyter-matplotlib stylesheets from CDNs at view time. Without network access in the *browser*, maths renders doubled — KaTeX ships an accessibility MathML copy that its stylesheet is responsible for hiding. That is a viewing artefact, not a build problem.

## Cross-references from prose into the API

In any MyST page, link to an API object with the `xref:` protocol and the name as exported from the top-level package:

```markdown
[`Mask2D`](xref:api#gridax_jax.Mask2D)
[`Advection2D`](xref:api#gridax_jax.Advection2D)
```

A target that does not exist in the inventory fails `myst build --strict`, so broken API links are caught on the pull request rather than in production.

## Never link to a file mystmd does not build

mystmd builds only the pages in its `toc`. A relative prose link to any other Markdown file, such as an API page under `docs/api/` (`[Masks](api/masks.md)`), does not fail: mystmd publishes the file as a raw download, the link resolves, and the reader gets a Markdown file. A link to a folder (`[examples/](examples/)`) does not resolve at all. Link to an API object with `xref:api#<name>`, to a whole API page by its deployed URL (`https://jejjohnson.github.io/gridax/reference/masks/`), and to a prose page only once it is in the `toc`. `source_link_problems` in `scripts/build_docs.py` fails the build on any of these.

## Known upstream issue: the `$` anchor abbreviation

A Sphinx inventory may record an object's anchor as the literal `$`, meaning "the anchor is the object's own name". mystmd 1.11.0 **lowercases the name when it expands that abbreviation**, which breaks links into mkdocstrings' case-sensitive anchors:

| Inventory entry | mystmd emits | Correct? |
|---|---|---|
| `pkg.Name` -> `page/#pkg.Name` | `#pkg.Name` | yes |
| `pkg.mod.Name` -> `page/#$` | `#pkg.mod.name` | **no** |

The failure is silent: the link resolves, the page loads, and the browser simply cannot find the anchor.

**In practice this does not bite** when you link to a name as its package exports it: mkdocstrings gives every name an API page lists with `::: package.Name` an *explicit* anchor. So the natural way to write a link is also the safe one:

```markdown
[`Mask2D`](xref:api#gridax_jax.Mask2D)      <!-- safe: explicit anchor -->
[`Name`](xref:api#pkg.mod.Name)          <!-- risky if its anchor is `$` -->
```

Prefer the top-level name. Two safety nets back that up in `scripts/build_docs.py`:

- `restore_anchor_case` repairs the lowercased anchors after the build, using the inventory as the source of truth. Delete it once mystmd fixes the expansion upstream.
- `verify_links` checks that **every** internal link in the assembled site resolves to a file that exists and, when it carries a fragment, to an anchor that is really in that file. It sees both generators' output at once, so it catches this class of bug regardless of cause — keep it either way.

Both operate on the **static** HTML. Anything the theme re-renders on hydration — the site nav above being the case that actually bit us — is invisible to them, which is why that one needs its own check.
