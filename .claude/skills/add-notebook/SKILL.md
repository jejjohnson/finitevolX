---
name: add-notebook
description: Add or update an example notebook in finitevolX's docs — a jupytext percent-format .py file in docs/notebooks/ that saves its figures as PNGs under docs/images/<notebook>/ and embeds them from markdown cells, rendered by mystmd without execution and listed under Examples in the toc of docs/myst.yml. Use when asked to write, extend, re-run or fix a tutorial, example, demo or benchmark notebook under docs/notebooks.
---

# Add or update a notebook

finitevolX's notebooks are **not** executed `.ipynb` files. Each one is a
jupytext percent-format `.py` file in `docs/notebooks/`; `scripts/build_docs.py` converts it to an unexecuted `.ipynb` (gitignored) that mystmd renders, so the page shows the code and markdown but no cell outputs. Every figure is therefore saved by the script itself
and embedded as an image.

## 1. Plan it

- One question per notebook; read the existing ones (`docs/notebooks/*.py`
  and the `Examples` part of the `toc` in `docs/myst.yml`) so you extend rather than
  duplicate.
- Use the public API the way a user would (`import gridax as fvx`, `import gridax.solvers as gs`, or `from gridax import ...`); no `gridax._src` imports.
- Keep it fast: small grids, short integrations; it is re-run by hand to
  refresh the figures.

## 2. Write `docs/notebooks/<name>.py`

- The jupytext header (copy it from `docs/notebooks/demo_masks.py`), then
  `# %% [markdown]` cells (title, goal, a link to the theory page such as
  `[Advection Theory](../advection.md)`) and `# %%` code cells.
- Figures, as the existing notebooks do:

  ```python
  from pathlib import Path

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "<name>"
  IMG_DIR.mkdir(parents=True, exist_ok=True)
  ...
  fig.savefig(IMG_DIR / "result.png", dpi=150, bbox_inches="tight")
  plt.show()
  ```

  and embed each one in the next markdown cell:
  `# ![Result](../images/<name>/result.png)` (the path is relative to the notebook file). Write maths for KaTeX: `1/2`, not `½`, inside `$…$`.
- Enable float64 explicitly if the notebook relies on it
  (`jax.config.update("jax_enable_x64", True)`); pin random keys.
- Keep PNGs small (the pre-commit `check-added-large-files` hook rejects
  files over 500 kB).

## 3. Run it and wire it in

```bash
uv run python docs/notebooks/<name>.py      # writes docs/images/<name>/*.png
```

- Commit the `.py` and the PNGs together; re-run the script whenever the
  code or the API it uses changes, so the figures match the code.
- Add `- file: notebooks/<name>.ipynb` with a short `title:` under its topic in `Examples` in the `toc` of `docs/myst.yml`. It is served at `/<name>/`, with `_` turned into `-`. Link to it from other pages as `notebooks/<name>.ipynb`, never the `.py`.
- `docs/` is excluded from ruff, so review the code cells yourself
  (88 columns, imports at the top of the cell that needs them).

## 4. Verify

```bash
make docs        # needs mystmd: npm install -g mystmd
```

It fails on MyST errors and any broken link, and lists MyST warnings without failing, so read its log for your page. Then open the page with `make docs-serve` to check that every image renders.
