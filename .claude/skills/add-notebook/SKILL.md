---
name: add-notebook
description: Add or update an example notebook in finitevolX's docs — a jupytext percent-format .py file in docs/notebooks/ that saves its figures as PNGs under docs/images/<notebook>/ and embeds them from markdown cells, rendered by mkdocs-jupyter without execution and listed in the mkdocs.yml Examples nav. Use when asked to write, extend, re-run or fix a tutorial, example, demo or benchmark notebook under docs/notebooks.
---

# Add or update a notebook

finitevolX's notebooks are **not** executed `.ipynb` files. Each one is a
jupytext percent-format `.py` file in `docs/notebooks/`; mkdocs-jupyter
renders it with `execute: false`, so the page shows the code and markdown
but no cell outputs. Every figure is therefore saved by the script itself
and embedded as an image.

## 1. Plan it

- One question per notebook; read the existing ones (`docs/notebooks/*.py`
  and the `Examples` section of `mkdocs.yml`) so you extend rather than
  duplicate.
- Use the public API the way a user would (`import finitevolx as fvx` or
  `from finitevolx import ...`); no `finitevolx._src` imports.
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
  `# ![Result](../../images/<name>/result.png)`.
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
- Add `- <Title>: notebooks/<name>.py` under `Examples` in `mkdocs.yml`
  (and under the topic's section if it is a tutorial for it).
- `docs/` is excluded from ruff, so review the code cells yourself
  (88 columns, imports at the top of the cell that needs them).

## 4. Verify

```bash
uv run mkdocs build 2>&1 | grep WARNING   # no line may mention your page
```

(`--strict` does not pass yet because of existing griffe docstring
warnings; if `pandoc` is installed locally, take it off `PATH` first, as
`AGENTS.md` explains) and open the page with `make docs-serve` to check that every image
renders and every link resolves.
