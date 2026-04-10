# Golden-output regression fixtures

This directory holds the canonical input fields, the deterministic
coastal mask, and the per-operator golden ``.npz`` files that the
masks-everywhere regression suite asserts against.

## Layout

| File | Purpose |
|---|---|
| ``inputs.py`` | Single source of truth for the 16×16 grid, the cross-shaped island mask, and the smooth analytic input fields (``h``, ``u``, ``v``, ``q``, ``f``). Both 2-D and 3-D variants. |
| ``_helpers.py`` | ``load_golden`` / ``assert_matches_golden`` / ``save_golden`` used by tests and by the generator. |
| ``_gen_golden.py`` | Re-runnable generator that materialises every entry it knows about into ``golden/<operator>__<method>__<variant>.npz``. |
| ``golden/`` | Generated ``.npz`` files. Committed to the repo; the test suite reads them at assertion time. |

## How to regenerate

```bash
uv run python tests/fixtures/_gen_golden.py
```

The script is idempotent — running it twice produces bit-identical
output. Re-run it whenever:

* an operator's math intentionally changes (and update the test or PR
  description to explain why);
* a new operator method gets added with mask support (also add the
  corresponding entry in ``_register_all`` inside ``_gen_golden.py``);
* you add a new variant — e.g. ``"all_ocean"`` for the
  "mask=ArakawaCGridMask.from_dimensions(...) ≡ unmasked" sanity test.

After regeneration, ``git diff tests/fixtures/golden/`` should be
**part of your PR diff** — that's the audit trail showing exactly which
operator outputs moved.

## Why golden files instead of analytic checks

Most of the operators in this repo already have analytic-truth tests
elsewhere (constant-field → 0, linear field → constant, etc.). The
goldens here are a different kind of test: they pin the **exact bit
pattern** the operators produce on a fixed, non-trivial input. That's
the right shape of test for a refactor that is supposed to be a no-op
on existing code paths and an additive op on new ones — any drift
shows up immediately, with the offending ``.npz`` file pointing at the
exact operator and variant that moved.

## Why these specific inputs

* **16×16** is small enough that golden ``.npz`` files stay diff-able
  but large enough to have a non-trivial interior, a coastline, and a
  cross-shaped island whose ``mask.psi`` (strict 4-of-4 corner
  propagation) differs from ``mask.h`` near the corners.
* **Smooth analytic fields** — sines and polynomials of normalised
  ``(x, y)``, see ``inputs.py`` — are deterministic across JAX
  versions and easy to reason about by hand.
* **Single mask** — there is exactly one canonical mask, used by every
  operator test, so reviewers only need to internalise one geometry.
