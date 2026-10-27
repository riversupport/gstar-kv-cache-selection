# Cleanup release notes

## Source inventory

Primary source: the user's `gstar-project(1).zip` (16 entries). Included code modules:
`gstar_day1_selector.py`, `gstar_prepare_pilot.py`, `gstar_paired_pilot.py`,
`gstar_expanded_eval.py`, `gstar_extract_coverage.py`, and `gstar_coverage_eval.py`.
The uploaded notebook, pilot data, and English presentation script were also used.

The following required dependencies were absent from that ZIP and were restored
from their previously supplied project attachments in the same conversation:

| Published path | Restored source |
|---|---|
| gstar_mask_smoke.py | Earlier gstar_mask_smoke.py attachment |
| gstar_fullkv_pilot/gstar_run_fullkv.py | Earlier gstar_fullkv_pilot.zip attachment |
| gstar_fresh_coverage_test.py | Earlier gstar_fresh_coverage_test.py attachment |

The supplied `gstar_prepare_pilot.py` is relocated into `gstar_fullkv_pilot/`, matching
the dependency paths expected by the original runners. Its data-generation logic is unchanged.

## Included supporting files

- English README, reproducibility notes, third-party notices, requirements, and .gitignore.
- A cleaned historical notebook containing the 28 nonempty project code cells.
- A new-run workflow notebook built from those cells and the supplied runner APIs.
- A compact pair of previously provided fresh-test numerical result files, copied unchanged.
- The supplied presentation script, preserved as Markdown.
- Lightweight CPU/source-level tests and a cleanup manifest.

## Deliberate exclusions

- The 30,288,272-byte raw SQuAD JSON (downloaded by the existing builder when needed).
- Proposal PDF/DOCX, which describe earlier plans and are not executable project dependencies.
- Model weights, notebook authentication data, cache tensors, generated full experiment trees,
  and unpublished credential/configuration files.

Nothing was removed from the original uploaded archive or the user's Modal environment.
The release does not silently retrofit old hash-locked experiments.
