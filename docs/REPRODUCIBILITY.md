# Reproducibility and checkpoint compatibility

## What changed

The numerical selection/attention/decoding/scoring algorithms were not intentionally
changed. Korean comments and docstrings were translated into English, diagnostic
messages were translated, and direct print calls were converted into logger calls.
The English model prompts, reference answers, candidate grids, budgets, constraints,
article splitting rules, and reported metrics were preserved.

Even these non-algorithmic edits change source-file hashes. The experiment runners
capture hashes of source code and data, then require the same versions when resuming.
We intentionally did **not** weaken these checks, rewrite old protocol files, forge
old hashes, or merge new output into old experiment identities.

## Safe use

- Publish this directory as a separate clean release.
- Keep the current Modal project and its code snapshots untouched for historical resumption.
- For a new cleaned-code run, use a separate directory with this full source layout.
- Do not copy old `results/` or `experiments/` into the new directory and expect automatic resumption.
- Record model revision, library versions, prompts, data hashes, and all output artifacts.
- The cleaned historical notebook retains old paths as documentation; the new-run notebook
  derives paths from returned values and contains explicit held-out/fresh-test opt-in cells.

Neither an unchanged prompt nor a successful CPU regression test guarantees identical
floating-point GPU outputs. Generated-answer changes can affect token-level F1. Do not
combine results from different environments or prompt versions in a single comparison.

## Testing scope of this release

The local cleanup environment used CPU PyTorch, not the original L4 runtime. See
`cleanup_validation.json` for the exact environment and completed checks. Source AST
comparisons remove only logging statements/imports and docstrings, and compare against
an explicitly translated copy of the original, leaving numerical statements intact.
Synthetic regression checks compare pre- and post-cleanup outputs exactly (including
NaN placement in zero-reference coverage diagnostics). Dataset inputs are byte-preserved.

The new-run notebook was syntax/structure checked, not executed end-to-end on a GPU.
The full research pipeline still requires a compatible GPU environment and data/model
access. Missing checkpoints or hash mismatches must be investigated, never bypassed.
