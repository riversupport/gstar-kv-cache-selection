# Constrained KV Cache Selection under Evidence Shift

**Jiwon Kang — GStar individual project**

A research prototype for selecting a reusable document KV cache after observing a
first question, then answering later questions that may require different evidence.
The study compares a total reconstruction-score constraint with a four-block
reconstruction-attention coverage constraint.

> This is a **selection-quality study using attention masking**, not a physical
> KV-compaction implementation or a memory/latency benchmark. The reconstruction
> score is a labeled proxy inspired by KVzip, not a native KVzip reproduction.
> The results do **not** establish overall superiority over the baselines.

## Method

For each layer and KV head, protect the same system/sink/recent positions within
an approximately 50% prefix-access budget. On the remaining document positions,
normalize first-question attention `a` and reconstruction-attention proxy `g`.
With `b` free slots, form five candidates:

$$R_\lambda=\operatorname{Top}_b((1-\lambda)a+\lambda g),
\quad\lambda\in\{0,0.25,0.5,0.75,1\}.$$

The total-score selector maximizes `A(R) = sum(a_i)` among those candidates subject
to `G(R) >= rho G(R_g)`. The coverage follow-up instead requires
`M_c(R) >= eta M_c(R_g)` for **each** of four replay-query blocks. The two constraints
are alternatives, not simultaneously enforced. All positions retain their original
indices. A different set may be selected for each layer/KV head.

For paired evaluation, the document and target question are fixed and only the
first question changes between same- and different-passage evidence conditions.
First and target questions use independent copies of the pristine document cache.
Target questions, reference answers, and condition labels are not selector inputs.

## Main observation

In the original study, development tuning selected the g-only endpoint for the
fixed, reserved-slot, and total-score-constrained families. No additional benefit
was obtained from the tuned total-score constraint in that evaluation.

The follow-up fixed coverage at `eta=0.95` before a fresh 20-article test:

| Method | First F1 | Same F1 | Different F1 | Worst F1 |
|---|---:|---:|---:|---:|
| Full KV | 59.14 | 65.46 | 65.46 | 65.46 |
| g-only (= dev-tuned fixed) | 57.80 | 54.08 | 54.08 | 54.08 |
| Additional fixed mixture, lambda=0.75 | 56.74 | 56.58 | 57.09 | 56.58 |
| Coverage, eta=0.95 | 56.58 | 54.83 | 63.67 | 54.83 |

`Worst = min(mean Same F1, mean Different F1)`, not the average of per-document
minima. Full KV uses all prefix positions; the other policies share the same
approximately 50% access budget including protected positions.
Coverage's Different-F1 difference versus the additional fixed mixture is +6.58
points, with a 95% paired article-bootstrap interval of [-4.66, +20.56]. Both
reported Different-F1 comparison intervals include zero. Some gains are sensitive
to wording and ambiguous questions; First and Worst are not jointly improved.
See [reported_results](reported_results/) for the original numeric artifacts.

## Repository map

```text
notebooks/reproduce.ipynb           New-run workflow with explicit test opt-in
notebooks/experiment_log.ipynb      Cleaned historical project cells (not Run All)
gstar_day1_selector.py             Synthetic CPU selector demo
gstar_fullkv_pilot/
  gstar_prepare_pilot.py           Source article split and three-document pilot
  gstar_run_fullkv.py              Full-KV cache-reuse checks
gstar_mask_smoke.py                Per-layer/head masking and invariance audits
gstar_paired_pilot.py              Scores, baseline policies, paired pilot
gstar_expanded_eval.py             Resumable development and original held-out
gstar_extract_coverage.py          Four-block reconstruction attention extraction
gstar_coverage_eval.py             Coverage selector and development comparison
gstar_fresh_coverage_test.py       Frozen fresh-article confirmation
data/pilot/                       Small SQuAD-derived pilot; no model weights
reported_results/                 Selected original result artifacts
tests/                            Lightweight CPU/source checks
docs/                             Release, reproducibility, and presentation notes
```

## Environment

The supplied experimental record targeted Linux/Modal, Python 3.12, PyTorch 2.8.0
(CUDA 12.9), Transformers 4.56.0, and Qwen2.5-3B-Instruct in BF16 on an NVIDIA L4.
Use a separate environment. Match the recorded checkpoint revision for exact-run
reproduction; model downloads are not included.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-dev.txt
```

Choose a compatible PyTorch wheel for your hardware; CUDA installation is not
needed for the synthetic CPU tests. The full runner uses Linux `fcntl` file locks.
See the official [PyTorch previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).
No installation is necessary merely to publish these files on GitHub.

## Quick checks (no GPU or downloads)

```bash
python gstar_day1_selector.py
python gstar_fullkv_pilot/gstar_prepare_pilot.py --self-test
python -m unittest discover -s tests -v
```

The data builder's self-test does not download SQuAD. Its normal command does:

```bash
python gstar_fullkv_pilot/gstar_prepare_pilot.py
```

## Running the experiment

Open `notebooks/reproduce.ipynb` and execute the stages deliberately. It derives new
experiment directories from returned values rather than hardcoded historical IDs:

1. Load the model once; run the Full-KV pilot and score probes.
2. Verify masking; prepare and run the development evaluation.
3. Freeze the chosen settings, then explicitly opt into original held-out evaluation.
4. Extract four-block coverage and evaluate it on development only.
5. Freeze the follow-up and explicitly opt into the fresh-article test.

The fresh-test script is specific to eta=0.95 and the prescribed additional
lambda=0.75 comparator. It rejects a conflicting development selection rather than
silently forcing that value. New experimental designs must be separately versioned.

**Do not overwrite your existing Modal scripts or old experiment snapshots with
this cleaned release.** Even comment/logging edits change SHA256 hashes. Old
results are deliberately not rehashed or modified. Run this release in a separate
project directory with fresh checkpoints. The existing original project remains
the source for resuming its historical runs. See [reproducibility notes](docs/REPRODUCIBILITY.md).

## Output and logging

Debug `print` statements were replaced with the standard-library `logging` API.
Detailed tables and per-call diagnostics are at DEBUG; useful save/checkpoint
messages are at INFO; errors and exceptions remain visible. Functions still return
summaries and save all numerical records to JSON/JSONL/CSV.

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("gstar").setLevel(logging.INFO)
# Use DEBUG for detailed diagnostics, or WARNING for quiet operation.
```

The notebook outputs and execution counts have been cleared. The unrelated
assignment cells from the uploaded scratch notebook are not part of this release.

## Data, provenance, and validation limits

The raw 30 MB SQuAD JSON, downloaded model cache, and generated runs are excluded
and ignored by Git. The small pilot includes attribution in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Three modules missing from the latest uploaded ZIP were restored from earlier
project attachments: the Full-KV runner, mask smoke runner, and fresh-test runner.
The data builder is placed beside the Full-KV runner, as expected by the code.
See [release provenance](docs/RELEASE_NOTES.md) and [cleanup manifest](docs/cleanup_manifest.json).

Cleanup validation checks syntax, non-logging algorithm structure, synthetic
selection equivalence, dataset construction, score aggregation, and EM/F1 behavior.
It does **not** rerun the pretrained Qwen/L4 evaluation or certify bitwise GPU
reproduction. Full experiment checkpoints and source weights are not included.
No project-wide software license was chosen during cleanup; the owner must decide
that separately. The third-party data license is independent.
