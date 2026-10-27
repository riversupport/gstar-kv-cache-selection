# Reported results (not rerun during code cleanup)

These two files are copied unchanged from the previously supplied fresh 20-article
review archive (`content (3).zip`). They are not new output from the cleaned code.
The original development, original held-out, and fresh test use different article sets;
do not compare their absolute scores as a causal effect of changing a method.

`comparison.csv` contains the five reported method labels. `fixed_tuned` is the same
g-only policy in this study and is not an additional independent comparison.
`paired_bootstrap.json` contains the prespecified paired article-bootstrap percentile
intervals (10,000 resamples; no multiplicity correction).

Coverage eta=0.95 minus the additional fixed lambda=0.75 comparator:
Different F1 +6.58, First F1 -0.17, Worst F1 -1.74. The Different-F1 95% interval
is approximately [-4.66, +20.56]. These results do not establish overall superiority.
Some score changes are sensitive to answer wording and ambiguous questions.

The original evaluation used attention access masking, not physical KV compaction;
no hardware memory/latency gains were measured. See the project write-up for the
full evidence and limitations. This directory does not include original KV tensors,
model weights, or resumable experiment checkpoints.
