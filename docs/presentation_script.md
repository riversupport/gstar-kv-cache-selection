### Slide 1 — Approach

Hi, I’m Jiwon. My project asks whether a cache selected for one question can still answer later questions requiring different evidence.

Using Qwen2.5-3B-Instruct, I combine first-question attention, a, with a reconstruction-attention proxy, g.

I form five mixture candidates. Among candidates satisfying the constraint, I select the one with the highest first-question score.

The constraints preserve either total reconstruction score or attention coverage in each of four reconstruction blocks.

I keep the document and target question fixed, and change only the first question. Questions are answered independently.

Selected-cache methods access roughly half the prefix KV positions. This is masking, not measured memory savings.

### Slide 2 — Results

In the earlier study, the original constraint reproduced g-only after development tuning, adding no benefit.

I then fixed coverage at eta zero point nine five and tested twenty fresh articles.

Compared with the additional fixed mixture at lambda zero point seven five, Different F1 was six point five eight points higher. First F1 was zero point one seven lower, but worst-group F1 was one point seven four lower.

Both ninety-five-percent bootstrap intervals cross zero. Some gains also depend on ambiguous questions.

So, coverage shows a positive average change on shifted questions, but overall superiority is not established.