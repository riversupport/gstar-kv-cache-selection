"""GSTAR: DEVELOPMENT-ONLY paired evidence-shift pilot.

Uses the already-loaded model/tokenizer and the user's validated gstar_mask_smoke.py.
No installation, downloads, training, physical KV removal, or held-out evaluation.
Target runtime: Qwen2ForCausalLM, Transformers 4.56.x, unquantized BF16, batch size 1.

The scorer intentionally preserves the existing reconstruction-attention PROXY:
full causal replay; max over grouped query heads and reconstruction query positions.
It is NOT a native KVzip reproduction. All experiments use eager attention.

Notebook:
    import runpy
    p = runpy.run_path('/mnt/gstar-project/gstar_paired_pilot.py')
    summary = p['run_paired_pilot'](model, tokenizer, '/mnt/gstar-project')

API references (pinned to the user's runtime):
https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/models/qwen2/modeling_qwen2.py
https://huggingface.co/docs/transformers/v4.56.0/en/attention_interface
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import runpy
import shutil
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import logging

logger = logging.getLogger("gstar")

VERSION = '1.0'
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RHOS = (0.8, 0.9, 0.95, 1.0)
RESERVES = (0.0, 0.25, 0.5, 0.75, 1.0)
RECONSTRUCTION_INSTRUCTION = 'Repeat the previous context exactly.'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def append_json(handle: Any, obj: Any) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + '\n')
    handle.flush()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_key(mask: torch.Tensor | None) -> str:
    if mask is None:
        return 'full_kv'
    x = mask.detach().cpu().contiguous()
    return hashlib.sha256(str(tuple(x.shape)).encode() + x.numpy().tobytes()).hexdigest()


def fixed_name(lam: float) -> str:
    if lam == 0:
        return 'a_only'
    if lam == 1:
        return 'g_proxy_only'
    return f'fixed_lambda_{lam:.2f}'


def build_policy_bank(a_raw: torch.Tensor, g_raw: torch.Tensor,
                      doc_positions: list[int], prefix_length: int,
                      keep_fraction: float, smoke: dict) -> dict[str, Any]:
    """Pure selection: receives NO answers, target question, or evidence labels.

    P: identical protected positions, counted in the total prefix budget.
    b: budget minus |P|. Normalize a and g over UNPROTECTED document tokens.
    R_lambda = top-b((1-lambda)*a + lambda*g).
    For each layer/KV-head, maximize A(R_lambda) subject to G >= rho*G(R_1).
    Fixed lambda, rho and reserve fraction are global candidate hyperparameters.
    """
    endpoints, budget = smoke['make_masks'](
        a_raw, g_raw, doc_positions, prefix_length, keep_fraction)
    protected = torch.zeros(prefix_length, dtype=torch.bool)
    protected[budget['protected_prefix_positions']] = True
    doc_idx = torch.tensor(doc_positions, dtype=torch.long)
    free_doc = ~protected[doc_idx]
    free_prefix = doc_idx[free_doc]
    b = budget['free_slots_per_layer_kv_head']
    a = a_raw.detach().float().cpu()[..., free_doc]
    g = g_raw.detach().float().cpu()[..., free_doc]
    a = a / a.sum(-1, keepdim=True)
    g = g / g.sum(-1, keepdim=True)
    require(bool(torch.isfinite(a).all() and torch.isfinite(g).all()), 'Invalid normalized scores.')
    layers, heads, n = a.shape

    def top_b(values: torch.Tensor) -> torch.Tensor:
        ranked = torch.argsort(values, dim=-1, descending=True, stable=True)
        chosen = torch.zeros_like(values, dtype=torch.bool)
        chosen.scatter_(-1, ranked[..., :b], True)
        return chosen

    def expand(free: torch.Tensor) -> torch.Tensor:
        keep = protected.view(1, 1, -1).expand(layers, heads, -1).clone()
        keep[..., free_prefix] = free
        require(bool((keep.sum(-1) == budget['kept_prefix_tokens_per_layer_kv_head']).all()),
                'A method violated the per-layer/head budget.')
        require(bool(keep[..., protected].all()), 'A protected token was excluded.')
        return keep

    candidates = torch.stack([top_b((1 - lam) * a + lam * g) for lam in LAMBDAS])
    require(torch.equal(expand(candidates[0]), endpoints['a_only']), 'a endpoint regression failed.')
    require(torch.equal(expand(candidates[-1]), endpoints['g_proxy_only']), 'g endpoint regression failed.')
    # Double precision only for sums/comparisons, not score extraction or model execution.
    A = (candidates.double() * a.double()).sum(-1)
    G = (candidates.double() * g.double()).sum(-1)
    G_best = G[-1]
    require(bool((G_best > 0).all()), 'Nonpositive g-only retained mass.')

    policies: dict[str, Any] = {
        'full_kv': {'family': 'full', 'parameter': None, 'keep': None,
                    'mask_key': 'full_kv', 'diagnostics': None, 'per_head': None}}

    def add(name: str, family: str, parameter: float, free: torch.Tensor,
            selected_index: torch.Tensor | None = None,
            feasible: torch.Tensor | None = None) -> None:
        keep = expand(free)
        a_mass = (free.double() * a.double()).sum(-1)
        g_mass = (free.double() * g.double()).sum(-1)
        retention = g_mass / G_best
        same_g = (free == candidates[-1]).all(-1)
        require(bool((retention <= 1 + 1e-10).all()), 'g-only was not the g-mass optimum.')
        diag = {
            'a_mass_mean': float(a_mass.mean()),
            'g_mass_mean': float(g_mass.mean()),
            'g_retention_mean': float(retention.mean()),
            'g_retention_min': float(retention.min()),
            'g_only_identical_head_fraction': float(same_g.double().mean()),
            'whole_cache_equals_g_only': bool(same_g.all()),
            'head_count': layers * heads,
            'lambda_histogram': None,
            'lambda_one_head_fraction': None,
            'only_g_grid_candidate_feasible_head_fraction': None,
        }
        detail = {'a_mass': a_mass, 'g_mass': g_mass, 'g_retention': retention,
                  'same_set_as_g_only': same_g}
        if selected_index is not None:
            counts = torch.bincount(selected_index.flatten(), minlength=len(LAMBDAS))
            diag['lambda_histogram'] = {str(lam): int(counts[i]) for i, lam in enumerate(LAMBDAS)}
            diag['lambda_one_head_fraction'] = float((selected_index == len(LAMBDAS)-1).double().mean())
            diag['only_g_grid_candidate_feasible_head_fraction'] = float((~feasible[:-1].any(0)).double().mean())
            detail['selected_lambda_index'] = selected_index
            detail['feasible_candidate_count'] = feasible.sum(0)
            require(float(retention.min()) + 1e-12 >= parameter, 'Constraint violation.')
        policies[name] = {'family': family, 'parameter': parameter, 'keep': keep,
                          'mask_key': mask_key(keep), 'diagnostics': diag, 'per_head': detail}

    for i, lam in enumerate(LAMBDAS):
        add(fixed_name(lam), 'fixed', lam, candidates[i])

    for beta in RESERVES:
        m = math.floor(beta * b)  # reserve a fraction of FREE slots, not protected slots
        reserve = torch.zeros_like(a, dtype=torch.bool)
        if m:
            order_g = torch.argsort(g, dim=-1, descending=True, stable=True)
            reserve.scatter_(-1, order_g[..., :m], True)
        if b - m:
            order_a = torch.argsort(a.masked_fill(reserve, float('-inf')),
                                   dim=-1, descending=True, stable=True)
            reserve.scatter_(-1, order_a[..., :b-m], True)
        add(f'reserve_beta_{beta:.2f}', 'reserve', beta, reserve)
    require(torch.equal(policies['reserve_beta_0.00']['keep'], endpoints['a_only']), 'Reserve beta=0 mismatch.')
    require(torch.equal(policies['reserve_beta_1.00']['keep'], endpoints['g_proxy_only']), 'Reserve beta=1 mismatch.')

    li = torch.arange(layers)[:, None]
    hi = torch.arange(heads)[None, :]
    for rho in RHOS:
        feasible = G >= rho * G_best.unsqueeze(0)
        require(bool(feasible[-1].all()), 'g-only must always be feasible.')
        # argmax returns first maximum: smaller lambda breaks an exact A tie.
        chosen = A.masked_fill(~feasible, float('-inf')).argmax(dim=0)
        selected = candidates[chosen, li, hi]
        add(f'constrained_rho_{rho:.2f}', 'constrained', rho, selected, chosen, feasible)

    return {
        'policies': policies, 'budget': budget,
        'normalized_a_free': a, 'normalized_g_free': g,
        'free_prefix_positions': free_prefix,
        'candidate_A': A, 'candidate_G': G,
        'lambda_grid': list(LAMBDAS),
        'selection_tie_break': 'smaller token position; then smaller lambda for exact A ties',
    }


def tokenized(tokenizer: Any, text: str, offsets: bool = False) -> dict:
    return tokenizer(text, add_special_tokens=False, return_tensors='pt',
                     **({'return_offsets_mapping': True} if offsets else {}))


def overlap_positions(offsets: list, start: int, end: int) -> list[int]:
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]


def make_layout(tokenizer: Any, render: Any, document: str) -> dict:
    marker = '__GSTAR_QUESTION_SLOT__'
    marked = render(tokenizer, document, marker)
    require(marked.count(marker) == 1, 'Question placeholder is not unique.')
    prefix_text = marked.split(marker)[0]
    require(prefix_text.count(document) == 1, 'Document substring is not unique.')
    encoded = tokenized(tokenizer, prefix_text, offsets=True)
    offsets = encoded['offset_mapping'][0].tolist()
    d0 = prefix_text.index(document)
    d1 = d0 + len(document)
    positions = overlap_positions(offsets, d0, d1)
    require(bool(positions), 'No document tokens.')
    return {'document': document, 'prefix_text': prefix_text,
            'prefix_ids': encoded['input_ids'], 'prefix_length': encoded['input_ids'].shape[1],
            'document_positions': positions, 'document_token_ids': encoded['input_ids'][0, positions].tolist(),
            'document_character_range': [d0, d1]}


@torch.no_grad()
def collect_scores(model: Any, ids: torch.Tensor, doc_positions: list[int],
                   query_positions: list[int], reduction: str, max_tokens: int) -> torch.Tensor:
    """Same eager full-forward definitions as the successful a and g probe cells."""
    require(ids.shape[0] == 1 and ids.shape[1] <= max_tokens,
            f'Score input length {ids.shape[1]} exceeds diagnostic limit {max_tokens}. No truncation.')
    require(model.config._attn_implementation == 'eager', 'Scoring requires eager.')
    require(query_positions and doc_positions and max(doc_positions) < min(query_positions),
            'Query/document ranges overlap or are reversed.')
    hq, hkv = model.config.num_attention_heads, model.config.num_key_value_heads
    require(hq % hkv == 0, 'Invalid GQA grouping.')
    ids = ids.to(model.device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   output_attentions=True, logits_to_keep=1, return_dict=True)
    require(output.attentions is not None and len(output.attentions) == model.config.num_hidden_layers,
            'Missing attention weights.')
    qi = torch.tensor(query_positions, device=model.device)
    di = torch.tensor(doc_positions, device=model.device)
    layers = []
    for weights in output.attentions:
        require(weights is not None, 'A layer did not return attention weights.')
        block = weights[0].index_select(1, qi).index_select(2, di).float()
        grouped = block.reshape(hkv, hq // hkv, len(query_positions), len(doc_positions))
        if reduction == 'mean':
            reduced = grouped.mean(dim=(1, 2))
        elif reduction == 'max':
            reduced = grouped.amax(dim=(1, 2))
        else:
            raise ValueError(reduction)
        layers.append(reduced.cpu())
    scores = torch.stack(layers)
    require(bool(torch.isfinite(scores).all() and (scores >= 0).all()), 'Invalid raw scores.')
    require(bool((scores.sum(-1) > 0).all()), 'Zero document mass in one layer/head.')
    return scores


def extract_a(model: Any, tokenizer: Any, render: Any, layout: dict,
              first_question: str, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    prompt = render(tokenizer, layout['document'], first_question)
    require(prompt.startswith(layout['prefix_text']), 'First-question prefix changed.')
    enc = tokenized(tokenizer, prompt, offsets=True)
    ids, offsets = enc['input_ids'], enc['offset_mapping'][0].tolist()
    p = layout['prefix_length']
    require(torch.equal(ids[:, :p], layout['prefix_ids']), 'First-question token prefix changed.')
    doc_positions = overlap_positions(offsets, *layout['document_character_range'])
    require(doc_positions == layout['document_positions'], 'Document token offset mapping changed.')
    q0 = len(layout['prefix_text'])
    qpositions = overlap_positions(offsets, q0, q0 + len(first_question))
    scores = collect_scores(model, ids, doc_positions, qpositions, 'mean', max_tokens)
    return scores, ids[:, p:], qpositions


def extract_g(model: Any, tokenizer: Any, render: Any, layout: dict,
              max_tokens: int) -> tuple[torch.Tensor, dict]:
    request = render(tokenizer, layout['document'], RECONSTRUCTION_INSTRUCTION)
    request_ids = tokenized(tokenizer, request)['input_ids']
    p = layout['prefix_length']
    require(torch.equal(request_ids[:, :p], layout['prefix_ids']), 'Reconstruction prefix changed.')
    replay = torch.tensor([layout['document_token_ids']], dtype=torch.long)
    ids = torch.cat((request_ids, replay), dim=1)
    query_positions = list(range(p, ids.shape[1]))
    scores = collect_scores(model, ids, layout['document_positions'], query_positions, 'max', max_tokens)
    meta = {'definition': 'single_chunk_full_causal_reconstruction_max',
            'native_kvzip_reproduction': False, 'query_start': p,
            'replay_start': request_ids.shape[1], 'total_tokens': ids.shape[1],
            'query_tokens': len(query_positions), 'instruction': RECONSTRUCTION_INSTRUCTION}
    return scores, meta


def _resolve_run_path(recorded: str, results_dir: Path) -> Path:
    p = Path(recorded)
    if p.is_file():
        return p
    fallback = results_dir / p.parent.name / p.name
    require(fallback.is_file(), f'Recorded file unavailable: {p}')
    return fallback


def discover_source(project: Path, source_smoke_dir: str | Path | None) -> tuple[Path, Path, dict]:
    if source_smoke_dir is None:
        passed = []
        for p in sorted((project / 'results').glob('mask_smoke_*/summary.json')):
            if json.loads(p.read_text(encoding='utf-8')).get('status') == 'PASS':
                passed.append(p.parent)
        require(bool(passed), 'No successful mask_smoke run found. Keep its saved results.')
        source = passed[-1]
    else:
        source = Path(source_smoke_dir)
    meta = json.loads((source / 'environment.json').read_text(encoding='utf-8'))
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    require(summary.get('status') == 'PASS', 'Source mask smoke did not pass.')
    env_path = _resolve_run_path(meta['source_environment'], project / 'results')
    old_env = json.loads(env_path.read_text(encoding='utf-8'))
    candidates = [env_path.parent / 'data_snapshot', Path(meta['source_data_directory']),
                  Path(old_env['data_directory'])]
    required = ('selection_inputs.jsonl', 'evaluation_only.jsonl', 'source_split.json')
    data = next((d for d in candidates if all((d / name).is_file() for name in required)), None)
    require(data is not None, 'The exact pilot data used by mask_smoke could not be located.')
    # Check against original hashes if available. Never silently change dataset/split.
    for name in required:
        old_sha = old_env.get('data_sha256', {}).get(name)
        require(not old_sha or old_sha == sha256(data / name), f'Data changed since Full-KV run: {name}')
    return source, data, meta


def aggregate(rows: list[dict], diag_rows: list[dict], banks: dict[str, dict]) -> dict:
    by_method: dict[str, list] = defaultdict(list)
    for r in rows:
        by_method[r['method']].append(r)
    methods = {}
    for method, items in by_method.items():
        groups = {
            'first': [r for r in items if r['role'] == 'first'],
            'same': [r for r in items if r['role'] == 'target' and r['condition'] == 'same'],
            'different': [r for r in items if r['role'] == 'target' and r['condition'] == 'different'],
        }
        require(all(groups.values()), 'Incomplete paired result groups.')
        stats = {name: {'n': len(rs), 'f1': sum(r['f1'] for r in rs)/len(rs),
                        'em': sum(r['em'] for r in rs)/len(rs)} for name, rs in groups.items()}
        stats['worst_target_f1'] = min(stats['same']['f1'], stats['different']['f1'])
        stats['same_minus_different_f1'] = stats['same']['f1'] - stats['different']['f1']
        stats['family'] = items[0]['family']
        stats['parameter'] = items[0]['parameter']
        methods[method] = stats

    diagnostics = {}
    for method in methods:
        ds = [r for r in diag_rows if r['method'] == method and r['diagnostics'] is not None]
        if not ds:
            continue
        count = sum(r['diagnostics']['head_count'] for r in ds)
        totals = Counter()
        for r in ds:
            totals.update(r['diagnostics']['lambda_histogram'] or {})
        def weighted(field: str):
            vals = [(r['diagnostics'][field], r['diagnostics']['head_count']) for r in ds]
            return sum(v * n for v, n in vals)/count if all(v is not None for v, n in vals) else None
        diagnostics[method] = {
            'layer_head_decisions': count, 'lambda_histogram': dict(totals) or None,
            'g_retention_mean': weighted('g_retention_mean'),
            'g_retention_min': min(r['diagnostics']['g_retention_min'] for r in ds),
            'g_only_identical_head_fraction': weighted('g_only_identical_head_fraction'),
            'whole_cache_g_only_cases': sum(r['diagnostics']['whole_cache_equals_g_only'] for r in ds),
            'cases': len(ds), 'lambda_one_head_fraction': weighted('lambda_one_head_fraction'),
            'only_g_grid_candidate_feasible_head_fraction': weighted('only_g_grid_candidate_feasible_head_fraction'),
        }

    winners = {}
    for family in ('fixed', 'reserve', 'constrained'):
        names = [name for name, x in methods.items() if x['family'] == family]
        # Grid ordering is deterministic; ties do NOT inspect held-out results.
        winners[family] = max(names, key=lambda name: (
            methods[name]['worst_target_f1'], methods[name]['first']['f1']))

    pair_rows = []
    doc_ids = sorted({r['document_id'] for r in rows})
    for doc in doc_ids:
        for method in methods:
            rs = [r for r in rows if r['document_id'] == doc and r['method'] == method and r['role'] == 'target']
            require(len(rs) == 2, 'Missing target pair.')
            rs = sorted(rs, key=lambda r: r['condition'] != 'same')
            a, b = rs
            require(a['question_id'] == b['question_id'] and a['question'] == b['question'], 'Target changed.')
            p1, p2 = banks[a['case_id']]['policies'][method], banks[b['case_id']]['policies'][method]
            mask_equal = p1['mask_key'] == p2['mask_key']
            pair_rows.append({'document_id': doc, 'method': method,
                              'same_f1': a['f1'], 'different_f1': b['f1'],
                              'same_minus_different_f1': a['f1'] - b['f1'],
                              'selected_masks_identical': mask_equal,
                              'target_tokens_identical': a['generated_token_ids'] == b['generated_token_ids']})
    invariants = [p for p in pair_rows if p['method'] in ('full_kv', 'g_proxy_only')]
    require(all(p['selected_masks_identical'] and p['target_tokens_identical'] for p in invariants),
            'Full-KV/g-only paired target invariant failed; inspect predictions before drawing conclusions.')
    return {'methods': methods, 'diagnostics': diagnostics, 'paired_targets': pair_rows,
            'pilot_selected_parameters_NOT_heldout': winners,
            'tuning_objective': 'max min(mean same-target F1, mean different-target F1); tie: mean first F1; then grid order',
            'invariant_pairs_passed': len(invariants)}


def print_summary(summary: dict) -> None:
    logger.debug('%s', '\n========== PAIRED PILOT SUMMARY ==========')
    logger.debug('%s', f"Documents: {summary['documents']} | Paired cases: {summary['paired_cases']} | "
          f"Unique questions: {summary['unique_questions']}")
    logger.debug('%s', 'DEVELOPMENT PILOT ONLY: tuning and reporting use the SAME tiny pilot.')
    logger.debug('%s', 'These are NOT held-out estimates or evidence of a general improvement.')
    logger.debug('%s', f"{'Method':25s} {'First F1':>9s} {'Same F1':>9s} {'Diff F1':>9s} {'Worst':>8s}")
    for name, m in summary['methods'].items():
        logger.debug('%s', f"{name:25s} {m['first']['f1']:9.2f} {m['same']['f1']:9.2f} "
              f"{m['different']['f1']:9.2f} {m['worst_target_f1']:8.2f}")
    logger.debug('%s %s', '\nPilot-selected settings (NOT final hyperparameters):', summary['pilot_selected_parameters_NOT_heldout'])
    logger.debug('%s', '\nConstraint diagnostics:')
    for rho in RHOS:
        name = f'constrained_rho_{rho:.2f}'
        d = summary['diagnostics'][name]
        logger.debug('%s', f"  rho={rho:.2f}: retention min/mean={d['g_retention_min']:.4f}/{d['g_retention_mean']:.4f}, "
              f"g-only identical heads={d['g_only_identical_head_fraction']:.1%}")
        logger.debug('%s %s', '    lambda counts:', d['lambda_histogram'])
    logger.debug('%s %s', 'Full-KV + g-only same/different invariant pairs:', summary['invariant_pairs_passed'])
    logger.debug('%s %s', 'Actual generation calls:', summary['actual_generation_calls'])
    logger.debug('%s %s', 'Generation calls avoided by identical mask/input:', summary['deduplicated_generation_records'])
    logger.debug('%s %s', 'Token-limit hits (actual calls):', summary['actual_generation_token_limit_hits'])
    logger.debug('%s %s', 'Original cache unchanged:', summary['original_kv_unchanged'])
    logger.debug('%s %s', 'Hooks restored:', summary['hook_counts_restored'])
    logger.debug('%s %s', 'Attention backend restored:', summary['backend_after'])
    logger.info('%s %s', 'Results saved:', summary['output_directory'])
    logger.debug('%s', 'MASKING ONLY; reconstruction proxy, NOT native KVzip; no memory/speed claim.')


def run_paired_pilot(model: Any, tokenizer: Any,
                     project_dir: str | Path = '/mnt/gstar-project',
                     max_documents: int = 3, keep_fraction: float = 0.5,
                     max_new_tokens: int = 64,
                     source_smoke_dir: str | Path | None = None) -> dict[str, Any]:
    """Run 3 documents x 2 first-question conditions x 15 policy settings x 2 questions.

    Up to 180 answer records. Exact mask/input duplicates WITHIN a case reuse an
    already computed result. We do NOT deduplicate across same/different cases,
    so query-independent target invariants are checked by independent executions.
    This is a correctness/diagnostic pilot, not the final held-out experiment.
    """
    import transformers
    require(transformers.__version__.startswith('4.56.'), 'Expected Transformers 4.56.x; do not reinstall automatically.')
    require(model.config.model_type == 'qwen2' and not getattr(model, 'is_quantized', False),
            'Use the existing unquantized Qwen2.5 model.')
    require(tokenizer.is_fast, 'A fast tokenizer is needed for character offsets.')
    require(1 <= max_documents <= 3, 'This runner is limited to the original 3-document pilot.')
    require(1 <= max_new_tokens <= 128, 'max_new_tokens must be in [1, 128].')
    project = Path(project_dir).expanduser().resolve()
    require(project.is_dir(), f'Missing Volume: {project}')
    smoke_path = project / 'gstar_mask_smoke.py'
    full_path = project / 'gstar_fullkv_pilot' / 'gstar_run_fullkv.py'
    require(smoke_path.is_file() and full_path.is_file(), 'Keep the existing smoke and Full-KV runner files.')
    smoke = runpy.run_path(str(smoke_path))
    full = runpy.run_path(str(full_path))
    source, data, source_meta = discover_source(project, source_smoke_dir)
    require(source_meta['model_id'] == model.config._name_or_path, 'Model changed since mask_smoke.')
    current_rev = getattr(model.config, '_commit_hash', None)
    old_rev = source_meta.get('model_revision')
    require(not (current_rev and old_rev) or current_rev == old_rev, 'Model revision changed.')
    require(source_meta.get('fullkv_runner_sha256') == sha256(full_path), 'Full-KV prompt/scorer runner changed.')
    old_smoke = source / 'runner_snapshot.py'
    require(not old_smoke.is_file() or sha256(old_smoke) == sha256(smoke_path), 'Validated smoke code changed.')
    require(source_meta.get('g_definition') == 'single_chunk_full_causal_reconstruction_max', 'g-proxy definition changed.')
    source_budget = source_meta['budget']
    require(source_budget.get('recent_document_tokens_config') == 16 and source_budget.get('sink_prefix_tokens_config') == 4,
            'Protection rule differs from the successful smoke test.')
    require(float(source_budget['keep_fraction_requested']) == float(keep_fraction), 'Keep the validated budget for this pilot.')
    require(source_meta['max_new_tokens'] == max_new_tokens, 'Keep the validated generation cap for this pilot.')

    # load_groups audits article split/answer offsets. These labels are never
    # passed into extract_a/extract_g/build_policy_bank; only the evaluator uses them.
    groups, manifest = full['load_groups'](data)
    groups = groups[:max_documents]
    require(len(groups) == max_documents, 'Not enough development documents.')
    render = full['render_prompt']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = project / 'results' / f'paired_pilot_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'data_snapshot').mkdir()
    for name in ('selection_inputs.jsonl', 'evaluation_only.jsonl', 'source_split.json', 'preview.txt'):
        if (data / name).is_file():
            shutil.copy2(data / name, out / 'data_snapshot' / name)
    shutil.copy2(__file__, out / 'runner_snapshot.py')
    shutil.copy2(smoke_path, out / 'mask_smoke_snapshot.py')
    shutil.copy2(full_path, out / 'fullkv_runner_snapshot.py')
    metadata = {
        'runner_version': VERSION, 'purpose': 'DEVELOPMENT paired pilot, NOT held-out evaluation',
        'model_id': model.config._name_or_path, 'model_revision': current_rev,
        'model_dtype': str(model.dtype), 'torch_version': str(torch.__version__),
        'transformers_version': transformers.__version__, 'device': str(model.device),
        'source_smoke_directory': str(source), 'source_data_directory': str(data),
        'document_ids': [x['document_id'] for x in groups], 'system_prompt': full['SYSTEM_PROMPT'],
        'keep_fraction': keep_fraction, 'lambda_grid': list(LAMBDAS),
        'rho_grid': list(RHOS), 'reserve_fraction_grid': list(RESERVES),
        'max_new_tokens': max_new_tokens, 'max_scoring_input_tokens': 1024,
        'score_definitions': {'a': 'mean over first-question tokens and grouped query heads',
                              'g_proxy': 'single_chunk_full_causal_reconstruction_max'},
        'native_kvzip_reproduction': False, 'physical_compaction': False,
        'hardware_savings_measured': False, 'attention_backend_all_methods': 'eager',
        'normalization': 'per layer/KV-head over unprotected document positions only',
        'budget': 'all persistent prefix positions, protected included; question/answer states temporary',
        'score_extraction_note': 'Full causal forward for each a and g probe; answers reuse a pristine prefix cache.',
        'deduplication': 'identical mask + suffix token IDs WITHIN each case only',
        'fixed_lambda_scope': 'one global hyperparameter; never individually optimized with target labels',
        'rho_scope': 'one global hyperparameter; feasible candidate selected per layer/KV head and case',
        'protected_tokens': 'all non-document prefix + first 4 prefix + last 16 document tokens',
        'data_sha256': {p.name: sha256(p) for p in (out/'data_snapshot').iterdir()},
        'smoke_sha256': sha256(smoke_path), 'fullkv_sha256': sha256(full_path),
    }
    write_json(out / 'environment.json', metadata)
    logger.debug('%s', '========== PAIRED DEVELOPMENT PILOT ==========')
    logger.debug('%s', f'Documents: {len(groups)} | paired cases: {2*len(groups)} | policy settings: 15')
    logger.debug('%s', f'At most {60*len(groups)} answer records; exact within-case duplicates will be reused.')
    logger.debug('%s %s', 'Output:', out)
    old_backend, old_training = model.config._attn_implementation, model.training
    old_hooks = smoke['hook_counts'](model)
    rows: list[dict] = []
    diag_rows: list[dict] = []
    all_banks: dict[str, dict] = {}
    actual_calls = 0
    actual_limit_hits = 0
    summary: dict = {}
    started = time.perf_counter()
    try:
        model.eval()
        model.set_attn_implementation('eager')
        with (out / 'predictions.jsonl').open('x', encoding='utf-8') as pred_log, \
             (out / 'diagnostics.jsonl').open('x', encoding='utf-8') as diag_log, \
             (out / 'audits.jsonl').open('x', encoding='utf-8') as audit_log:
            for di, group in enumerate(groups, 1):
                doc_id, document = group['document_id'], group['document']
                logger.debug('%s', f'\n[{di}/{len(groups)}] {doc_id}: extracting one g_proxy and two a scores ...')
                layout = make_layout(tokenizer, render, document)
                g, g_meta = extract_g(model, tokenizer, render, layout, 1024)
                case_suffixes = {}
                case_as = {}
                # Build BOTH selections BEFORE any evaluation query is forwarded.
                for selection, _label in group['cases']:
                    cid = selection['case_id']
                    a, first_suffix, qpos = extract_a(model, tokenizer, render, layout, selection['first_question'], 1024)
                    bank = build_policy_bank(a, g, layout['document_positions'], layout['prefix_length'], keep_fraction, smoke)
                    all_banks[cid] = bank
                    case_suffixes[cid] = first_suffix
                    case_as[cid] = {'a_raw': a, 'question_positions': qpos,
                                    'first_question': selection['first_question']}
                    torch.save({'case_id': cid, 'document_id': doc_id, **bank}, out / f'{cid}_selection.pt')
                torch.save({'document_id': doc_id, 'layout': layout, 'g_raw_proxy': g,
                            'g_metadata': g_meta, 'cases': case_as}, out / f'{doc_id}_scores.pt')
                p = layout['prefix_length']
                budget = all_banks[group['cases'][0][0]['case_id']]['budget']
                logger.debug('%s', f"  prefix={p}, protected={budget['protected_tokens']}, "
                      f"kept={budget['kept_prefix_tokens_per_layer_kv_head']}, g shape={tuple(g.shape)}")
                with torch.no_grad():
                    ids = layout['prefix_ids'].to(model.device)
                    pref = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                                 logits_to_keep=1, return_dict=True)
                base = pref.past_key_values
                del pref
                snapshot = smoke['prefix_cache_snapshot'](base)
                require(int(base.get_seq_length()) == p, 'Prefix cache length mismatch.')
                first_cid = group['cases'][0][0]['case_id']
                suffix0 = case_suffixes[first_cid].to(model.device)
                trace0, next_token, _ = smoke['fixed_trace'](model, suffix0, base, None)
                all_keep = torch.ones_like(all_banks[first_cid]['policies']['a_only']['keep'])
                trace1, _, all_audit = smoke['fixed_trace'](model, suffix0, base, all_keep, next_token)
                comparison = smoke['compare_traces'](trace0, trace1)
                require(comparison['exactly_equal'], 'All-keep equivalence failed.')
                append_json(audit_log, {'document_id': doc_id, 'test': 'all_keep',
                                        'comparison': comparison, 'audit': all_audit})

                for selection, label in group['cases']:
                    cid = selection['case_id']
                    bank = all_banks[cid]
                    policies = bank['policies']
                    condition = label['condition']
                    logger.debug('%s', f'  {cid} / {condition}: mask audits then answer generation ...')
                    # Validate a representative adaptive mask, including forced continuation,
                    # even when a model could immediately emit EOS during normal generation.
                    test_keep = policies['constrained_rho_0.90']['keep']
                    first_suffix = case_suffixes[cid].to(model.device)
                    clean, cont, clean_audit = smoke['fixed_trace'](model, first_suffix, base, test_keep)
                    poisoned = smoke['poison_excluded'](base, test_keep)
                    changed, _, changed_audit = smoke['fixed_trace'](model, first_suffix, poisoned, test_keep, cont)
                    del poisoned
                    comp = smoke['compare_traces'](clean, changed)
                    require(comp['exactly_equal'], 'Excluded-KV perturbation changed outputs.')
                    append_json(audit_log, {'case_id': cid, 'test': 'constrained_rho_0.90_perturbation',
                                            'comparison': comp, 'clean_audit': clean_audit,
                                            'perturbed_audit': changed_audit})
                    require(smoke['cache_intact'](base, snapshot, p), 'Scoring/audits modified the original KV.')

                    references = [('first', label['first_question_reference']),
                                  ('target', label['target_question_reference'])]
                    encoded_questions = []
                    for role, ref in references:
                        ids = tokenized(tokenizer, render(tokenizer, document, ref['question']))['input_ids']
                        require(torch.equal(ids[:, :p], layout['prefix_ids']), 'Evaluation prefix tokenization changed.')
                        require(p < ids.shape[1] <= 1152, 'Unexpected evaluation input length; no truncation.')
                        encoded_questions.append((role, ref, ids[:, p:].to(model.device)))
                    memo: dict[tuple, tuple[str, dict]] = {}  # cleared between paired cases
                    for method, policy in policies.items():
                        drow = {'case_id': cid, 'document_id': doc_id, 'condition': condition,
                                'method': method, 'family': policy['family'], 'parameter': policy['parameter'],
                                'mask_key': policy['mask_key'], 'diagnostics': policy['diagnostics']}
                        append_json(diag_log, drow)
                        diag_rows.append(drow)
                        result_scores = {}
                        for role, ref, suffix in encoded_questions:
                            memo_key = (policy['mask_key'], tuple(suffix.detach().cpu().flatten().tolist()))
                            row_id = f'{cid}/{method}/{role}'
                            reused_from = None
                            if memo_key in memo:
                                reused_from, result = memo[memo_key]
                            else:
                                result = smoke['generate_answer'](model, tokenizer, suffix, base,
                                                                  policy['keep'], max_new_tokens)
                                memo[memo_key] = (row_id, result)
                                actual_calls += 1
                                actual_limit_hits += int(result['hit_token_limit'])
                            intact = smoke['cache_intact'](base, snapshot, p)
                            require(intact, 'Original KV was modified during answer generation.')
                            gold = [a['text'] for a in ref['answers']]
                            metrics = full['score_answer'](result['answer'], gold)
                            row = {'record_id': row_id, 'document_id': doc_id, 'case_id': cid,
                                   'project_split': 'development', 'condition': condition,
                                   'method': method, 'family': policy['family'], 'parameter': policy['parameter'],
                                   'role': role, 'question_id': ref['question_id'], 'question': ref['question'],
                                   'references': gold, 'em': metrics['em'], 'f1': metrics['f1'],
                                   'prefix_tokens': p,
                                   'kept_prefix_tokens': p if policy['keep'] is None else budget['kept_prefix_tokens_per_layer_kv_head'],
                                   'mask_key': policy['mask_key'], 'original_kv_unchanged': intact,
                                   'reused_from_record': reused_from, **result}
                            append_json(pred_log, row)
                            rows.append(row)
                            result_scores[role] = metrics['f1']
                        logger.debug('%s', f"    {method:25s} first F1={result_scores['first']:6.2f} "
                              f"target F1={result_scores['target']:6.2f} | records {len(rows)}/{60*len(groups)}")
                del base, snapshot, case_as

        stats = aggregate(rows, diag_rows, all_banks)
        summary = {
            'status': 'PASS', 'documents': len(groups), 'paired_cases': 2*len(groups),
            'unique_questions': len({(r['document_id'], r['question_id']) for r in rows}),
            'policy_settings': 15, 'answer_records': len(rows), 'actual_generation_calls': actual_calls,
            'deduplicated_generation_records': len(rows)-actual_calls,
            'actual_generation_token_limit_hits': actual_limit_hits,
            'original_kv_unchanged': all(r['original_kv_unchanged'] for r in rows),
            'wall_seconds_including_scoring_and_audits': time.perf_counter()-started,
            'native_kvzip_reproduction': False, 'physical_compaction': False,
            'heldout_evaluation': False, 'output_directory': str(out), **stats,
        }
        write_json(out / 'paired_targets.json', stats['paired_targets'])
        table = []
        for method, m in stats['methods'].items():
            table.append({'method': method, 'family': m['family'], 'parameter': m['parameter'],
                          'first_f1': m['first']['f1'], 'same_f1': m['same']['f1'],
                          'different_f1': m['different']['f1'], 'worst_target_f1': m['worst_target_f1']})
        with (out/'comparison.csv').open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    except BaseException as exc:
        write_json(out/'failure.json', {'status': 'FAILED', 'error': str(exc),
                                       'traceback': traceback.format_exc(), 'answer_records_saved': len(rows)})
        logger.error('%s', f'Run stopped. Previously saved results are preserved: {out}')
        raise
    finally:
        model.set_attn_implementation(old_backend)
        model.train(old_training)
        restoration = {'attention_backend_restored': model.config._attn_implementation == old_backend,
                       'backend_after': model.config._attn_implementation,
                       'hook_counts_restored': smoke['hook_counts'](model) == old_hooks}
        write_json(out/'restoration.json', restoration)
        if summary:
            summary.update(restoration)
            require(restoration['attention_backend_restored'] and restoration['hook_counts_restored'],
                    'Model settings were not completely restored.')
            write_json(out/'summary.json', summary)
    print_summary(summary)
    return summary
