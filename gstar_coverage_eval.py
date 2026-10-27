"""GSTAR: coverage-constrained DEVELOPMENT comparison, using completed V1 runs.

1. prepare_coverage_selectors(source_experiment_dir, coverage_dir): CPU only.
2. run_coverage_development(model, tokenizer, selector_dir, max_new_documents=1).
3. Repeat with max_new_documents=None to finish the existing development set.

No new scoring forward, downloads, training, prompt changes, or held-out access.
Baseline predictions remain immutable. Fresh Full/g-only controls must reproduce
source tokens before archived baselines can be used for this comparison.

Definition for each layer/KV head, on UNPROTECTED document positions:
  R_lambda = top_b((1-lambda)*a + lambda*g), lambda = 0,.25,.5,.75,1.
  M_c(R) = sum(coverage_raw[c,i] for i in R).
  Choose maximum A(R_lambda) subject to M_c(R_lambda)>=eta*M_c(R_g) for ALL c.
This REPLACES the original total-g-mass constraint. It is not added to it.
A zero reference block gives a vacuous inequality; ratios for it are omitted.
Coverage ratios can exceed 1 and eta=1 need not give the g-only set.

Original prompt, 50% prefix budget, protected positions, greedy generation and
64-token cap are inherited. This is attention access masking, not KV compaction.
The g scorer is still a reconstruction-attention proxy, NOT native KVzip.

API: https://huggingface.co/docs/transformers/v4.56.0/en/attention_interface
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import runpy
import shutil
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

import torch

import logging

logger = logging.getLogger("gstar")

VERSION = 'coverage-selector-eval-v1.0'
ETAS = (0.8, 0.9, 0.95, 1.0)
CONTROL_METHODS = ('full_kv', 'g_proxy_only')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


@contextmanager
def run_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another cell is using this run. Do not run in parallel.') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def checked_attempt(folder: Path) -> tuple[Path, dict]:
    marker = folder/'done.json'
    require(marker.is_file(), f'No completed checkpoint: {marker}')
    done = read_json(marker)
    require(done.get('status') == 'COMPLETE', f'Incomplete checkpoint: {folder}')
    attempt = (folder/done['attempt']).resolve()
    require(attempt.is_relative_to(folder.resolve()), 'Invalid checkpoint path.')
    require(bool(done.get('file_hashes')), 'No checkpoint hashes.')
    for name, expected in done['file_hashes'].items():
        path = (attempt/name).resolve()
        require(path.is_relative_to(attempt) and path.is_file() and digest(path) == expected,
                f'Checkpoint file missing/changed: {path}')
    return attempt, done


def load_tools(source: Path, protocol: dict) -> dict:
    code = source/'code'
    paths = {'expanded': code/'gstar_expanded_eval.py', 'paired': code/'paired.py',
             'smoke': code/'smoke.py', 'full': code/'full.py'}
    expected = {'expanded': protocol['script_sha256'], **protocol['dependency_hashes']}
    for key, path in paths.items():
        require(path.is_file() and digest(path) == expected[key], f'Source code changed: {path}')
    return {key: runpy.run_path(str(path)) for key, path in paths.items()}


def same_layout(left: dict, right: dict) -> None:
    for key in ('document', 'prefix_text', 'prefix_length', 'document_positions',
                'document_token_ids', 'document_character_range'):
        require(left[key] == right[key], f'Layout changed: {key}')
    require(torch.equal(left['prefix_ids'], right['prefix_ids']), 'Prefix token IDs changed.')


def build_coverage_bank(old_bank: dict, coverage_raw: torch.Tensor,
                        document_positions: list[int]) -> dict:
    """Pure CPU selector. No question strings, target labels, answers or F1 input."""
    lambdas = old_bank['lambda_grid']
    require(lambdas == [0., .25, .5, .75, 1.], 'Keep the original five candidates.')
    a = old_bank['normalized_a_free'].detach().double().cpu()
    g = old_bank['normalized_g_free'].detach().double().cpu()
    cov = coverage_raw.detach().double().cpu()
    require(cov.ndim == 4 and tuple(cov.shape[:2]) == tuple(a.shape[:2])
            and cov.shape[-1] == len(document_positions), 'Coverage/score shape mismatch.')
    require(cov.shape[2] == 4, 'This controlled follow-up uses four query blocks.')
    require(bool(torch.isfinite(cov).all() and (cov >= 0).all()), 'Invalid coverage.')
    free_prefix = old_bank['free_prefix_positions'].long().cpu()
    doc_to_col = {p: i for i, p in enumerate(document_positions)}
    require(len(doc_to_col) == len(document_positions), 'Duplicate document positions.')
    require(all(int(p) in doc_to_col for p in free_prefix), 'A free slot is outside the document.')
    free_col = torch.tensor([doc_to_col[int(p)] for p in free_prefix], dtype=torch.long)
    cov_free = cov.index_select(-1, free_col)  # L,H,C,Nfree
    names = ['a_only', 'fixed_lambda_0.25', 'fixed_lambda_0.50',
             'fixed_lambda_0.75', 'g_proxy_only']
    candidate_masks = torch.stack([old_bank['policies'][n]['keep'].bool().cpu() for n in names])
    candidate_free = candidate_masks.index_select(-1, free_prefix)
    A = (candidate_free.double()*a).sum(-1)  # K,L,H
    G = (candidate_free.double()*g).sum(-1)
    require(torch.equal(A, old_bank['candidate_A']), 'Candidate A changed.')
    require(torch.equal(G, old_bank['candidate_G']), 'Candidate G changed.')
    # Accumulate in float64. Use one consistent reduction for feasibility + diagnostics.
    M = torch.stack([(r.double().unsqueeze(-2)*cov_free).sum(-1) for r in candidate_free])
    ref = M[-1]  # g-only reference for EACH block
    positive = ref > 0
    n_layers, n_heads, n_blocks = ref.shape
    li, hi = torch.arange(n_layers)[:, None], torch.arange(n_heads)[None, :]
    budget = old_bank['budget']
    protected = torch.tensor(budget['protected_prefix_positions'], dtype=torch.long)
    policies = {}
    for eta in ETAS:
        feasible = (M >= eta*ref.unsqueeze(0)).all(-1)  # K,L,H
        require(bool(feasible[-1].all()), 'g-only must always be feasible.')
        chosen = A.masked_fill(~feasible, float('-inf')).argmax(0)
        keep = candidate_masks[chosen, li, hi].clone()
        selected_mass = M[chosen, li, hi]
        require(bool((selected_mass >= eta*ref).all()), 'Coverage inequality violated.')
        require(bool((keep.sum(-1) == budget['kept_prefix_tokens_per_layer_kv_head']).all()),
                'Unequal per-layer/head budget.')
        require(bool(keep[..., protected].all()), 'Protected KV removed.')
        ratios = torch.full_like(ref, float('nan'))
        ratios[positive] = selected_mass[positive]/ref[positive]
        active = ratios[positive]
        g_ratio = G[chosen, li, hi]/G[-1]
        same_g = (keep == candidate_masks[-1]).all(-1)
        old_name = f'constrained_rho_{eta:.2f}'
        same_old = (keep == old_bank['policies'][old_name]['keep']).all(-1)
        hist = torch.bincount(chosen.flatten(), minlength=len(lambdas))
        diagnostic = {
            'head_count': n_layers*n_heads,
            'block_comparisons': int(positive.sum()),
            'zero_reference_blocks': int((~positive).sum()),
            'coverage_ratio_min': float(active.min()) if active.numel() else None,
            'coverage_ratio_mean': float(active.mean()) if active.numel() else None,
            'coverage_ratio_max': float(active.max()) if active.numel() else None,
            'coverage_constraint_violations': int((selected_mass < eta*ref).sum()),
            'g_retention_min': float(g_ratio.min()),
            'g_retention_mean': float(g_ratio.mean()),
            'g_only_identical_head_fraction': float(same_g.double().mean()),
            'whole_cache_equals_g_only': bool(same_g.all()),
            'same_as_old_total_g_constraint_head_fraction': float(same_old.double().mean()),
            'whole_cache_equals_old_total_g_constraint': bool(same_old.all()),
            'lambda_histogram': {str(lam): int(hist[k]) for k, lam in enumerate(lambdas)},
            'only_g_candidate_feasible_head_fraction': float((~feasible[:-1].any(0)).double().mean()),
            'feasible_candidates_mean': float(feasible.sum(0).double().mean()),
        }
        # Match original mask_key convention exactly.
        key = hashlib.sha256(str(tuple(keep.shape)).encode()+keep.contiguous().numpy().tobytes()).hexdigest()
        name = f'coverage_eta_{eta:.2f}'
        policies[name] = {
            'family': 'coverage', 'parameter': eta, 'source_method': name,
            'keep': keep, 'mask_key': key, 'diagnostics': diagnostic,
            'per_head': {'selected_lambda_index': chosen, 'feasible_candidate_count': feasible.sum(0),
                         'block_retained_mass': selected_mass, 'block_reference_mass': ref,
                         'coverage_ratio': ratios, 'positive_reference': positive,
                         'g_retention': g_ratio, 'same_set_as_g_only': same_g},
        }
    return {'policies': policies, 'budget': budget, 'lambda_grid': lambdas,
            'eta_grid': list(ETAS), 'free_prefix_positions': free_prefix,
            'candidate_block_mass': M, 'reference_block_mass': ref, 'candidate_A': A,
            'constraint': 'all-block coverage floor; replaces total-g constraint',
            'zero_reference_rule': 'vacuous inequality; ratio excluded from numeric summaries'}


def _source_inputs(source: Path, protocol: dict) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in read_jsonl(source/'data/development/selection_inputs.jsonl'):
        require(set(row) == {'case_id','document_id','document','first_question'}, 'Unexpected selector input.')
        grouped[row['document_id']].append(row)
    require(set(grouped) == set(protocol['document_ids']['development']), 'Development input IDs changed.')
    for doc, rows in grouped.items():
        require(len(rows) == 2 and len({r['case_id'] for r in rows}) == 2 and
                rows[0]['document'] == rows[1]['document'], f'Invalid pair: {doc}')
    return dict(grouped)


def _selection_summary(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        for row in result['diagnostics']:
            grouped[row['method']].append(row['diagnostics'])
    methods = {}
    for name, ds in grouped.items():
        heads = sum(d['head_count'] for d in ds)
        blocks = sum(d['block_comparisons'] for d in ds)
        hist = Counter()
        for d in ds:
            hist.update(d['lambda_histogram'])
        valid = [d for d in ds if d['block_comparisons']]
        methods[name] = {
            'layer_head_decisions': heads, 'block_comparisons': blocks,
            'constraint_violations': sum(d['coverage_constraint_violations'] for d in ds),
            'zero_reference_blocks': sum(d['zero_reference_blocks'] for d in ds),
            'coverage_ratio_min': min(d['coverage_ratio_min'] for d in valid) if valid else None,
            'coverage_ratio_mean': sum(d['coverage_ratio_mean']*d['block_comparisons'] for d in valid)/blocks if blocks else None,
            'coverage_ratio_max': max(d['coverage_ratio_max'] for d in valid) if valid else None,
            'g_only_identical_head_fraction': sum(d['g_only_identical_head_fraction']*d['head_count'] for d in ds)/heads,
            'same_as_old_total_g_constraint_head_fraction': sum(d['same_as_old_total_g_constraint_head_fraction']*d['head_count'] for d in ds)/heads,
            'whole_cache_g_only_cases': sum(d['whole_cache_equals_g_only'] for d in ds),
            'whole_cache_old_constraint_cases': sum(d['whole_cache_equals_old_total_g_constraint'] for d in ds),
            'lambda_histogram': dict(hist),
            'g_retention_mean': sum(d['g_retention_mean']*d['head_count'] for d in ds)/heads,
            'g_retention_min': min(d['g_retention_min'] for d in ds),
        }
    return {'documents': len(results), 'cases': sum(len(x['case_ids']) for x in results), 'methods': methods}


def prepare_coverage_selectors(source_experiment_dir: str | Path,
                               coverage_dir: str | Path) -> str:
    """CPU ONLY: all original development docs; no model or evaluation labels read."""
    source, covdir = Path(source_experiment_dir).resolve(), Path(coverage_dir).resolve()
    protocol = read_json(source/'protocol.json')
    cp = read_json(covdir/'coverage_protocol.json')
    require(protocol['version'] == 'expanded-span-v2.1', 'Unexpected original protocol.')
    require(cp['source_protocol_sha256'] == digest(source/'protocol.json') and
            cp['split'] == 'development' and cp['n_blocks'] == 4, 'Wrong coverage source/config.')
    require(cp['source_runtime'] == protocol['runtime'], 'Coverage runtime differs.')
    require(cp['normalized'] is False and cp['protected_positions_applied'] is False,
            'Expected raw coverage with all document keys.')
    require(digest(covdir/'extractor_snapshot.py') == cp['extractor_sha256'], 'Coverage extractor snapshot changed.')
    inputs_path = source/'data/development/selection_inputs.jsonl'
    require(digest(inputs_path) == cp['selection_inputs_sha256'], 'Development inputs changed.')
    tools = load_tools(source, protocol)
    groups = _source_inputs(source, protocol)
    docs = protocol['document_ids']['development']
    identity = {
        'version': VERSION, 'runner_sha256': digest(Path(__file__)),
        'source_experiment': str(source), 'source_protocol_sha256': digest(source/'protocol.json'),
        'coverage_directory': str(covdir), 'coverage_protocol_sha256': digest(covdir/'coverage_protocol.json'),
        'selection_inputs_sha256': digest(inputs_path), 'document_ids': docs,
        'eta_grid': list(ETAS), 'lambda_grid': protocol['lambda_grid'], 'n_blocks': 4,
        'constraint': 'For every layer/KV-head and every block c, M_c(R)>=eta*M_c(R_g); maximize A among original 5 candidates.',
        'total_g_constraint_also_applied': False, 'protected_tokens_count_in_budget': True,
        'zero_reference_rule': 'No epsilon denominator. Zero reference gives a vacuous inequality; omit its ratio.',
        'tie_break': 'smaller lambda among exact A ties; candidate token ties inherited from V1',
        'prediction_reuse_rule': 'exact full mask AND same original case/question tokens only; fresh Full/g-only controls required per case',
        'candidate_selection_uses_gold_or_target': False,
        'development_only': True, 'heldout_read': False,
        'evaluation_scope': 'exploratory V2, after observing V1 heldout; needs a fresh test for new validation',
        'tuning_objective': protocol['tuning_objective'],
    }
    out = source.parent/f'coverage_eval_v1_b4_{object_digest(identity)[:12]}'
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out/'.run.lock'):
        ipath = out/'selector_protocol.json'
        if ipath.is_file():
            require(read_json(ipath) == identity, 'Selector identity changed.')
        else:
            save_json(ipath, identity)
            shutil.copy2(__file__, out/'runner_snapshot.py')
        results = []
        logger.debug('%s', '========== COVERAGE SELECTOR BUILD (CPU ONLY) ==========')
        for i, doc in enumerate(docs, 1):
            original, original_done = checked_attempt(source/'runs/development'/doc)
            coverage, coverage_done = checked_attempt(covdir/doc)
            fingerprint = {
                'source_done_sha256': digest(source/'runs/development'/doc/'done.json'),
                'coverage_done_sha256': digest(covdir/doc/'done.json'),
                'source_scores_sha256': original_done['file_hashes']['scores.pt'],
                'coverage_file_sha256': coverage_done['file_hashes']['coverage.pt'],
            }
            folder = out/'selectors'/doc
            if (folder/'done.json').is_file():
                completed, done = checked_attempt(folder)
                require(done['source_fingerprint'] == fingerprint, 'Source checkpoint changed.')
                results.append(read_json(completed/'summary.json'))
                logger.debug('%s', f'[{i}/{len(docs)}] {doc}: verified existing selectors.')
                continue
            attempt = folder/f'attempt_{stamp()}'
            attempt.mkdir(parents=True, exist_ok=False)
            try:
                scores = torch.load(original/'scores.pt', map_location='cpu', weights_only=True)
                cov = torch.load(coverage/'coverage.pt', map_location='cpu', weights_only=True)
                layout = scores['layout']
                require(cov['document_id'] == doc and cov['split'] == 'development', 'Coverage document changed.')
                require(cov['source_fingerprint']['scores_sha256'] == fingerprint['source_scores_sha256']
                        and cov['source_fingerprint']['done_sha256'] == fingerprint['source_done_sha256'],
                        'Coverage is linked to a different V1 checkpoint.')
                require(cov['source_protocol_sha256'] == identity['source_protocol_sha256'], 'Coverage protocol changed.')
                require(cov['g_recheck']['within_tolerance'] is True, 'Coverage g recheck did not pass.')
                require(cov['normalized'] is False and cov['protected_positions_applied'] is False, 'Unexpected coverage transformation.')
                for key in ('document_positions', 'document_token_ids', 'prefix_length'):
                    require(cov[key] == layout[key], f'Coverage layout mismatch: {key}')
                require(torch.equal(cov['prefix_input_ids'], layout['prefix_ids']), 'Coverage prefix tokens differ.')
                require(groups[doc][0]['document'] == layout['document'], 'Input document differs from scores.')
                require(set(scores['cases']) == {r['case_id'] for r in groups[doc]}, 'Score case IDs differ.')
                diagnostics = []
                for row in groups[doc]:
                    cid = row['case_id']
                    require(scores['cases'][cid]['first_question'] == row['first_question'], 'Scoring question differs.')
                    bank = tools['paired']['build_policy_bank'](
                        scores['cases'][cid]['a_raw'], scores['g_raw_proxy'], layout['document_positions'],
                        layout['prefix_length'], protocol['keep_fraction'], tools['smoke'])
                    previous = torch.load(original/f'{cid}_selection.pt', map_location='cpu', weights_only=True)
                    require(bank['budget'] == previous['budget'], 'Protection or budget changed.')
                    require(set(bank['policies']) == set(previous['policies']), 'Original method grid changed.')
                    for name, p in bank['policies'].items():
                        q = previous['policies'][name]
                        require(p['mask_key'] == q['mask_key'] and (p['keep'] is None and q['keep'] is None
                                or p['keep'] is not None and q['keep'] is not None and torch.equal(p['keep'],q['keep'])),
                                f'Original mask not reproduced: {cid}/{name}')
                    new = build_coverage_bank(bank, cov['coverage_raw'], layout['document_positions'])
                    torch.save({'case_id': cid, 'document_id': doc, **new}, attempt/f'{cid}_selection.pt')
                    diagnostics.extend({'case_id': cid, 'document_id': doc, 'method': name,
                                        'family': 'coverage', 'parameter': p['parameter'],
                                        'mask_key': p['mask_key'], 'diagnostics': p['diagnostics']}
                                       for name, p in new['policies'].items())
                result = {'status': 'COMPLETE', 'document_id': doc,
                          'case_ids': [r['case_id'] for r in groups[doc]], 'diagnostics': diagnostics,
                          'source_fingerprint': fingerprint, 'source_attempt': str(original),
                          'coverage_attempt': str(coverage), 'original_masks_reproduced': True}
                save_json(attempt/'summary.json', result)
                hashes = {p.name: digest(p) for p in attempt.iterdir() if p.is_file()}
                save_json(folder/'done.json', {'status': 'COMPLETE', 'attempt': attempt.name,
                                              'source_fingerprint': fingerprint, 'file_hashes': hashes})
                results.append(result)
                logger.debug('%s', f'[{i}/{len(docs)}] {doc}: 4 eta settings; budgets and block constraints PASS.')
            except BaseException as exc:
                save_json(attempt/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
                raise
        summary = _selection_summary(results)
        save_json(out/'selection_summary.json', summary)
        save_jsonl(out/'selection_diagnostics.jsonl', [r for x in results for r in x['diagnostics']])
        logger.debug('%s', '\n========== COVERAGE SELECTOR SUMMARY ==========')
        logger.debug('%s', f"Documents: {summary['documents']} | cases: {summary['cases']}")
        for name, d in summary['methods'].items():
            logger.debug('%s', f"{name}: violations={d['constraint_violations']}, min coverage={d['coverage_ratio_min']}, "
                  f"g-only heads={d['g_only_identical_head_fraction']:.1%}, "
                  f"same-as-old-constraint heads={d['same_as_old_total_g_constraint_head_fraction']:.1%}")
            logger.debug('%s %s', '  lambda counts:', d['lambda_histogram'])
        logger.info('%s %s', 'Selector checks only; no answers generated. Saved under:', out)
    return str(out)


def _verify_selector_root(selector: Path) -> tuple[dict, Path, dict, dict]:
    identity = read_json(selector/'selector_protocol.json')
    require(identity['version'] == VERSION and identity['runner_sha256'] == digest(Path(__file__)),
            'Selector runner changed; use the original file, not a silent patch.')
    source = Path(identity['source_experiment'])
    require(digest(source/'protocol.json') == identity['source_protocol_sha256'], 'Source protocol changed.')
    cp = Path(identity['coverage_directory'])/'coverage_protocol.json'
    require(digest(cp) == identity['coverage_protocol_sha256'], 'Coverage metadata changed.')
    require(digest(source/'data/development/selection_inputs.jsonl') == identity['selection_inputs_sha256'],
            'Source development inputs changed.')
    protocol = read_json(source/'protocol.json')
    tools = load_tools(source, protocol)
    return identity, source, protocol, tools


def _combined_summary(results: list[dict], expanded: dict, planned: int) -> dict:
    summary = expanded['summarize'](results, 'development', planned)
    summary['scope'] = 'Exploratory V2 DEVELOPMENT. Original 15 rows reused from V1; four coverage settings added.'
    summary['source_baselines_are_archived_not_new_generations'] = True
    summary['coverage_diagnostics'] = _selection_summary([
        {'case_ids': sorted({r['case_id'] for r in x['coverage_diagnostics']}),
         'diagnostics': x['coverage_diagnostics']} for x in results])['methods']
    summary['new_coverage_answer_records'] = sum(x['new_coverage_answer_records'] for x in results)
    summary['new_coverage_generation_calls'] = sum(x['new_coverage_generation_calls'] for x in results)
    summary['control_generation_calls'] = sum(x['control_generation_calls'] for x in results)
    summary['control_token_limit_hits'] = sum(x['control_token_limit_hits'] for x in results)
    summary['coverage_records_reusing_old_predictions'] = sum(x['reuse_old'] for x in results)
    summary['coverage_records_reusing_new_predictions'] = sum(x['reuse_new'] for x in results)
    summary['new_coverage_limit_hits'] = sum(x['new_coverage_limit_hits'] for x in results)
    summary['new_coverage_records_hitting_limit_including_reuse'] = sum(r['hit_token_limit'] for x in results
                                                                     for r in x['rows'] if r['family']=='coverage')
    summary['coverage_selected_on_development'] = None
    if len(results) == planned:
        methods = summary['methods']
        names = [k for k, v in methods.items() if v['family']=='coverage']
        summary['coverage_selected_on_development'] = max(names, key=lambda k:(methods[k]['worst_target_f1'], methods[k]['first']['f1']))
    return summary


def _write_comparison(out: Path, summary: dict, results: list[dict], expanded: dict) -> None:
    expanded['_write_summary'](out, summary, results)
    save_jsonl(out/'coverage_diagnostics.jsonl', [r for x in results for r in x['coverage_diagnostics']])
    logger.debug('%s', '\n========== COVERAGE DEVELOPMENT SUMMARY ==========')
    logger.debug('%s', f"Status: {summary['status']} | documents {summary['documents_completed']}/{summary['documents_planned']}")
    logger.debug('%s', 'EXPLORATORY DEVELOPMENT; not a new held-out estimate.')
    logger.debug('%s', f"{'Method':27s} {'First':>8s} {'Same':>8s} {'Diff':>8s} {'Worst':>8s}")
    for name, m in summary['methods'].items():
        logger.debug('%s', f"{name:27s} {m['first']['f1']:8.2f} {m['same']['f1']:8.2f} {m['different']['f1']:8.2f} {m['worst_target_f1']:8.2f}")
    logger.debug('%s %s', 'Coverage selected on complete development:', summary['coverage_selected_on_development'])
    logger.debug('%s %s %s %s', 'New coverage QA calls:', summary['new_coverage_generation_calls'], '| validation QA calls:', summary['control_generation_calls'])
    logger.debug('%s %s %s %s', 'Exact-mask reuse: old=', summary['coverage_records_reusing_old_predictions'], 'new=', summary['coverage_records_reusing_new_predictions'])
    logger.debug('%s %s', 'New coverage QA token-limit hits:', summary['new_coverage_limit_hits'])
    logger.info('%s %s', 'Results:', out)


def _evaluate_document(model: Any, tokenizer: Any, group: dict, selector: Path,
                       source: Path, tools: dict, protocol: dict, folder: Path) -> dict:
    smoke, paired, full, expanded = (tools[k] for k in ('smoke','paired','full','expanded'))
    doc = group['document_id']
    selected, selected_done = checked_attempt(selector/'selectors'/doc)
    source_attempt, source_done = checked_attempt(source/'runs/development'/doc)
    checkpoint = read_json(selected/'summary.json')
    require(checkpoint['source_fingerprint']['source_done_sha256'] == digest(source/'runs/development'/doc/'done.json'),
            'Development checkpoint changed since selector preparation.')
    scores = torch.load(source_attempt/'scores.pt', map_location='cpu', weights_only=True)
    old_result = read_json(source_attempt/'result.json')
    layout = paired['make_layout'](tokenizer, expanded['render_prompt'], group['document'])
    same_layout(layout, scores['layout'])
    banks, old_banks = {}, {}
    for ins, label in group['cases']:
        cid = ins['case_id']
        banks[cid] = torch.load(selected/f'{cid}_selection.pt', map_location='cpu', weights_only=True)
        old_banks[cid] = torch.load(source_attempt/f'{cid}_selection.pt', map_location='cpu', weights_only=True)
        require(ins['first_question'] == scores['cases'][cid]['first_question'], 'Question changed.')
    attempt = folder/f'attempt_{stamp()}'
    attempt.mkdir(parents=True, exist_ok=False)
    new_rows, new_diag, audits = [], [], []
    controls, calls, limit_hits, control_limit_hits, reuse_old, reuse_new = 0, 0, 0, 0, 0, 0
    try:
        p = layout['prefix_length']
        with torch.no_grad():
            ids = layout['prefix_ids'].to(model.device)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                        logits_to_keep=1, return_dict=True)
        base = out.past_key_values
        del out
        snapshot = smoke['prefix_cache_snapshot'](base)
        require(int(base.get_seq_length()) == p, 'Cache length mismatch.')
        # Original saved rows are never modified.
        old_rows = {(r['case_id'], r['method'], r['role']): r for r in old_result['rows']}
        for ins, label in group['cases']:
            cid = ins['case_id']
            questions = []
            for role, field in [('first','first_question_reference'),('target','target_question_reference')]:
                ref = label[field]
                ids = paired['tokenized'](tokenizer, expanded['render_prompt'](tokenizer, group['document'],ref['question']))['input_ids']
                require(torch.equal(ids[:,:p], layout['prefix_ids']), 'Question prefix changed.')
                require(p < ids.shape[1] <= expanded['MAX_ANSWER_INPUT_TOKENS'], 'Input too long; no truncation.')
                questions.append((role, ref, ids[:,p:].to(model.device)))
            # Fresh controls: same inputs, code, backend, weights and generation configuration.
            for method in CONTROL_METHODS:
                keep = old_banks[cid]['policies'][method]['keep']
                for role, ref, suffix in questions:
                    fresh = smoke['generate_answer'](model, tokenizer, suffix, base, keep, protocol['max_new_tokens'])
                    expected = old_rows[(cid, method, role)]
                    controls += 1
                    control_limit_hits += int(fresh['hit_token_limit'])
                    ok = fresh['generated_token_ids'] == expected['generated_token_ids']
                    audits.append({'case_id':cid, 'test':'fresh_source_control','method':method,'role':role,
                                   'tokens_identical':ok, 'answer':fresh['answer'],'expected':expected['answer'],
                                   'mask_audit':fresh['mask_audit']})
                    save_json(attempt/'audits.json', audits)
                    require(ok, 'Fresh Full/g-only control differs from archived source. Stop; do not mix baselines. Inspect audits.json.')
            # All-keep equation test with explicit question + one continuation.
            first_suffix = questions[0][2]
            clean, token, _ = smoke['fixed_trace'](model, first_suffix, base, None)
            all_keep = torch.ones_like(banks[cid]['policies']['coverage_eta_0.80']['keep'])
            traced, _, audit = smoke['fixed_trace'](model, first_suffix, base, all_keep, token)
            match = smoke['compare_traces'](clean, traced)
            require(match['exactly_equal'], 'All-keep equivalence failed.')
            audits.append({'case_id':cid,'test':'all_keep','comparison':match,'audit':audit})
            # Audit every distinct new coverage mask with a fixed continuation as well.
            seen = set()
            for name, policy in banks[cid]['policies'].items():
                if policy['mask_key'] in seen:
                    continue
                seen.add(policy['mask_key'])
                trace, token, audit1 = smoke['fixed_trace'](model, first_suffix, base, policy['keep'])
                poisoned = smoke['poison_excluded'](base, policy['keep'])
                altered, _, audit2 = smoke['fixed_trace'](model, first_suffix, poisoned, policy['keep'], token)
                del poisoned
                match = smoke['compare_traces'](trace, altered)
                require(match['exactly_equal'], 'Excluded KV perturbation changed logits.')
                audits.append({'case_id':cid,'test':'excluded_kv','method':name,'comparison':match,
                               'clean_audit':audit1,'perturbed_audit':audit2})
            require(smoke['cache_intact'](base, snapshot, p), 'Original cache changed during controls/audits.')
            # Exact whole-mask equivalence, not score similarity, allows old prediction reuse.
            memo = {}
            for name, policy in banks[cid]['policies'].items():
                new_diag.append({'case_id':cid,'document_id':doc,'condition':label['condition'],
                                 'method':name,'source_method':name,'family':'coverage','parameter':policy['parameter'],
                                 'mask_key':policy['mask_key'],'diagnostics':policy['diagnostics']})
                equivalent = next((n for n, old in old_banks[cid]['policies'].items()
                                   if old['keep'] is not None and old['mask_key']==policy['mask_key']
                                   and torch.equal(old['keep'],policy['keep'])), None)
                for role, ref, suffix in questions:
                    key = (policy['mask_key'], tuple(suffix.detach().cpu().flatten().tolist()))
                    if equivalent is not None:
                        old = old_rows[(cid, equivalent, role)]
                        result = {k:copy.deepcopy(old[k]) for k in ('answer','generated_token_ids','hit_token_limit','mask_audit')}
                        origin, reuse_ref = 'archived_v1_exact_mask', old['record_id']
                        require(old['question']==ref['question'] and old['mask_key']==policy['mask_key'], 'Archived input/mask mismatch.')
                        reuse_old += 1
                    elif key in memo:
                        result, reuse_ref = memo[key]
                        origin = 'current_run_exact_mask'
                        reuse_new += 1
                    else:
                        result = smoke['generate_answer'](model,tokenizer,suffix,base,policy['keep'],protocol['max_new_tokens'])
                        origin, reuse_ref = 'new_coverage_generation', None
                        calls += 1; limit_hits += int(result['hit_token_limit'])
                        memo[key] = (result, f'{cid}/{name}/{role}')
                    intact = smoke['cache_intact'](base,snapshot,p)
                    require(intact, 'Original document KV modified.')
                    gold = [a['text'] for a in ref['answers']]
                    row = {'record_id':f'{cid}/{name}/{role}','case_id':cid,'document_id':doc,
                           'project_split':'development','source_article':label['source_article'],
                           'condition':label['condition'],'role':role,'method':name,'source_method':name,
                           'family':'coverage','parameter':policy['parameter'],'question_id':ref['question_id'],
                           'question':ref['question'],'references':gold,**full['score_answer'](result['answer'],gold),
                           'mask_key':policy['mask_key'],'original_kv_unchanged':intact,'prefix_tokens':p,
                           'kept_prefix_tokens':banks[cid]['budget']['kept_prefix_tokens_per_layer_kv_head'],
                           'prediction_origin':origin,'reused_from_record':reuse_ref,
                           'answer_is_verbatim_context_substring':bool(result['answer']) and result['answer'] in group['document'],
                           'answer_whitespace_words':len(result['answer'].split()),**copy.deepcopy(result)}
                    new_rows.append(row)
                    save_jsonl(attempt/'new_predictions.jsonl',new_rows)
            logger.debug('%s', f'  {cid}: coverage answers recorded; controls/audits PASS.')
        archived = [dict(r,prediction_origin='archived_v1_baseline') for r in old_result['rows']]
        combined = archived+new_rows
        expanded['_check_pairs'](combined)
        result = {'document_id':doc,'source_article':old_result['source_article'],
                  'rows':combined,'diagnostics':copy.deepcopy(old_result['diagnostics'])+new_diag,
                  'coverage_diagnostics':new_diag,'audits':audits,
                  'actual_generation_calls':calls+controls,'token_limit_hits':limit_hits+control_limit_hits,
                  'new_coverage_generation_calls':calls,'new_coverage_limit_hits':limit_hits,
                  'control_generation_calls':controls,'control_token_limit_hits':control_limit_hits,
                  'new_coverage_answer_records':len(new_rows),
                  'reuse_old':reuse_old,'reuse_new':reuse_new}
        save_json(attempt/'audits.json',audits)
        save_json(attempt/'result.json',result)
        save_json(folder/'done.json',{'status':'COMPLETE','attempt':attempt.name,
                  'source_done_sha256':digest(source/'runs/development'/doc/'done.json'),
                  'selector_done_sha256':digest(selector/'selectors'/doc/'done.json'),
                  'file_hashes':{q.name:digest(q) for q in attempt.iterdir() if q.is_file()}})
        return result
    except BaseException as exc:
        save_json(attempt/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),
                                         'new_answer_records_saved':len(new_rows)})
        raise


def run_coverage_development(model: Any, tokenizer: Any, selector_dir: str | Path,
                             max_new_documents: int | None = 1) -> dict:
    """Resume document checkpoints. None processes all remaining DEVELOPMENT docs."""
    selector = Path(selector_dir).resolve()
    identity, source, protocol, tools = _verify_selector_root(selector)
    expanded, smoke = tools['expanded'], tools['smoke']
    require(max_new_documents is None or (type(max_new_documents) is int and max_new_documents>0), 'Invalid batch size.')
    runtime = expanded['runtime_fingerprint'](model,tokenizer)
    require(runtime==protocol['runtime'], 'Model/tokenizer/runtime differs. Do not bypass or reinstall.')
    require(expanded['SYSTEM_PROMPT']==protocol['system_prompt'], 'Prompt changed.')
    inputs = read_jsonl(source/'data/development/selection_inputs.jsonl')
    labels = read_jsonl(source/'data/development/evaluation_only.jsonl')
    require(object_digest({'inputs':inputs,'labels':labels})==protocol['data_content_hashes']['development'],
            'Development inputs/labels changed.')
    groups = expanded['_groups'](source,'development')
    require([g['document_id'] for g in groups]==identity['document_ids'], 'Document order changed.')
    old_backend, old_training = model.config._attn_implementation, model.training
    old_hooks = smoke['hook_counts'](model)
    require(all(x==(0,0) for x in old_hooks), 'Existing attention hooks; do not run concurrently.')
    out = selector/'evaluation'
    out.mkdir(parents=True,exist_ok=True)
    results, new_count = [], 0
    with run_lock(selector/'.run.lock'), run_lock(source/'.run.lock'):
        try:
            model.eval(); model.set_attn_implementation('eager')
            for i, group in enumerate(groups,1):
                doc = group['document_id']
                folder = out/doc
                if (folder/'done.json').is_file():
                    attempt, done = checked_attempt(folder)
                    require(done['source_done_sha256']==digest(source/'runs/development'/doc/'done.json')
                            and done['selector_done_sha256']==digest(selector/'selectors'/doc/'done.json'),
                            'Source/selector changed after evaluation.')
                    results.append(read_json(attempt/'result.json'))
                    logger.debug('%s', f'[{i}/{len(groups)}] {doc}: verified completed evaluation; skipping GPU.')
                    continue
                if max_new_documents is not None and new_count>=max_new_documents:
                    continue
                logger.debug('%s', f'[{i}/{len(groups)}] {doc}: fresh controls, masking audits, coverage QA ...')
                result = _evaluate_document(model,tokenizer,group,selector,source,tools,protocol,folder)
                results.append(result); new_count += 1
                summary = _combined_summary(results,expanded,len(groups))
                expanded['_write_summary'](out,summary,results)
                logger.info('%s', f"  Saved {doc}: {result['new_coverage_generation_calls']} new QA calls, "
                      f"{result['control_generation_calls']} control QA calls.")
            require(bool(results),'No completed documents.')
            summary = _combined_summary(results,expanded,len(groups))
            _write_comparison(out,summary,results,expanded)
        except BaseException as exc:
            save_json(out/f'failure_{stamp()}.json',{'error':str(exc),'traceback':traceback.format_exc()})
            logger.error('%s', 'Stopped. Completed checkpoints remain. Inspect error; do not change source files.')
            raise
        finally:
            try:
                model.set_attn_implementation(old_backend)
            finally:
                model.train(old_training)
                restored = {'backend_after':model.config._attn_implementation,
                            'attention_backend_restored':model.config._attn_implementation==old_backend,
                            'training_mode_restored':model.training==old_training,
                            'hook_counts_restored':smoke['hook_counts'](model)==old_hooks,
                            'original_protocol_unchanged':digest(source/'protocol.json')==identity['source_protocol_sha256']}
                save_json(out/'restoration.json',restored)
                require(all(v for k,v in restored.items() if k!='backend_after'),'Model/settings not restored.')
    logger.debug('%s %s %s %s', 'Hooks restored:', restored['hook_counts_restored'], '| backend:', restored['backend_after'])
    summary['output_directory']=str(out)
    summary['restoration'] = restored
    save_json(out/'summary.json', summary)
    return summary


def export_review(selector_dir: str | Path) -> str:
    """Small ZIP for review. No GPU, no tensor files, no held-out data."""
    root = Path(selector_dir).resolve()
    identity, source, protocol, tools = _verify_selector_root(root)
    archive = root/f'coverage_review_{stamp()}.zip'
    files = [root/n for n in ('selector_protocol.json','selection_summary.json','selection_diagnostics.jsonl')]
    files += [root/'evaluation'/n for n in ('summary.json','predictions.jsonl','diagnostics.jsonl',
                                          'coverage_diagnostics.jsonl','paired_targets.json','comparison.csv','restoration.json')]
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as z:
        for path in files:
            if path.is_file():
                z.write(path,arcname=str(path.relative_to(root)))
        for name in ('selection_inputs.jsonl','evaluation_only.jsonl'):
            z.write(source/'data/development'/name,arcname=f'source_development/{name}')
    logger.info('%s %s', 'Review ZIP:', archive)
    return str(archive)
