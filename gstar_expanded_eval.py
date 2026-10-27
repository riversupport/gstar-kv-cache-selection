"""GSTAR expanded, resumable paired evaluation (short-document prototype).

Uses the user's existing, validated gstar_paired_pilot.py, gstar_mask_smoke.py,
and gstar_fullkv_pilot/{gstar_run_fullkv.py,gstar_prepare_pilot.py} WITHOUT editing them.

Stages: prepare_experiment -> run_development -> freeze_protocol -> run_heldout.
No training, model downloads, installations, or native KVzip / speed claims.
A source-data download is needed only if the original SQuAD JSON cache is missing.
Target: Transformers 4.56.x, unquantized Qwen2.5, batch=1, same original model.

All original pilot articles remain in the development pool. Source-article splitting
is inherited and verified, never re-randomized. The development sample includes the
original pilot; held-out means held out from THIS PROJECT, not from LLM pretraining.

A new span-format prompt is used for ALL methods, including fresh Full-KV scoring.
The common prefix is identical for QA and reconstruction; the repeat-context
instruction is explicitly exempted from the short-answer instruction.

Checkpoint unit: one fully completed document. Repeating a run skips verified
completed documents. An interrupted document is retried in a NEW attempt directory.
Completed files are hash-checked; partial attempts are preserved, not aggregated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import runpy
import shutil
import time
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

VERSION = 'expanded-span-v2.1'
SYSTEM_PROMPT = (
    'Use only the provided context. For questions, return only a short verbatim '
    'answer span from the context, without an introductory sentence or explanation. '
    'If asked to repeat the context, reproduce the context exactly.'
)
MAX_SCORING_TOKENS = 1024
MAX_ANSWER_INPUT_TOKENS = 1152
DEPENDENCIES = {
    'paired': 'gstar_paired_pilot.py',
    'smoke': 'gstar_mask_smoke.py',
    'full': 'gstar_fullkv_pilot/gstar_run_fullkv.py',
    'builder': 'gstar_fullkv_pilot/gstar_prepare_pilot.py',
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')


def log_row(handle: Any, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    handle.flush()


@contextmanager
def run_lock(path: Path):
    # Modal is Linux. flock is automatically released even if the process dies.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another cell/process is using this experiment. Do not run it in parallel.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_tools(project: Path) -> tuple[dict, dict]:
    tools, hashes = {}, {}
    for name, relative in DEPENDENCIES.items():
        path = project / relative
        require(path.is_file(), f'Keep the existing dependency: {path}')
        hashes[name] = digest(path)
        tools[name] = runpy.run_path(str(path))
    return tools, hashes


def runtime_fingerprint(model: Any, tokenizer: Any) -> dict:
    import transformers
    require(transformers.__version__.startswith('4.56.'), 'Expected Transformers 4.56.x; do not reinstall automatically.')
    require(model.config.model_type == 'qwen2', 'Use the same Qwen2.5 model as the pilot.')
    require(not getattr(model, 'is_quantized', False), 'Use the original unquantized model.')
    require(tokenizer.is_fast, 'A fast tokenizer is required for token offsets.')
    require(not getattr(model.config, 'use_sliding_window', False), 'Sliding-window models are not supported.')
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    # Configuration and pinned checkpoint identify the model; weights are NOT hashed.
    cfg = model.config.to_dict()
    for key in ['_attn_implementation', '_attn_implementation_internal', '_attn_implementation_autoset']:
        cfg.pop(key, None)
    return {
        'model_id': model.config._name_or_path,
        'model_revision': getattr(model.config, '_commit_hash', None),
        'config_hash': object_digest(cfg),
        'dtype': str(model.dtype),
        'device_type': model.device.type,
        'torch_version': str(torch.__version__),
        'transformers_version': transformers.__version__,
        'tokenizer_hash': tokenizer_hash,
        'chat_template': tokenizer.chat_template,
        'eos_token_id': model.generation_config.eos_token_id,
    }


def render_prompt(tokenizer: Any, document: str, question: str) -> str:
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Context:\n{document}\n\nQuestion:\n{question}'},
    ], tokenize=False, add_generation_prompt=True)


def token_lengths_and_layout(pack: dict, tokenizer: Any, paired: dict, smoke: dict) -> dict:
    """Input-only eligibility check. No model calls, metrics, or predictions."""
    layout = paired['make_layout'](tokenizer, render_prompt, pack['document'])
    p = layout['prefix_length']
    request = render_prompt(tokenizer, pack['document'], paired['RECONSTRUCTION_INSTRUCTION'])
    req = paired['tokenized'](tokenizer, request)['input_ids']
    require(torch.equal(req[:, :p], layout['prefix_ids']), 'Reconstruction prefix-token mismatch.')
    reconstruction_tokens = req.shape[1] + len(layout['document_positions'])
    require(reconstruction_tokens <= MAX_SCORING_TOKENS, 'reconstruction_length_exceeds_1024')
    lengths = {}
    for key in ['target', 'same_first', 'different_first']:
        enc = paired['tokenized'](tokenizer, render_prompt(tokenizer, pack['document'], pack[key]['question']))
        ids = enc['input_ids']
        require(torch.equal(ids[:, :p], layout['prefix_ids']), 'Question prefix-token mismatch.')
        limit = MAX_ANSWER_INPUT_TOKENS if key == 'target' else MAX_SCORING_TOKENS
        require(p < ids.shape[1] <= limit, f'{key}_length_outside_limit')
        lengths[key] = ids.shape[1]
    dummy = torch.ones(1, 1, len(layout['document_positions']))
    _, budget = smoke['make_masks'](dummy, dummy, layout['document_positions'], p, 0.5)
    return {'prefix_tokens': p, 'reconstruction_tokens': reconstruction_tokens,
            'question_input_lengths': lengths, 'kept_prefix_tokens': budget['kept_prefix_tokens_per_layer_kv_head']}


def _sample_packs(articles: list[dict], count: int, split: str,
                  tokenizer: Any, tools: dict) -> tuple[list[dict], list[dict]]:
    packs, screening = [], []
    builder = tools['builder']
    for article in articles:
        triple = builder['choose_triple'](article)
        if triple is None:
            screening.append({'source_article': article['title'], 'selected': False, 'reason': 'no_original_builder_triple'})
            continue
        pack = builder['make_pack'](article, triple, len(packs) + 1)
        pack['document_id'] = f'{split}_{len(packs)+1:03d}'
        pack['project_split'] = split
        try:
            lengths = token_lengths_and_layout(pack, tokenizer, tools['paired'], tools['smoke'])
        except ValueError as exc:
            screening.append({'source_article': article['title'], 'selected': False, 'reason': str(exc)})
            continue
        screening.append({'source_article': article['title'], 'selected': True, **lengths})
        packs.append(pack)
        if len(packs) == count:
            break
    require(len(packs) == count, f'Only {len(packs)} eligible {split} articles; requested {count}. Nothing was evaluated.')
    return packs, screening


def _export_split(packs: list[dict], builder: dict, split: str) -> tuple[list[dict], list[dict]]:
    inputs, labels = builder['export_rows'](packs)
    for label in labels:
        label['project_split'] = split
        label['manual_review_status'] = 'not_reviewed'
    return inputs, labels


def _preview(inputs: list[dict], labels: list[dict], split: str) -> str:
    s = [f'GSTAR {split.upper()} INPUT REVIEW — no model predictions',
         'Original contexts/questions/answers are unmodified. Labels are passage annotations, not proof of unique evidence.',
         'No examples are removed based on method performance. Automatic span checks are NOT a semantic human review.',
         'Data source: SQuAD 1.1, Rajpurkar et al. (2016), CC BY-SA 4.0.', '']
    by_id = {r['case_id']: r for r in labels}
    seen = set()
    for row in inputs:
        label = by_id[row['case_id']]
        if row['document_id'] not in seen:
            s += ['='*70, row['document_id'] + ' | ' + label['source_article'], row['document'], '']
            target = label['target_question_reference']
            s += ['TARGET: ' + target['question'], 'REF: ' + ' | '.join(a['text'] for a in target['answers'])]
            seen.add(row['document_id'])
        first = label['first_question_reference']
        s += [label['condition'].upper() + ' FIRST: ' + first['question'],
              'REF: ' + ' | '.join(a['text'] for a in first['answers']), '']
    return '\n'.join(s) + '\n'


def prepare_experiment(model: Any, tokenizer: Any, project_dir: str | Path = '/mnt/gstar-project',
                       n_dev: int = 20, n_heldout: int = 20) -> str:
    """Prepare fixed, article-disjoint input splits. Does NOT perform GPU inference."""
    require(1 <= n_dev <= 60 and 1 <= n_heldout <= 60, 'Choose 1..60 documents per split.')
    project = Path(project_dir).expanduser().resolve()
    require(project.is_dir(), f'Project Volume missing: {project}')
    tools, hashes = load_tools(project)
    runtime = runtime_fingerprint(model, tokenizer)
    _, old_data, old_smoke_meta = tools['paired']['discover_source'](project, None)
    require(old_smoke_meta['model_id'] == runtime['model_id'], 'Model differs from successful pilot.')
    old_rev = old_smoke_meta.get('model_revision')
    require(not old_rev or old_rev == runtime['model_revision'], 'Model revision differs from successful pilot.')
    old_manifest = read_json(old_data / 'source_split.json')
    builder = tools['builder']
    # Prefer either existing cache. Only the fresh runpy namespace is changed.
    raw_candidates = [project/'gstar_fullkv_pilot/data/raw/train-v1.1.json', project/'data/raw/train-v1.1.json']
    raw = next((x for x in raw_candidates if x.is_file()), None)
    if raw is not None:
        payload = raw.read_bytes()
        source = builder['parse_source'](payload)
        source_sha = hashlib.sha256(payload).hexdigest()
    else:
        source, source_sha = builder['load_source']()
    require(source_sha == old_manifest['source_sha256'], 'Source JSON changed since the pilot. Do not silently change dataset.')
    dev_articles, test_articles = builder['split_articles'](source['data'])
    require([x['title'] for x in dev_articles] == old_manifest['development_article_titles'], 'Original development split changed.')
    require([x['title'] for x in test_articles] == old_manifest['heldout_article_titles'], 'Original held-out split changed.')
    logger.debug('%s', 'Preparing input-only article-disjoint splits; no GPU inference ...')
    dev, dev_screen = _sample_packs(dev_articles, n_dev, 'development', tokenizer, tools)
    heldout, test_screen = _sample_packs(test_articles, n_heldout, 'heldout', tokenizer, tools)
    dev_titles, test_titles = {p['source_article'] for p in dev}, {p['source_article'] for p in heldout}
    require(not dev_titles.intersection(test_titles), 'Article leakage.')
    # Original pilot examples remain development examples; never remove an inconvenient result.
    pilot_titles = set(old_manifest['pilot_article_titles'])
    require(pilot_titles.issubset(dev_titles), 'Requested dev size/length rule would drop original pilot articles; stop rather than silently exclude them.')
    builder['validate_packs'](dev, set(old_manifest['development_article_titles']), set(old_manifest['heldout_article_titles']))
    builder['validate_packs'](heldout, set(old_manifest['heldout_article_titles']), set(old_manifest['development_article_titles']))
    splits = {}
    for split, packs in [('development', dev), ('heldout', heldout)]:
        ins, labs = _export_split(packs, builder, split)
        splits[split] = {'inputs': ins, 'labels': labs}
    # Verify original development documents/questions/answers were not edited.
    old_inputs = {r['case_id']: r for r in read_jsonl(old_data/'selection_inputs.jsonl')}
    by_article_condition = {(r['source_article'], r['condition']): r for r in splits['development']['labels']}
    new_inputs = {r['case_id']: r for r in splits['development']['inputs']}
    for old in read_jsonl(old_data/'evaluation_only.jsonl'):
        new = by_article_condition[(old['source_article'], old['condition'])]
        require(old_inputs[old['case_id']]['document'] == new_inputs[new['case_id']]['document'], 'Original pilot document changed.')
        for key in ['first_question_reference', 'target_question_reference']:
            require(old[key] == new[key], 'Original pilot QA changed.')
    protocol = {
        'version': VERSION, 'script_sha256': digest(Path(__file__)), 'dependency_hashes': hashes,
        'runtime': runtime, 'system_prompt': SYSTEM_PROMPT, 'source_sha256': source_sha,
        'source_dataset': 'SQuAD 1.1 train; project-specific article split',
        'data_license': 'CC BY-SA 4.0', 'n_development': n_dev, 'n_heldout': n_heldout,
        'original_pilot_articles_retained_in_development': sorted(pilot_titles),
        'article_split_seed': old_manifest['seed'],
        'split_rule': old_manifest['article_split_rule'],
        'sampling_rule': 'Original builder, first eligible triple per article, original deterministic article order; input-length eligibility only.',
        'manual_semantic_review': False, 'semantic_ambiguities_not_automatically_resolved': True,
        'keep_fraction': 0.5, 'max_new_tokens': 64, 'max_scoring_input_tokens': MAX_SCORING_TOKENS,
        'lambda_grid': list(tools['paired']['LAMBDAS']), 'rho_grid': list(tools['paired']['RHOS']),
        'reserve_grid': list(tools['paired']['RESERVES']),
        'tuning_objective': 'Maximize min(mean same-target F1, mean different-target F1); ties: first F1, then grid order.',
        'score_definition': 'same single_chunk_full_causal_reconstruction_max proxy; NEW prompt for all methods',
        'physical_compaction': False, 'native_kvzip_reproduction': False,
        'data_content_hashes': {split: object_digest(value) for split, value in splits.items()},
        'document_ids': {split: list(dict.fromkeys(r['document_id'] for r in value['inputs'])) for split, value in splits.items()},
        'heldout_not_a_claim_of_pretraining_noncontamination': True,
    }
    identifier = object_digest(protocol)
    experiment = project/'experiments'/f'paired_span_v2_d{n_dev}_h{n_heldout}_{identifier[:12]}'
    if (experiment/'protocol.json').is_file():
        require(read_json(experiment/'protocol.json') == protocol, 'Experiment identity collision.')
        _validate_data(experiment, protocol)
        logger.info('%s %s', 'Existing experiment reused (completed documents will be skipped):', experiment)
        return str(experiment)
    require(not experiment.exists(), f'Incomplete preparation at {experiment}; preserve it and inspect before continuing.')
    experiment.mkdir(parents=True)
    for split, value in splits.items():
        folder = experiment/'data'/split
        save_jsonl(folder/'selection_inputs.jsonl', value['inputs'])
        save_jsonl(folder/'evaluation_only.jsonl', value['labels'])
        (folder/'preview.txt').write_text(_preview(value['inputs'], value['labels'], split), encoding='utf-8')
    save_json(experiment/'screening.json', {'development': dev_screen, 'heldout': test_screen})
    save_json(experiment/'source_split.json', old_manifest)
    code = experiment/'code'; code.mkdir()
    shutil.copy2(__file__, code/'gstar_expanded_eval.py')
    for name, relative in DEPENDENCIES.items():
        shutil.copy2(project/relative, code/(name + '.py'))
    save_json(experiment/'protocol.json', protocol)
    _validate_data(experiment, protocol)
    logger.info('%s', f'Prepared: development {n_dev} documents / {3*n_dev} unique QA; heldout {n_heldout} documents / {3*n_heldout} unique QA.')
    logger.debug('%s', 'Original pilot retained. No held-out answers generated. Semantic review NOT completed by code.')
    logger.debug('%s %s', 'Development preview:', experiment/'data/development/preview.txt')
    logger.debug('%s %s', 'Experiment:', experiment)
    return str(experiment)


def _validate_data(experiment: Path, protocol: dict) -> None:
    title_sets = {}
    for split in ['development', 'heldout']:
        ins = read_jsonl(experiment/'data'/split/'selection_inputs.jsonl')
        labs = read_jsonl(experiment/'data'/split/'evaluation_only.jsonl')
        require(object_digest({'inputs': ins, 'labels': labs}) == protocol['data_content_hashes'][split], f'{split} data changed after preparation.')
        title_sets[split] = {x['source_article'] for x in labs}
        require(all(x['project_split'] == split for x in labs), 'Split labels changed.')
        require(len(title_sets[split]) == protocol['n_development' if split == 'development' else 'n_heldout'], 'Not one document per article.')
    require(not title_sets['development'].intersection(title_sets['heldout']), 'Article overlap.')


def _groups(experiment: Path, split: str) -> list[dict]:
    ins = read_jsonl(experiment/'data'/split/'selection_inputs.jsonl')
    labs = {r['case_id']: r for r in read_jsonl(experiment/'data'/split/'evaluation_only.jsonl')}
    require(len(labs) == len(ins), 'Input/label count mismatch.')
    buckets = defaultdict(list)
    for row in ins:
        require(set(row) == {'case_id', 'document_id', 'document', 'first_question'}, 'Selection input has extra fields.')
        label = labs[row['case_id']]
        require(label['document_id'] == row['document_id'], 'Document ID mismatch.')
        require(label['first_question_reference']['question'] == row['first_question'], 'First question mismatch.')
        for key in ['first_question_reference', 'target_question_reference']:
            for answer in label[key]['answers']:
                start, text = answer['document_answer_start'], answer['text']
                require(row['document'][start:start+len(text)] == text, 'Broken answer span.')
        buckets[row['document_id']].append((row, label))
    groups = []
    for doc_id, cases in sorted(buckets.items()):
        cases.sort(key=lambda x: x[1]['condition'] != 'same')
        require(len(cases) == 2 and [x[1]['condition'] for x in cases] == ['same', 'different'], 'Bad paired cases.')
        a, b = cases
        require(a[0]['document'] == b[0]['document'], 'Paired documents differ.')
        require(a[1]['target_question_reference'] == b[1]['target_question_reference'], 'Paired targets differ.')
        require(a[0]['first_question'] != b[0]['first_question'], 'First questions duplicated.')
        require(a[1]['first_question_reference']['evidence_passage'] == a[1]['target_question_reference']['evidence_passage'], 'Invalid same label.')
        require(b[1]['first_question_reference']['evidence_passage'] != b[1]['target_question_reference']['evidence_passage'], 'Invalid different label.')
        groups.append({'document_id': doc_id, 'document': a[0]['document'], 'cases': cases})
    return groups


def _verified_document(folder: Path) -> dict | None:
    marker = folder/'done.json'
    if not marker.is_file():
        return None
    done = read_json(marker)
    require(done['status'] == 'COMPLETE', f'Invalid checkpoint: {marker}')
    attempt = folder/done['attempt']
    for name, sha in done['file_hashes'].items():
        require((attempt/name).is_file() and digest(attempt/name) == sha, f'Completed file missing/modified: {attempt/name}')
    return read_json(attempt/'result.json')


def _policies_for_split(bank: dict, frozen: dict | None) -> dict:
    policies = bank['policies']
    if frozen is None:
        return {k: dict(v, source_method=k) for k, v in policies.items()}
    mapping = {'full_kv': 'full_kv', 'a_only': 'a_only', 'g_proxy_only': 'g_proxy_only',
               'fixed_tuned': frozen['selected']['fixed'], 'reserve_tuned': frozen['selected']['reserve'],
               'constrained_tuned': frozen['selected']['constrained']}
    return {alias: dict(policies[source], source_method=source) for alias, source in mapping.items()}


def _run_document(model: Any, tokenizer: Any, group: dict, folder: Path,
                  split: str, tools: dict, frozen: dict | None) -> dict:
    paired, smoke, full = tools['paired'], tools['smoke'], tools['full']
    attempt = folder/f'attempt_{stamp()}'; attempt.mkdir(parents=True)
    document, doc_id = group['document'], group['document_id']
    rows, diagnostics, audits, banks = [], [], [], {}
    actual_calls, limit_hits = 0, 0
    t0 = time.perf_counter()
    try:
        layout = paired['make_layout'](tokenizer, render_prompt, document)
        g, gmeta = paired['extract_g'](model, tokenizer, render_prompt, layout, MAX_SCORING_TOKENS)
        scores, suffixes = {}, {}
        for selection, _label in group['cases']:
            cid = selection['case_id']
            a, suffix, positions = paired['extract_a'](model, tokenizer, render_prompt, layout, selection['first_question'], MAX_SCORING_TOKENS)
            banks[cid] = paired['build_policy_bank'](a, g, layout['document_positions'], layout['prefix_length'], 0.5, smoke)
            suffixes[cid] = suffix.to(model.device)
            scores[cid] = {'a_raw': a, 'question_positions': positions, 'first_question': selection['first_question']}
            # In held-out mode, save ONLY evaluated policies; no held-out grid scoring.
            policies = _policies_for_split(banks[cid], frozen)
            torch.save({'case_id': cid, 'budget': banks[cid]['budget'], 'policies': policies,
                        'free_prefix_positions': banks[cid]['free_prefix_positions'],
                        'normalized_a_free': banks[cid]['normalized_a_free'],
                        'normalized_g_free': banks[cid]['normalized_g_free']}, attempt/f'{cid}_selection.pt')
        torch.save({'layout': layout, 'g_raw_proxy': g, 'g_metadata': gmeta, 'cases': scores}, attempt/'scores.pt')
        p = layout['prefix_length']
        with torch.no_grad():
            ids = layout['prefix_ids'].to(model.device)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1, return_dict=True)
        base = output.past_key_values; del output
        snapshot = smoke['prefix_cache_snapshot'](base)
        first_cid = group['cases'][0][0]['case_id']
        tr0, cont, _ = smoke['fixed_trace'](model, suffixes[first_cid], base, None)
        allkeep = torch.ones_like(banks[first_cid]['policies']['a_only']['keep'])
        tr1, _, audit = smoke['fixed_trace'](model, suffixes[first_cid], base, allkeep, cont)
        comp = smoke['compare_traces'](tr0, tr1)
        require(comp['exactly_equal'], 'All-keep equivalence failed.')
        audits.append({'test': 'all_keep', 'comparison': comp, 'audit': audit})
        with (attempt/'predictions.jsonl').open('x', encoding='utf-8') as pred:
            for selection, label in group['cases']:
                cid = selection['case_id']; bank = banks[cid]
                policies = _policies_for_split(bank, frozen)
                adaptive = 'constrained_rho_0.90' if frozen is None else frozen['selected']['constrained']
                keep = bank['policies'][adaptive]['keep']
                clean, continuation, audit1 = smoke['fixed_trace'](model, suffixes[cid], base, keep)
                poisoned = smoke['poison_excluded'](base, keep)
                changed, _, audit2 = smoke['fixed_trace'](model, suffixes[cid], poisoned, keep, continuation)
                del poisoned
                comp = smoke['compare_traces'](clean, changed)
                require(comp['exactly_equal'], 'Excluded KV perturbation changed logits.')
                audits.append({'case_id': cid, 'test': 'excluded_kv', 'policy': adaptive,
                               'comparison': comp, 'clean_audit': audit1, 'perturbed_audit': audit2})
                require(smoke['cache_intact'](base, snapshot, p), 'Audit modified pristine KV.')
                questions = []
                for role, key in [('first', 'first_question_reference'), ('target', 'target_question_reference')]:
                    ref = label[key]
                    ids = paired['tokenized'](tokenizer, render_prompt(tokenizer, document, ref['question']))['input_ids']
                    require(torch.equal(ids[:, :p], layout['prefix_ids']), 'Evaluation prefix changed.')
                    require(p < ids.shape[1] <= MAX_ANSWER_INPUT_TOKENS, 'Evaluation input too long; no truncation.')
                    questions.append((role, ref, ids[:, p:].to(model.device)))
                memo = {}  # Deliberately independent across paired cases.
                for method, policy in policies.items():
                    diagnostics.append({'case_id': cid, 'document_id': doc_id, 'condition': label['condition'],
                                        'method': method, 'source_method': policy['source_method'],
                                        'family': policy['family'], 'parameter': policy['parameter'],
                                        'mask_key': policy['mask_key'], 'diagnostics': policy['diagnostics']})
                    for role, ref, suffix in questions:
                        key = (policy['mask_key'], tuple(suffix.cpu().flatten().tolist()))
                        row_id = f'{cid}/{method}/{role}'
                        reused_from = None
                        if key in memo:
                            reused_from, result = memo[key]
                        else:
                            result = smoke['generate_answer'](model, tokenizer, suffix, base, policy['keep'], 64)
                            memo[key] = (row_id, result)
                            actual_calls += 1
                            limit_hits += int(result['hit_token_limit'])
                        intact = smoke['cache_intact'](base, snapshot, p)
                        require(intact, 'Answer generation modified pristine KV.')
                        gold = [x['text'] for x in ref['answers']]
                        metrics = full['score_answer'](result['answer'], gold)
                        row = {'record_id': row_id, 'document_id': doc_id, 'case_id': cid,
                               'project_split': split, 'source_article': label['source_article'],
                               'condition': label['condition'], 'role': role, 'method': method,
                               'family': policy['family'], 'parameter': policy['parameter'],
                               'source_method': policy['source_method'], 'question_id': ref['question_id'],
                               'question': ref['question'], 'references': gold, **metrics,
                               'mask_key': policy['mask_key'], 'original_kv_unchanged': intact,
                               'prefix_tokens': p, 'kept_prefix_tokens': p if policy['keep'] is None else bank['budget']['kept_prefix_tokens_per_layer_kv_head'],
                               'reused_from_record': reused_from,
                               'answer_is_verbatim_context_substring': bool(result['answer']) and result['answer'] in document,
                               'answer_whitespace_words': len(result['answer'].split()), **result}
                        log_row(pred, row); rows.append(row)
        # No retained model/cache objects in a completed checkpoint.
        result = {'document_id': doc_id, 'source_article': group['cases'][0][1]['source_article'],
                  'rows': rows, 'diagnostics': diagnostics, 'audits': audits,
                  'actual_generation_calls': actual_calls, 'token_limit_hits': limit_hits,
                  'elapsed_seconds_including_audits_not_speed_benchmark': time.perf_counter()-t0}
        _check_pairs(rows)
        save_jsonl(attempt/'diagnostics.jsonl', diagnostics)
        save_json(attempt/'audits.json', audits)
        save_json(attempt/'result.json', result)
        files = {x.name: digest(x) for x in attempt.iterdir() if x.is_file()}
        save_json(folder/'done.json', {'status': 'COMPLETE', 'attempt': attempt.name, 'file_hashes': files})
        return result
    except BaseException as exc:
        save_json(attempt/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'partial_answer_records': len(rows)})
        raise


def _check_pairs(rows: list[dict]) -> list[dict]:
    pairs = defaultdict(list)
    for row in rows:
        if row['role'] == 'target':
            pairs[(row['document_id'], row['method'])].append(row)
    results = []
    for (doc, method), group in pairs.items():
        require(len(group) == 2, 'Missing/extra target pair.')
        group.sort(key=lambda r: r['condition'] != 'same')
        a, b = group
        require([a['condition'], b['condition']] == ['same', 'different'], 'Bad target conditions.')
        require(a['question_id'] == b['question_id'] and a['question'] == b['question'], 'Target changed between paired cases.')
        mask_equal = a['mask_key'] == b['mask_key']
        token_equal = a['generated_token_ids'] == b['generated_token_ids']
        if method in ['full_kv', 'g_proxy_only']:
            require(mask_equal and token_equal, 'Query-independent target invariant failed.')
        results.append({'document_id': doc, 'method': method, 'same_f1': a['f1'], 'different_f1': b['f1'],
                        'same_minus_different_f1': a['f1']-b['f1'], 'selected_masks_identical': mask_equal,
                        'target_tokens_identical': token_equal})
    return results


def summarize(results: list[dict], split: str, planned_docs: int) -> dict:
    rows = [r for x in results for r in x['rows']]
    diag = [r for x in results for r in x['diagnostics']]
    methods, diagnostics = {}, {}
    ordered_methods = list(dict.fromkeys(r['method'] for r in rows))
    for method in ordered_methods:
        items = [r for r in rows if r['method'] == method]
        groups = {'first': [r for r in items if r['role'] == 'first'],
                  'same': [r for r in items if r['role'] == 'target' and r['condition'] == 'same'],
                  'different': [r for r in items if r['role'] == 'target' and r['condition'] == 'different']}
        require(all(len(v) == len(results)*(2 if k == 'first' else 1) for k, v in groups.items()), 'Metric group count mismatch.')
        stats = {k: {'n': len(v), 'f1': sum(r['f1'] for r in v)/len(v), 'em': sum(r['em'] for r in v)/len(v)} for k, v in groups.items()}
        stats.update(worst_target_f1=min(stats['same']['f1'], stats['different']['f1']),
                     family=items[0]['family'], parameter=items[0]['parameter'], source_method=items[0]['source_method'],
                     verbatim_context_substring_fraction=sum(r['answer_is_verbatim_context_substring'] for r in items)/len(items))
        methods[method] = stats
        ds = [r['diagnostics'] for r in diag if r['method'] == method and r['diagnostics'] is not None]
        if ds:
            count = sum(d['head_count'] for d in ds)
            hist = Counter()
            for d in ds:
                hist.update(d['lambda_histogram'] or {})
            diagnostics[method] = {
                'layer_head_decisions': count, 'lambda_histogram': dict(hist) or None,
                'g_retention_min': min(d['g_retention_min'] for d in ds),
                'g_retention_mean': sum(d['g_retention_mean']*d['head_count'] for d in ds)/count,
                'g_only_identical_head_fraction': sum(d['g_only_identical_head_fraction']*d['head_count'] for d in ds)/count,
                'whole_cache_g_only_cases': sum(d['whole_cache_equals_g_only'] for d in ds),
            }
    completed = len(results) == planned_docs
    winners = None
    if split == 'development' and completed:
        winners = {}
        for family in ['fixed', 'reserve', 'constrained']:
            names = [k for k, v in methods.items() if v['family'] == family]
            winners[family] = max(names, key=lambda k: (methods[k]['worst_target_f1'], methods[k]['first']['f1']))
    return {'status': 'COMPLETE' if completed else 'INCOMPLETE', 'split': split,
            'documents_completed': len(results), 'documents_planned': planned_docs,
            'paired_cases': 2*len(results), 'unique_questions': len({r['question_id'] for r in rows}),
            'answer_records': len(rows), 'actual_generation_calls': sum(x['actual_generation_calls'] for x in results),
            'token_limit_hits': sum(x['token_limit_hits'] for x in results),
            'original_kv_unchanged': all(r['original_kv_unchanged'] for r in rows),
            'methods': methods, 'diagnostics': diagnostics, 'selected_on_development': winners,
            'paired_targets': _check_pairs(rows), 'physical_compaction': False, 'native_kvzip_reproduction': False,
            'semantic_human_review': 'not performed automatically',
            'inference_calls_are_not_independent_statistical_samples': True}


def _write_summary(folder: Path, summary: dict, results: list[dict]) -> None:
    save_json(folder/'summary.json', summary)
    save_jsonl(folder/'predictions.jsonl', [r for x in results for r in x['rows']])
    save_jsonl(folder/'diagnostics.jsonl', [r for x in results for r in x['diagnostics']])
    save_json(folder/'paired_targets.json', summary['paired_targets'])
    with (folder/'comparison.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['method', 'first_f1', 'same_f1', 'different_f1', 'worst_target_f1'])
        for k, v in summary['methods'].items():
            writer.writerow([k, v['first']['f1'], v['same']['f1'], v['different']['f1'], v['worst_target_f1']])


def _print_summary(summary: dict, folder: Path) -> None:
    logger.debug('%s', f"\n========== EXPANDED {summary['split'].upper()} SUMMARY ==========")
    logger.debug('%s', f"Status: {summary['status']} | documents: {summary['documents_completed']}/{summary['documents_planned']} | unique QA: {summary['unique_questions']}")
    if summary['split'] == 'development':
        logger.debug('%s', 'TUNING DATA: not a held-out performance estimate.')
    logger.debug('%s', f"{'Method':25s} {'First':>8s} {'Same':>8s} {'Diff':>8s} {'Worst':>8s}")
    for k, v in summary['methods'].items():
        logger.debug('%s', f"{k:25s} {v['first']['f1']:8.2f} {v['same']['f1']:8.2f} {v['different']['f1']:8.2f} {v['worst_target_f1']:8.2f}")
    logger.debug('%s %s', 'Development-selected settings:', summary['selected_on_development'])
    for k, d in summary['diagnostics'].items():
        if d['lambda_histogram'] is not None:
            logger.debug('%s', f"{k}: retention min/mean={d['g_retention_min']:.4f}/{d['g_retention_mean']:.4f}; g-only heads={d['g_only_identical_head_fraction']:.1%}; lambda counts={d['lambda_histogram']}")
    logger.debug('%s %s %s %s', 'Actual generation calls:', summary['actual_generation_calls'], '| token-limit hits:', summary['token_limit_hits'])
    logger.debug('%s', 'Masking only; reconstruction proxy; no native KVzip or speed/memory claim.')
    logger.info('%s %s', 'Results:', folder)


def _validate_protocol(experiment: Path, project: Path, hashes: dict, model: Any, tokenizer: Any) -> dict:
    protocol = read_json(experiment/'protocol.json')
    require(protocol['version'] == VERSION and protocol['script_sha256'] == digest(Path(__file__)), 'Expanded runner changed; use a new experiment rather than overwrite results.')
    require(protocol['dependency_hashes'] == hashes, 'Existing helper code changed; do not mix runs.')
    require(protocol['system_prompt'] == SYSTEM_PROMPT, 'Prompt changed.')
    require(protocol['runtime'] == runtime_fingerprint(model, tokenizer), 'Model/tokenizer/runtime changed; do not merge checkpoints.')
    _validate_data(experiment, protocol)
    return protocol


def _run_split(model: Any, tokenizer: Any, experiment_dir: str | Path,
               project_dir: str | Path, split: str, max_new_documents: int | None) -> dict:
    project, experiment = Path(project_dir).resolve(), Path(experiment_dir).resolve()
    tools, hashes = load_tools(project)
    protocol = _validate_protocol(experiment, project, hashes, model, tokenizer)
    require(max_new_documents is None or max_new_documents >= 1, 'max_new_documents must be positive or None.')
    frozen = None
    if split == 'heldout':
        path = experiment/'frozen_protocol.json'
        require(path.is_file(), 'Run complete development, then call freeze_protocol BEFORE held-out evaluation.')
        frozen = read_json(path)
        require(frozen['protocol_sha256'] == digest(experiment/'protocol.json'), 'Frozen protocol changed.')
        require(frozen['development_summary_sha256'] == digest(experiment/'runs/development/summary.json'), 'Development report changed after freeze.')
    groups = _groups(experiment, split)
    folder = experiment/'runs'/split; folder.mkdir(parents=True, exist_ok=True)
    planned = len(groups)
    smoke = tools['smoke']
    old_backend, old_training = model.config._attn_implementation, model.training
    old_hooks = smoke['hook_counts'](model)
    results, newly_completed = [], 0
    with run_lock(experiment/'.run.lock'):
        # A marker prevents silently changing the chosen settings after any test invocation.
        if split == 'heldout' and not (experiment/'heldout_started.json').exists():
            save_json(experiment/'heldout_started.json', {'started_utc': stamp(), 'frozen_sha256': digest(experiment/'frozen_protocol.json')})
        if split == 'heldout':
            require(read_json(experiment/'heldout_started.json')['frozen_sha256'] == digest(experiment/'frozen_protocol.json'), 'Frozen settings changed after held-out started.')
        try:
            model.eval(); model.set_attn_implementation('eager')
            for index, group in enumerate(groups, 1):
                doc_folder = folder/group['document_id']
                cached = _verified_document(doc_folder)
                if cached is not None:
                    results.append(cached)
                    logger.debug('%s', f"[{index}/{planned}] {group['document_id']}: verified checkpoint; skipping GPU work.")
                    continue
                if max_new_documents is not None and newly_completed >= max_new_documents:
                    continue
                require(not (split == 'development' and (experiment/'frozen_protocol.json').exists()), 'Cannot add development results after protocol freeze.')
                logger.debug('%s', f"[{index}/{planned}] {group['document_id']}: scoring, masking audits, paired generation ...")
                result = _run_document(model, tokenizer, group, doc_folder, split, tools, frozen)
                results.append(result); newly_completed += 1
                partial = summarize(results, split, planned)
                _write_summary(folder, partial, results)
                logger.info('%s', f"  SAVED: {group['document_id']} | answer records={len(result['rows'])} | actual generations={result['actual_generation_calls']}")
            summary = summarize(results, split, planned)
            _write_summary(folder, summary, results)
        except BaseException as exc:
            save_json(folder/f'failure_{stamp()}.json', {'error': str(exc), 'traceback': traceback.format_exc(),
                                                       'completed_documents_this_scan': len(results)})
            logger.error('%s', 'Stopped; completed document checkpoints are preserved. Re-run the same call to resume.')
            raise
        finally:
            model.set_attn_implementation(old_backend); model.train(old_training)
            restored = {'backend_after': model.config._attn_implementation,
                        'attention_backend_restored': model.config._attn_implementation == old_backend,
                        'hook_counts_restored': smoke['hook_counts'](model) == old_hooks}
            save_json(folder/'restoration.json', restored)
            require(restored['attention_backend_restored'] and restored['hook_counts_restored'], 'Model settings not restored; inspect restoration.json.')
    _print_summary(summary, folder)
    return summary


def run_development(model: Any, tokenizer: Any, experiment_dir: str | Path,
                    project_dir: str | Path = '/mnt/gstar-project', max_new_documents: int | None = None) -> dict:
    return _run_split(model, tokenizer, experiment_dir, project_dir, 'development', max_new_documents)


def freeze_protocol(experiment_dir: str | Path) -> dict:
    """EXPLICIT stage, never auto-called. Requires all planned development docs."""
    experiment = Path(experiment_dir).resolve()
    with run_lock(experiment/'.run.lock'):
        protocol = read_json(experiment/'protocol.json')
        summary_path = experiment/'runs/development/summary.json'
        summary = read_json(summary_path)
        require(summary['status'] == 'COMPLETE' and summary['documents_completed'] == protocol['n_development'], 'Finish ALL development documents before freezing.')
        require(summary['selected_on_development'], 'No development selections.')
        for doc_id in protocol['document_ids']['development']:
            require(_verified_document(experiment/'runs/development'/doc_id) is not None, 'Missing development checkpoint.')
        value = {'protocol_sha256': digest(experiment/'protocol.json'), 'development_summary_sha256': digest(summary_path),
                 'selected': summary['selected_on_development'], 'tuning_objective': protocol['tuning_objective'],
                 'heldout_metrics_used': False}
        destination = experiment/'frozen_protocol.json'
        if destination.is_file():
            require(read_json(destination) == value, 'A different protocol is already frozen; do not overwrite it.')
        else:
            require(not (experiment/'heldout_started.json').exists(), 'Held-out already started; cannot reselect.')
            save_json(destination, value)
        logger.info('%s %s', 'Frozen from development ONLY:', value['selected'])
        logger.debug('%s', 'Held-out has not been evaluated by this function.')
        return value


def run_heldout(model: Any, tokenizer: Any, experiment_dir: str | Path,
                project_dir: str | Path = '/mnt/gstar-project', max_new_documents: int | None = None) -> dict:
    """Only Full/a/g and THREE dev-frozen methods. No held-out hyperparameter search."""
    return _run_split(model, tokenizer, experiment_dir, project_dir, 'heldout', max_new_documents)


def export_review(experiment_dir: str | Path, split: str = 'development') -> str:
    """Export summaries + per-question records, not large tensors; NO model calls."""
    require(split in ['development', 'heldout'], 'Unknown split.')
    experiment = Path(experiment_dir).resolve()
    folder = experiment/'runs'/split
    require((folder/'summary.json').is_file(), 'No run summary yet.')
    archive = experiment/f'{split}_review_{stamp()}.zip'
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as z:
        for path in [experiment/'protocol.json', experiment/'screening.json', experiment/'frozen_protocol.json',
                     folder/'summary.json', folder/'predictions.jsonl', folder/'diagnostics.jsonl',
                     folder/'paired_targets.json', folder/'comparison.csv', folder/'restoration.json',
                     experiment/'data'/split/'selection_inputs.jsonl', experiment/'data'/split/'evaluation_only.jsonl']:
            if path.is_file():
                z.write(path, arcname=str(path.relative_to(experiment)))
    logger.info('%s %s', 'Review ZIP:', archive)
    return str(archive)
