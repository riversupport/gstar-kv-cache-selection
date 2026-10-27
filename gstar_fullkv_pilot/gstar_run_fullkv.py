"""Small development-only SQuAD Full-KV pilot for an ALREADY LOADED model.

Notebook use:
    result = run_fullkv_pilot(model, tokenizer, project_dir='/mnt/gstar-project')

No pip calls, no weight downloads, no training, no compression.
Compatible target: Transformers 4.56.0, Qwen2.5-3B-Instruct, one GPU.
The sibling data builder is the user's original gstar_prepare_pilot.py.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import string
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger("gstar")

SYSTEM_PROMPT = (
    'Answer the question using only the provided context. '
    'Return only the answer, without explanation.'
)
DATA_FILES = ('selection_inputs.jsonl', 'evaluation_only.jsonl', 'source_split.json', 'preview.txt')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_data(project_dir: Path, data_dir: str | Path | None = None) -> Path:
    """Use existing project pilot files, or recreate with the original builder."""
    if data_dir is not None:
        chosen = Path(data_dir).expanduser().resolve()
        require(all((chosen / f).is_file() for f in DATA_FILES),
                f'Pilot folder is incomplete: {chosen}. Do not delete existing data.')
        return chosen

    candidates = [project_dir / 'data' / 'pilot', Path(__file__).resolve().parent / 'data' / 'pilot']
    for candidate in candidates:
        present = [(candidate / f).is_file() for f in DATA_FILES]
        if all(present):
            logger.debug('%s', f'Using existing pilot data: {candidate}')
            return candidate
        if any(present):
            raise ValueError(f'Pilot files are incomplete: {candidate}. Inspect the existing files without deleting them.')

    builder = _load_module(Path(__file__).with_name('gstar_prepare_pilot.py'), '_gstar_pilot_builder')
    logger.debug('%s', 'Preparing the short development pilot with the existing data builder.')
    builder.self_test()
    source, source_sha = builder.load_source()
    dev, heldout = builder.split_articles(source['data'])
    packs = builder.build_packs(dev, builder.PILOT_PACKS)
    builder.validate_packs(packs, {x['title'] for x in dev}, {x['title'] for x in heldout})
    builder.save_outputs(packs, dev, heldout, source_sha)
    logger.info('%s', f'Data validation: PASS | {len(packs)} documents | held-out exported: 0')
    return Path(builder.OUT_DIR)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_groups(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit the separation of selection inputs and evaluation-only references."""
    inputs = read_jsonl(data_dir / 'selection_inputs.jsonl')
    labels = read_jsonl(data_dir / 'evaluation_only.jsonl')
    manifest = json.loads((data_dir / 'source_split.json').read_text(encoding='utf-8'))
    require(manifest.get('heldout_examples_exported') == 0, 'Expected a DEVELOPMENT-only pilot.')
    dev = set(manifest['development_article_titles'])
    heldout = set(manifest['heldout_article_titles'])
    require(not (dev & heldout), 'Development / held-out article overlap.')
    labels_by_id = {r['case_id']: r for r in labels}
    require(len(labels_by_id) == len(labels), 'Duplicate evaluation case IDs.')
    require(len({r['case_id'] for r in inputs}) == len(inputs), 'Duplicate input case IDs.')
    require({r['case_id'] for r in inputs} == set(labels_by_id), 'Input / label case ID mismatch.')

    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in inputs:
        require(set(row) == {'case_id', 'document_id', 'document', 'first_question'},
                'Unexpected fields in selection input.')
        label = labels_by_id[row['case_id']]
        require(row['document_id'] == label['document_id'], 'Document ID mismatch.')
        require(label['project_split'] == 'development', 'Non-development case encountered.')
        require(label['source_article'] in dev and label['source_article'] not in heldout,
                'Non-development article encountered.')
        require(row['first_question'] == label['first_question_reference']['question'],
                'First-question text mismatch.')
        for key in ('first_question_reference', 'target_question_reference'):
            reference = label[key]
            require(bool(reference['answers']), 'No reference answers.')
            passage = next(p for p in label['passages']
                           if p['passage_number'] == reference['evidence_passage'])
            for a in reference['answers']:
                start, text = a['document_answer_start'], a['text']
                require(row['document'][start:start + len(text)] == text, 'Broken answer span.')
                require(passage['document_start'] <= start and start + len(text) <= passage['document_end'],
                        'Answer outside annotated passage.')
        grouped[row['document_id']].append((row, label))

    groups = []
    for doc_id, cases in sorted(grouped.items()):
        require(len(cases) == 2, 'Each document must have exactly two paired cases.')
        cases.sort(key=lambda pair: pair[1]['condition'] != 'same')
        (a, la), (b, lb) = cases
        require([la['condition'], lb['condition']] == ['same', 'different'], 'Bad condition labels.')
        require(a['document'] == b['document'], 'Paired documents differ.')
        require(la['source_article'] == lb['source_article'], 'Paired article labels differ.')
        require(la['target_question_reference'] == lb['target_question_reference'], 'Paired targets differ.')
        require(a['first_question'] != b['first_question'], 'First questions must differ.')
        refs = [la['first_question_reference'], lb['first_question_reference'], la['target_question_reference']]
        require(len({r['question_id'] for r in refs}) == 3, 'Repeated question IDs.')
        require(refs[0]['evidence_passage'] == refs[2]['evidence_passage'], 'Invalid same condition.')
        require(refs[1]['evidence_passage'] != refs[2]['evidence_passage'], 'Invalid different condition.')
        groups.append({'document_id': doc_id, 'document': a['document'], 'cases': cases})
    require(bool(groups), 'No pilot documents.')
    return groups, manifest


def normalize_answer(text: str) -> str:
    # SQuAD v1.1 convention: lowercase, ASCII punctuation, articles, whitespace.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def score_answer(prediction: str, references: list[str]) -> dict[str, float]:
    require(bool(references), 'Cannot score without reference answers.')
    pred = normalize_answer(prediction)
    pred_words = pred.split()
    best_em, best_f1 = 0.0, 0.0
    for reference in references:
        gold = normalize_answer(reference)
        gold_words = gold.split()
        best_em = max(best_em, float(pred == gold))
        overlap = sum((Counter(pred_words) & Counter(gold_words)).values())
        # Same zero-overlap convention as the official v1.1 scorer.
        f1 = 2.0 * overlap / (len(pred_words) + len(gold_words)) if overlap else 0.0
        best_f1 = max(best_f1, f1)
    return {'em': 100.0 * best_em, 'f1': 100.0 * best_f1}


def render_prompt(tokenizer: Any, document: str, question: str) -> str:
    # No reference answers or evidence labels enter the model prompt.
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Context:\n{document}\n\nQuestion:\n{question}'},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def cache_snapshot(cache: Any) -> list[tuple[Any, Any]]:
    # CPU snapshot of ALL populated key/value tensors, not merely sequence length.
    require(hasattr(cache, 'layers'), 'Expected a Transformers 4.56-style cache with .layers.')
    require(bool(cache.layers), 'Empty cache.')
    result = []
    for layer in cache.layers:
        require(layer.keys is not None and layer.values is not None, 'Uninitialized KV layer.')
        result.append((layer.keys.detach().to(device='cpu', copy=True),
                       layer.values.detach().to(device='cpu', copy=True)))
    return result


def cache_unchanged(cache: Any, snapshot: list[tuple[Any, Any]], length: int) -> bool:
    import torch
    if int(cache.get_seq_length()) != length or len(cache.layers) != len(snapshot):
        return False
    for layer, (keys, values) in zip(cache.layers, snapshot):
        if not torch.equal(layer.keys.detach().cpu(), keys):
            return False
        if not torch.equal(layer.values.detach().cpu(), values):
            return False
    return True


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(rows), 'Empty metric group.')
    return {'n': len(rows),
            'em': sum(r['reused_em'] for r in rows) / len(rows),
            'f1': sum(r['reused_f1'] for r in rows) / len(rows)}


def run_fullkv_pilot(
    model: Any,
    tokenizer: Any,
    project_dir: str | Path = '/mnt/gstar-project',
    data_dir: str | Path | None = None,
    max_documents: int = 3,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Run the pilot using existing model/tokenizer objects, without reloading them."""
    import torch
    import transformers
    from transformers import GenerationConfig

    project = Path(project_dir).expanduser().resolve()
    require(project.is_dir(), f'Project Volume not found: {project}')
    require(max_documents > 0 and max_new_tokens > 0, 'Limits must be positive.')
    model.eval()
    device = model.device
    data_path = prepare_data(project, data_dir)
    groups, manifest = load_groups(data_path)
    groups = groups[:max_documents]
    n_evaluations = 4 * len(groups)

    # Exclusive, timestamped output directory: never overwrite previous runs.
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = project / 'results' / f'fullkv_pilot_{run_id}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data_snapshot').mkdir()
    for name in DATA_FILES:
        shutil.copy2(data_path / name, output / 'data_snapshot' / name)
    shutil.copy2(Path(__file__), output / 'runner_snapshot.py')

    config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.eos_token_id
    eos = config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else []))
    metadata = {
        'purpose': 'development-only short Full-KV correctness pilot; not a benchmark score',
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'device': str(device), 'dtype': str(model.dtype),
        'model_id': getattr(model.config, '_name_or_path', None),
        'model_revision': getattr(model.config, '_commit_hash', None),
        'tokenizer_revision': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'),
        'attention_backend': getattr(model.config, '_attn_implementation', None),
        'system_prompt': SYSTEM_PROMPT, 'generation_config': config.to_dict(),
        'data_directory': str(data_path),
        'data_sha256': {name: hashlib.sha256((data_path / name).read_bytes()).hexdigest()
                        for name in DATA_FILES},
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'document_ids': [g['document_id'] for g in groups],
        'manual_review_status': 'not asserted by this runner; review data/preview.txt',
        'no_compression': True, 'score_scale': '0-100',
        'retained_prefix_fraction': 1.0,
        'timing_note': 'Diagnostic only. Copies, CPU KV checks, and fresh control runs are included.',
    }
    (output / 'environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    logger.debug('%s', f'\nRun: {len(groups)} documents / {n_evaluations} QA evaluations (fresh + reused each)')
    logger.debug('%s', f'Results directory: {output}')
    logger.debug('%s', 'Evaluate answer accuracy separately from cache-correctness checks.\n')
    rows = []
    start_time = time.perf_counter()
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as log:
        for group in groups:
            document = group['document']
            marker = '__GSTAR_QUESTION_SLOT__'
            template = render_prompt(tokenizer, document, marker)
            require(template.count(marker) == 1, 'Question marker is not unique.')
            prefix_text = template.split(marker, 1)[0]
            prefix_inputs = tokenizer(prefix_text, add_special_tokens=False, return_tensors='pt').to(device)
            prefix_length = prefix_inputs['input_ids'].shape[1]
            # A safety check, NOT truncation. Keep the complete document and answers.
            require(prefix_length <= 4096, 'This is a short pilot; prefix exceeds 4096 tokens. No truncation performed.')
            with torch.no_grad():
                pref = model(**prefix_inputs, use_cache=True, logits_to_keep=1)
            base = pref.past_key_values
            del pref
            require(int(base.get_seq_length()) == prefix_length, 'Prefill cache length mismatch.')
            snapshot = cache_snapshot(base)
            logger.debug('%s', f"[{group['document_id']}] prefix = {prefix_length} tokens")

            for selection, label in group['cases']:
                # Evaluate first and target independently. Never keep a previous answer.
                for role, ref_key in [('first', 'first_question_reference'), ('target', 'target_question_reference')]:
                    reference = label[ref_key]
                    question = reference['question']
                    full = tokenizer(render_prompt(tokenizer, document, question),
                                     add_special_tokens=False, return_tensors='pt').to(device)
                    require(torch.equal(full['input_ids'][:, :prefix_length], prefix_inputs['input_ids']),
                            'Prefix TOKEN boundary mismatch. Stop and inspect; do not retokenize cached positions silently.')
                    input_length = full['input_ids'].shape[1]
                    with torch.no_grad():
                        fresh_output = model.generate(**full, generation_config=config)
                        working_cache = copy.deepcopy(base)
                        reused_output = model.generate(**full, past_key_values=working_cache,
                                                       generation_config=config)
                    fresh_ids = fresh_output[0, input_length:].detach().cpu().tolist()
                    reused_ids = reused_output[0, input_length:].detach().cpu().tolist()
                    fresh_answer = tokenizer.decode(fresh_ids, skip_special_tokens=True).strip()
                    reused_answer = tokenizer.decode(reused_ids, skip_special_tokens=True).strip()
                    answers = [a['text'] for a in reference['answers']]
                    fresh_score = score_answer(fresh_answer, answers)
                    reused_score = score_answer(reused_answer, answers)
                    intact = cache_unchanged(base, snapshot, prefix_length)
                    row = {
                        'document_id': group['document_id'], 'case_id': selection['case_id'],
                        'project_split': 'development', 'condition': label['condition'], 'role': role,
                        'question_id': reference['question_id'], 'question': question,
                        'references': answers, 'prefix_tokens': prefix_length,
                        'input_tokens': input_length, 'fresh_answer': fresh_answer,
                        'reused_answer': reused_answer, 'fresh_token_ids': fresh_ids,
                        'reused_token_ids': reused_ids, 'tokens_match': fresh_ids == reused_ids,
                        'cache_unchanged': intact, 'fresh_em': fresh_score['em'], 'fresh_f1': fresh_score['f1'],
                        'reused_em': reused_score['em'], 'reused_f1': reused_score['f1'],
                        'fresh_hit_token_limit': len(fresh_ids) >= max_new_tokens and fresh_ids[-1] not in eos_ids,
                        'reused_hit_token_limit': len(reused_ids) >= max_new_tokens and reused_ids[-1] not in eos_ids,
                    }
                    rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
                    log.flush()
                    logger.debug('%s', f"  {len(rows):02d}/{n_evaluations} {label['condition']:9s}/{role:6s} "
                          f"tokens_match={row['tokens_match']} | KV_unchanged={intact} | "
                          f"EM={reused_score['em']:.1f} F1={reused_score['f1']:.1f}")
                    logger.debug('%s', f'    Q: {question}\n    A: {reused_answer}\n    Gold: {answers}')
                    del fresh_output, reused_output, working_cache, full
                    require(intact, f'Original KV values changed; partial results saved in {output}')
            del base, snapshot, prefix_inputs

    pair_checks = []
    for group in groups:
        targets = [r for r in rows if r['document_id'] == group['document_id'] and r['role'] == 'target']
        require(len(targets) == 2, 'Missing paired target results.')
        pair_checks.append({'document_id': group['document_id'],
                            'target_tokens_identical': targets[0]['reused_token_ids'] == targets[1]['reused_token_ids']})

    # Same target occurs twice by design; do NOT treat that as two independent questions.
    unique = {}
    for row in rows:
        unique.setdefault((row['document_id'], row['question_id']), row)
    summary = {
        'documents': len(groups), 'qa_evaluations': len(rows), 'unique_questions': len(unique),
        'matched_generation_count': sum(r['tokens_match'] for r in rows),
        'unchanged_cache_count': sum(r['cache_unchanged'] for r in rows),
        'same_different_target_match_count': sum(p['target_tokens_identical'] for p in pair_checks),
        'paired_target_checks': pair_checks,
        'first_questions': _metrics([r for r in rows if r['role'] == 'first']),
        'target_same': _metrics([r for r in rows if r['role'] == 'target' and r['condition'] == 'same']),
        'target_different': _metrics([r for r in rows if r['role'] == 'target' and r['condition'] == 'different']),
        'unique_question_scores': _metrics(list(unique.values())),
        'token_limit_hits': sum(r['fresh_hit_token_limit'] or r['reused_hit_token_limit'] for r in rows),
        'wall_seconds_including_checks': time.perf_counter() - start_time,
        'interpretation': 'Development diagnostic only. EM/F1 need not be 100. '
                          'A generation mismatch needs investigation, not automatic rejection: '
                          'BF16 computation order can change near-tied greedy decisions. '
                          'No compression, evidence-shift benefit, memory saving or speedup is established.',
        'output_dir': str(output),
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    logger.debug('%s', '\n========== FULL-KV PILOT SUMMARY ==========')
    logger.debug('%s', f"Documents: {summary['documents']} | Unique questions: {summary['unique_questions']}")
    logger.debug('%s', f"Fresh/reused token matches: {summary['matched_generation_count']}/{len(rows)}")
    logger.debug('%s', f"Original KV values unchanged: {summary['unchanged_cache_count']}/{len(rows)}")
    logger.debug('%s', f"Same/different target matches: {summary['same_different_target_match_count']}/{len(groups)}")
    for key in ['first_questions', 'target_same', 'target_different']:
        values = summary[key]
        logger.debug('%s', f"{key}: EM={values['em']:.2f}, F1={values['f1']:.2f} (n={values['n']})")
    logger.debug('%s', f"Generation token-limit hits: {summary['token_limit_hits']}")
    logger.info('%s', f'Results saved: {output}')
    if summary['matched_generation_count'] < len(rows):
        logger.warning('%s', 'Warning: fresh/reused outputs differ. Preserve the results and investigate.')
    return summary
